import argparse
from functools import partial
import glob
import json
import os
import torch.multiprocessing as mp
import torchaudio
from tqdm import tqdm
import torch
from torch.nn import functional as F
from audio_mae import AudioMaskedAutoencoderViT
from models_ft import SynthesizerTrn
from post_processing import PostProcessing
import torch.nn as nn
import utils
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')
# Mel spectrogram constants
N_FFT = 2048
NUM_MELS = 128
SAMPLING_RATE = 24000
HOP_SIZE = 300
WIN_SIZE = 1200
FMIN = 20
FMAX = 12000

generator = None
pp = None
mel_spectrogram = None
rank = None

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def init_worker(hps: utils.HParams, ranks: list[int]):
    process = mp.current_process()
    global rank
    rank = ranks[process.ident%len(ranks)]
    global generator, mel_spectrogram, pp
    hps = hps
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE,
                                                            n_fft=N_FFT,
                                                            win_length=WIN_SIZE,
                                                            hop_length=HOP_SIZE,
                                                            n_mels=NUM_MELS,
                                                            f_min=FMIN,
                                                            f_max=FMAX,
                                                            mel_scale="slaney",
                                                            norm="slaney",
                                                            normalized=False,
                                                            center=False,
                                                            pad_mode="reflect",
                                                            onesided=True,
                                                            power=1).cuda(rank)
    hm = hps.model_MAE
    pt_encoder = AudioMaskedAutoencoderViT(hm.num_mels, hm.mel_len, hm.patch_size, hm.in_chans,
                                           hm.embed_dim, hm.encoder_depth, hm.num_heads,
                                           hm.decoder_embed_dim, hm.decoder_depth, hm.decoder_num_heads,
                                           hm.mlp_ratio, hm.mask_token_valu, norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                           )
    net_g = SynthesizerTrn(
        hps.data.n_mel_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mask_ratio=0,
        **hps.model)
    utils.joint_model(net_g, pt_encoder)
    del pt_encoder
    utils.load_checkpoint(hps.ckpt_file, net_g, None)
    net_g.eval()
    net_g = net_g.cuda(rank)
    generator = net_g
    pp = PostProcessing(rank)
    print(f"Initialized worker {process.ident} for rank {rank}")

def super_resolution_audios(input_dir: str, output_dir: str, format: str, num_workers: int, ranks: list[int], hps: utils.HParams):
    input_files = glob.glob(os.path.join(input_dir, '**', f'*.{format}'), recursive=True)
    os.makedirs(output_dir, exist_ok=True)
    inputs = [(input_file, output_dir, format)
              for input_file in input_files]
    with mp.Pool(num_workers, initializer=init_worker, initargs=(hps,ranks)) as pool:
        for _ in tqdm(pool.imap(super_resolution_audio, inputs, chunksize=64), total=len(inputs)):
            pass

def super_resolution_audio(*args):
    with torch.no_grad():
        input_file, output_dir, format = args[0]
        audio, sampling_rate = torchaudio.load(input_file)
        assert audio.shape[0] == 1, "Only mono audio is supported"
        audio = audio.cuda(rank)
        # we assume that the silence is already removed
        # pad to the nearest multiple of HOP_LENGTH
        if sampling_rate != SAMPLING_RATE:
            # kaiser best by https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best
            input_audio = torchaudio.functional.resample(
                waveform=audio,
                orig_freq=sampling_rate//2,
                new_freq=SAMPLING_RATE//2,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
        else:
            input_audio = audio.clone()
        if sampling_rate != 48000:
            # src_audio = resample_poly(audio_cpu.numpy().squeeze(), 48000, sampling_rate)
            # src_audio = torch.from_numpy(src_audio).cuda(rank).unsqueeze(0)
            src_audio = torchaudio.functional.resample(
                waveform=audio,
                orig_freq=sampling_rate//2,
                new_freq=48000//2,
            )
        else:
            src_audio = audio.clone()
        # loudness normalization
        input_audio = input_audio / torch.abs(input_audio).max() * 0.95
        src_audio = src_audio / torch.abs(src_audio).max() * 0.95

        if input_audio.size(-1) % 300 != 0:
            input_audio = F.pad(input_audio, (0, 300 - (input_audio.size(-1) % 300)), 'constant')
    
        if (input_audio.size(-1) * 2) > src_audio.size(-1):
            src_audio = F.pad(src_audio, (0, (input_audio.size(-1) * 2) - src_audio.size(-1)))
        elif (input_audio.size(-1) * 2) < src_audio.size(-1):
            src_audio = src_audio[:input_audio.size(-1) * 2]
        input_audio = torch.nn.functional.pad(input_audio.unsqueeze(1), (int((N_FFT-HOP_SIZE)/2), int((N_FFT-HOP_SIZE)/2)), mode='reflect')
        input_audio = input_audio.squeeze(1)
        mels: torch.Tensor = mel_spectrogram(input_audio).squeeze(0)
        mels = dynamic_range_compression_torch(mels)
        # split into timewise segments
        mel_segs = mels.split(hps.data.max_length, dim=-1)
        mel_padded = torch.zeros(len(mel_segs), mels.size(0), hps.data.max_length, device=mels.device)
        for i, seg in enumerate(mel_segs):
            mel_padded[i, :, :seg.size(-1)] = seg
        mel_lengths = torch.tensor([seg.size(-1) for seg in mel_segs], device=mels.device)
        audio_lengths = mel_lengths * hps.data.sampling_rate / 80 # WTF?
        y_hat = generator.infer(mel_padded, mel_lengths).squeeze(1)
        y_hat_total = torch.cat(torch.chunk(y_hat, y_hat.size(0), dim=0), dim=1)
        y_hat_total = y_hat_total[:, :audio_lengths.sum().long()]
        y_hat_total = y_hat_total / torch.abs(y_hat_total).max() * 0.95
        y_hat_pp = pp.post_processing(y_hat_total, src_audio, length=src_audio.size(-1))
        torchaudio.save(os.path.join(output_dir, os.path.basename(input_file).replace(format, 'flac')), y_hat_pp.cpu(), 48000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-f', '--format', type=str, default='wav')
    parser.add_argument('-m', '--model', type=str, default='logs/finetune/pt_rd_80_ft_ub_mrv2/best.pth')
    parser.add_argument('-n', '--num_workers', type=int, default=1)
    parser.add_argument('-r', '--ranks', type=int, nargs='+', default=[0])
    parser.add_argument('-c', '--config', type=str, default='logs/finetune/pt_rd_80_ft_ub_mrv2/config.json')
    args = parser.parse_args()
    hps = utils.HParams(**json.load(open(args.config)))
    hps.ckpt_file = args.model
    super_resolution_audios(args.input_dir, args.output_dir, args.format, args.num_workers, args.ranks, hps)