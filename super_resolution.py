import argparse
from functools import partial
import glob
import json
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch.multiprocessing as mp
import torchaudio
from tqdm import tqdm
import torch
from torch.nn import functional as F
from audio_mae import AudioMaskedAutoencoderViT
from models_ft import SynthesizerTrn
from post_processing import PostProcessing
import torch.nn as nn
from torch.utils.data import DataLoader
import utils

torch.set_float32_matmul_precision('high')
# torch.backends.cudnn.benchmark = True
# Mel spectrogram constants
N_FFT = 2048
NUM_MELS = 128
SAMPLING_RATE = 24000
TARGET_SAMPLING_RATE = 48000
HOP_SIZE = 300
WIN_SIZE = 1200
FMIN = 20
FMAX = 12000

generator = None
dataloader = None
pp = None
rank = None


class AudioMelDataset:
    def __init__(self, input_files: list[str], hps: utils.HParams):
        self.input_files = input_files
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE,
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
                                                            power=1)
        self.hps = hps
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        input_file = self.input_files[index]
        audio, sampling_rate = torchaudio.load(input_file)
        assert audio.shape[0] == 1, "Only mono audio is supported"
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
        if sampling_rate != TARGET_SAMPLING_RATE:
            # src_audio = resample_poly(audio_cpu.numpy().squeeze(), 48000, sampling_rate)
            # src_audio = torch.from_numpy(src_audio).cuda(rank).unsqueeze(0)
            src_audio = torchaudio.functional.resample(
                waveform=audio,
                orig_freq=sampling_rate//2,
                new_freq=TARGET_SAMPLING_RATE//2,
            )
        else:
            src_audio = audio.clone()
        # loudness normalization
        input_audio = input_audio / torch.abs(input_audio).max() * 0.95
        src_audio = src_audio / torch.abs(src_audio).max() * 0.95
        # pad to the nearest multiple of HOP_SIZE
        if input_audio.size(-1) % HOP_SIZE != 0:
            input_audio = F.pad(input_audio, (0, HOP_SIZE - (input_audio.size(-1) % HOP_SIZE)), 'constant')
        # pad to the nearest multiple of input_audio.size(-1) * 2 for ease of post-processing
        if (input_audio.size(-1) * 2) > src_audio.size(-1):
            src_audio = F.pad(src_audio, (0, (input_audio.size(-1) * 2) - src_audio.size(-1)))
        elif (input_audio.size(-1) * 2) < src_audio.size(-1):
            src_audio = src_audio[:input_audio.size(-1) * 2]
        # manually padding for mel spectrogram
        input_audio = torch.nn.functional.pad(input_audio.unsqueeze(1), (int((N_FFT-HOP_SIZE)/2), int((N_FFT-HOP_SIZE)/2)), mode='reflect')
        input_audio = input_audio.squeeze(1)
        mels: torch.Tensor = self.mel_spectrogram(input_audio).squeeze(0)
        # needed for better performance
        mels = dynamic_range_compression_torch(mels)
        # split into timewise segments
        mel_segs = mels.split(self.hps.data.max_length, dim=-1)
        mel_padded = torch.zeros(len(mel_segs), mels.size(0), self.hps.data.max_length, device=mels.device)
        for i, seg in enumerate(mel_segs):
            mel_padded[i, :, :seg.size(-1)] = seg
        mel_lengths = torch.tensor([seg.size(-1) for seg in mel_segs], device=mels.device)
        audio_lengths = mel_lengths * self.hps.data.sampling_rate / 80 # WTF?
        return {
            'mel': mel_padded,
            'audio_lengths': audio_lengths,
            'src_audio': src_audio,
            'input_file': input_file
        }

def collate_fn(batch):
    return {
        'mel': torch.cat([item['mel'] for item in batch]),
        'audio_lengths': [item['audio_lengths'] for item in batch],
        'src_audio': [item['src_audio'] for item in batch],
        'input_file': [item['input_file'] for item in batch]
    }

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def init_rank(hps: utils.HParams, input_files: list[str], num_workers: int, input_rank: int, batch_size: int):
    process = mp.current_process()
    global generator, pp, dataloader, rank
    rank = input_rank
    dataset = AudioMelDataset(input_files, hps)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
    
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
    net_g.dec.remove_weight_norm()
    generator = net_g
    pp = PostProcessing(rank)
    print(f"Initialized worker {process.ident} for rank {rank}")

def execute_rank(hps: utils.HParams, input_files: list[str], num_workers: int, rank: int, output_dir: str, format: str, batch_size: int):
    init_rank(hps, input_files, num_workers, rank, batch_size)
    return super_resolution_audio(output_dir, format)


def super_resolution_audios(input_dir: str, output_dir: str,
                            format: str, num_workers: int, ranks: list[int], hps: utils.HParams, batch_size: int):
    input_files = glob.glob(os.path.join(input_dir, '**', f'*.{format}'), recursive=True)
    input_files_per_rank = []
    chunk_size = len(input_files)//len(ranks)
    for i in range(0, len(input_files), chunk_size):
        input_files_per_rank.append(input_files[i:i+chunk_size])
    os.makedirs(output_dir, exist_ok=True)
    processes = []
    for rank in ranks:
        ctx = mp.get_context('fork')
        p = ctx.Process(target=execute_rank, args=(hps, input_files_per_rank[rank], num_workers, rank, output_dir, format, batch_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def super_resolution_audio(output_dir: str, format: str):
    total_time_duration = 0
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for batch in tqdm(dataloader):
                mel_padded, audio_lengths, src_audio = batch['mel'], batch['audio_lengths'], batch['src_audio']
                input_files = batch['input_file']
                mel_padded = mel_padded.to(rank, non_blocking=True)
                audios_lengths = [audio_length.to(rank, non_blocking=True) for audio_length in audio_lengths]
                src_audios = [src_audio.to(rank, non_blocking=True) for src_audio in src_audio]
                y_hat = generator.infer(mel_padded).squeeze(1)
                patch_counter = 0
                for input_file, audio_lengths, src_audio in zip(input_files, audios_lengths, src_audios):
                    y_hat_total = y_hat[patch_counter:patch_counter+audio_lengths.size(0), :].reshape(1, -1)
                    patch_counter += audio_lengths.size(0)
                    y_hat_total = y_hat_total[:, :audio_lengths.sum().long()]
                    # y_hat_total = y_hat_total / torch.abs(y_hat_total).max() * 0.95, enabling this make validation worse
                    y_hat_pp = pp.post_processing(y_hat_total, src_audio, length=src_audio.size(-1))
                    torchaudio.save(os.path.join(output_dir, os.path.basename(input_file).replace(format, 'flac')), y_hat_pp.cpu(), TARGET_SAMPLING_RATE)
                    total_time_duration += y_hat_pp.size(-1) / TARGET_SAMPLING_RATE        
    print(f"Total time duration: {total_time_duration/3600} hours")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-f', '--format', type=str, default='wav')
    parser.add_argument('-m', '--model', type=str, default='logs/finetune/pt_rd_80_ft_ub_mrv2/best.pth')
    parser.add_argument('-n', '--num_workers', type=int, default=16)
    parser.add_argument('-r', '--ranks', type=int, nargs='+', default=[0])
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-c', '--config', type=str, default='logs/finetune/pt_rd_80_ft_ub_mrv2/config.json')
    args = parser.parse_args()
    hps = utils.HParams(**json.load(open(args.config)))
    hps.ckpt_file = args.model
    super_resolution_audios(args.input_dir, args.output_dir, args.format, args.num_workers, args.ranks, hps, args.batch_size)