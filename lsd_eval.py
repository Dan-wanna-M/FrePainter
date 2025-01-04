import glob
import os
import torch
import torchaudio
from tqdm import tqdm

def compute_log_spec_dist(ground_truth, generated, device, stft_transform):
    # Convert numpy arrays to torch tensors and move to GPU
    ground_truth = ground_truth.to(device)
    generated = generated.to(device)
    
    assert ground_truth.shape == generated.shape, f"Shape mismatch: {ground_truth.shape} and {generated.shape}"
    
    # loudness normalization
    ground_truth = ground_truth / torch.abs(ground_truth).max()
    generated = generated / torch.abs(generated).max()
    
    # Compute spectrograms
    lsd = stft_transform(ground_truth)
    lsd_generated = stft_transform(generated)
    
    # Swap axes to match original implementation
    lsd = torch.swapaxes(lsd, -2, -1)
    lsd_generated = torch.swapaxes(lsd_generated, -2, -1)
    
    # Compute log magnitude spectrograms
    lsd = torch.log(torch.abs(lsd) + 1e-9)
    lsd_generated = torch.log(torch.abs(lsd_generated) + 1e-9)
    
    # Compute MSE and final LSD
    mse = torch.mean((lsd - lsd_generated) ** 2, dim=-1)
    return torch.mean(torch.sqrt(mse)).cpu().item()

def compute_lsd_all(input_dir, output_dir, device='cuda'):
    input_files = glob.glob(os.path.join(input_dir, '*.wav'))
    mean_lsd = 0
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=2048,
        hop_length=300,
        win_length=1200,
        pad_mode='reflect',
        power=None  # Return complex tensor
    ).to(device)
    for input_file in tqdm(input_files):
        output_file = os.path.join(output_dir, os.path.basename(input_file)).replace(".wav", ".flac")
        ground_truth, g_sr = torchaudio.load(input_file)
        generated, ge_sr = torchaudio.load(output_file)
        # Ensure mono audio
        if ground_truth.size(0) > 1:
            ground_truth = torch.mean(ground_truth, dim=0)
        if generated.size(0) > 1:
            generated = torch.mean(generated, dim=0)
            
        assert g_sr == ge_sr, f"Sampling rate mismatch: {g_sr} and {ge_sr}"
        lsd = compute_log_spec_dist(ground_truth, generated, device, stft_transform)
        mean_lsd += lsd
    mean_lsd /= len(input_files)
    print(f"Mean LSD: {mean_lsd}")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_lsd_all("dataset/testset/gt_wavs/", "testset_24000", device)
