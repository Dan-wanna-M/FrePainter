import glob
import os
import librosa
import numpy as np
from tqdm import tqdm

def compute_log_spec_dist(ground_truth, generated):
    assert ground_truth.shape == generated.shape, "Shape mismatch: {} and {}".format(ground_truth.shape, generated.shape)
    lsd = librosa.stft(ground_truth, n_fft=2048, hop_length=300, win_length=1200)
    lsd_generated = librosa.stft(generated, n_fft=2048, hop_length=300, win_length=1200)
    lsd = np.swapaxes(lsd, -2, -1)
    lsd_generated = np.swapaxes(lsd_generated, -2, -1)
    lsd = np.log(np.abs(lsd)+1e-9)
    lsd_generated = np.log(np.abs(lsd_generated)+1e-9)
    mse = np.mean((lsd - lsd_generated) ** 2, axis=-1)
    return np.mean(np.sqrt(mse))

def compute_lsd_all(input_dir, output_dir):
    input_files = glob.glob(os.path.join(input_dir, '*.wav'))
    mean_lsd = 0
    for input_file in tqdm(input_files):
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        ground_truth, g_sr = librosa.load(input_file, sr=None)
        generated, ge_sr = librosa.load(output_file, sr=None)
        assert g_sr == ge_sr, "Sampling rate mismatch: {} and {}".format(g_sr, ge_sr)
        lsd = compute_log_spec_dist(ground_truth, generated)
        mean_lsd += lsd
    mean_lsd /= len(input_files)
    print("Mean LSD: {}".format(mean_lsd))


if __name__ == "__main__":
    compute_lsd_all("dataset/OOD_test/gt_wavs", "logs/results/pt_rd_80_ft_ub_mrv2_best/24000")
