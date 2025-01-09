import os
import glob
import random
import shutil
import torchaudio
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple

from tqdm import tqdm


@dataclass
class AudioInfo:
    filepath: str
    duration: float
    
    
def get_audio_info(filepath: str) -> AudioInfo:
    """Get duration info for a single audio file"""
    try:
        info = torchaudio.info(filepath)
        duration = info.num_frames / info.sample_rate
        return AudioInfo(filepath=filepath, duration=duration)
    except:
        return None


def collect_audio_infos(input_dir: str, format: str = 'wav', num_workers: int = 16) -> List[AudioInfo]:
    """Collect audio information using process pool"""
    audio_files = glob.glob(os.path.join(input_dir, '**', f'*.{format}'), recursive=True)
    
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=num_workers) as pool:
        audio_infos = list(filter(None, 
            tqdm(pool.imap(get_audio_info, audio_files), 
                total=len(audio_files),
                desc="Collecting audio info")))
        
    return audio_infos


def sample_hours(input_dir: str, output_dir: str, target_hours: float, 
                format: str = 'wav', num_workers: int = 16, seed: int = 42) -> Tuple[float, List[str]]:
    """
    Sample specified hours of audio from input directory and copy to output directory
    
    Args:
        input_dir: Input audio directory
        output_dir: Output directory to copy files to
        target_hours: Number of hours to sample
        format: Audio format to look for
        num_workers: Number of threads for parallel processing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (total_hours, list of sampled files)
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect audio information
    audio_infos = collect_audio_infos(input_dir, format, num_workers)
    
    # Randomly shuffle
    random.shuffle(audio_infos)
    
    total_hours = 0
    sampled_files = []
    
    # Keep adding files until we reach target hours
    for audio_info in tqdm(audio_infos, desc="Sampling audio files"):
        if total_hours >= target_hours:
            break
            
        hours = audio_info.duration / 3600
        total_hours += hours
        
        # Copy file to output directory
        dst = os.path.join(output_dir, os.path.basename(audio_info.filepath))
        shutil.copy2(audio_info.filepath, dst)
        sampled_files.append(audio_info.filepath)
        
    return total_hours, sampled_files


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True) 
    parser.add_argument('-t', '--target_hours', type=float, required=True)
    parser.add_argument('-f', '--format', type=str, default='flac')
    parser.add_argument('-n', '--num_workers', type=int, default=60)
    parser.add_argument('-s', '--seed', type=int, default=828)
    
    args = parser.parse_args()
    
    total_hours, sampled_files = sample_hours(
        args.input_dir,
        args.output_dir, 
        args.target_hours,
        args.format,
        args.num_workers,
        args.seed
    )
    
    print(f"Sampled {total_hours:.2f} hours of audio")
    print(f"Number of files: {len(sampled_files)}")
