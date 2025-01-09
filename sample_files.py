import os
import random
import shutil
from pathlib import Path
import argparse

def sample_files(input_dir, output_dir, sample_size, random_seed=None):
    """
    Sample files from input directory and copy them to output directory.
    
    Args:
        input_dir (str): Path to input directory
        output_dir (str): Path to output directory
        sample_size (int): Number of files to sample
        random_seed (int, optional): Random seed for reproducibility
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files from input directory
    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # Check if sample size is valid
    if sample_size > len(all_files):
        print(f"Warning: Sample size ({sample_size}) is larger than total files ({len(all_files)})")
        sample_size = len(all_files)
    
    # Randomly sample files
    sampled_files = random.sample(all_files, sample_size)
    
    # Copy files to output directory preserving directory structure
    for src_path in sampled_files:
        # Get relative path
        rel_path = os.path.relpath(src_path, input_dir)
        # Create destination path
        dst_path = os.path.join(output_dir, rel_path)
        # Create necessary directories
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        # Copy file
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {rel_path}")

def main():
    parser = argparse.ArgumentParser(description="Sample files from input directory")
    parser.add_argument("input_dir", help="Input directory path")
    parser.add_argument("output_dir", help="Output directory path")
    parser.add_argument("sample_size", type=int, help="Number of files to sample")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility", default=None)
    
    args = parser.parse_args()
    
    sample_files(args.input_dir, args.output_dir, args.sample_size, args.seed)

if __name__ == "__main__":
    main()
