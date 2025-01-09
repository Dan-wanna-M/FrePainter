#!/usr/bin/env python3
import os
import torch
import torchaudio
from datetime import timedelta
from tqdm import tqdm
def get_audio_duration(directory):
    total_seconds = 0
    file_count = 0
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        for file in tqdm(files):
            if file.lower().endswith('.flac'):
                try:
                    # Get full file path
                    file_path = os.path.join(root, file)
                    
                    # Get metadata without loading the full audio
                    metadata = torchaudio.info(file_path)
                    
                    # Calculate duration in seconds
                    duration = metadata.num_frames / metadata.sample_rate
                    total_seconds += duration
                    file_count += 1
                    
                    # Print individual file duration
                    # print(f"{file}: {str(timedelta(seconds=int(duration)))}")
                    
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

    # Convert total seconds to hours, minutes, seconds
    total_duration = timedelta(seconds=int(total_seconds))
    
    print("\nSummary:")
    print(f"Total files: {file_count}")
    print(f"Total duration: {total_duration}")

if __name__ == "__main__":
    # Use current directory, or you can specify a path
    directory = "Emilia/ZH_48k"
    get_audio_duration(directory)
