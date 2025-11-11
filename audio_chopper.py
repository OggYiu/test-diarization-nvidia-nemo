import os
import glob
from pathlib import Path
import shutil
import wave
import numpy as np
from pydub import AudioSegment
import argparse



def read_all_rttm_files(rttm_folder):
    """
    Read all RTTM files from a folder.
    
    Args:
        rttm_folder: Path to folder containing RTTM files
        
    Returns:
        Dictionary mapping filename -> list of segments
    """
    all_segments = {}
    
    rttm_files = glob.glob(os.path.join(rttm_folder, "*.rttm"))
    
    for rttm_path in rttm_files:
        print(f"Reading RTTM file: {rttm_path}")
        segments = read_rttm_file(rttm_path)
        
        if segments:
            # Group by filename (in case multiple files are in same RTTM)
            filename = segments[0]['filename']
            if filename not in all_segments:
                all_segments[filename] = []
            all_segments[filename].extend(segments)
    
    # Sort segments by start time for each file
    for filename in all_segments:
        all_segments[filename].sort(key=lambda x: x['start'])
    
    return all_segments


def chop_all_audio_files(wav_folder, rttm_path, output_folder, filename=None, padding_ms=100):
    """
    Process audio files based on their RTTM diarization results.
    
    Args:
        wav_folder: Folder containing WAV files
        rttm_path: Path to RTTM file or folder containing RTTM files
        output_folder: Folder to save chopped audio files
        filename: Optional filename (without extension) to process. If None, processes all files.
                  Example: "201" will process "201.wav" and "201.rttm"
        padding_ms: Padding in milliseconds to add before/after each segment
    """
    # Detect if rttm_path is a file or folder
    is_rttm_file = os.path.isfile(rttm_path) and rttm_path.endswith('.rttm')
    
    # If rttm_path is a specific RTTM file, process it directly
    if is_rttm_file:
        print("=" * 60)
        print(f"Processing RTTM file: {rttm_path}")
        print("=" * 60)
        
        if not os.path.exists(rttm_path):
            print(f"Error: RTTM file not found: {rttm_path}")
            return
        
        print(f"Reading RTTM file: {rttm_path}")
        segments = read_rttm_file(rttm_path)
        
        if not segments:
            print("No segments found in RTTM file!")
            return
        
        print(f"Found {len(segments)} segments\n")
        
        # Extract filename from the RTTM segments or use provided filename
        if filename:
            wav_filename = filename
        else:
            # Use the filename from the first segment
            wav_filename = segments[0]['filename']
        
        # Find the corresponding WAV file
        wav_path = os.path.join(wav_folder, f"{wav_filename}.wav")
        
        # Clear output folder before writing chopped files
        if os.path.exists(output_folder):
            try:
                for child in Path(output_folder).iterdir():
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    elif child.is_dir():
                        shutil.rmtree(child)
            except Exception as e:
                print(f"Warning: failed to clear output folder '{output_folder}': {e}")
        else:
            os.makedirs(output_folder, exist_ok=True)

        chop_audio_file(wav_path, segments, output_folder, padding_ms)
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)
        return
    
    # Otherwise, treat rttm_path as a folder
    rttm_folder = rttm_path
    
    # If filename is specified, process only that file
    if filename:
        print("=" * 60)
        print(f"Processing single file: {filename}")
        print("=" * 60)
        
        # Read the specific RTTM file
        rttm_file_path = os.path.join(rttm_folder, f"{filename}.rttm")
        if not os.path.exists(rttm_file_path):
            print(f"Error: RTTM file not found: {rttm_file_path}")
            return
        
        print(f"Reading RTTM file: {rttm_file_path}")
        segments = read_rttm_file(rttm_file_path)
        
        if not segments:
            print("No segments found in RTTM file!")
            return
        
        print(f"Found {len(segments)} segments\n")
        
        # Find the corresponding WAV file
        wav_path = os.path.join(wav_folder, f"{filename}.wav")
        
        # Chop the audio
        # Clear output folder before writing chopped files
        if os.path.exists(output_folder):
            try:
                for child in Path(output_folder).iterdir():
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    elif child.is_dir():
                        shutil.rmtree(child)
            except Exception as e:
                print(f"Warning: failed to clear output folder '{output_folder}': {e}")
        else:
            os.makedirs(output_folder, exist_ok=True)

        chop_audio_file(wav_path, segments, output_folder, padding_ms)
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)
        return
    
    # Otherwise, process all files in folder
    print("=" * 60)
    print("Reading RTTM files...")
    print("=" * 60)
    all_segments = read_all_rttm_files(rttm_folder)
    
    if not all_segments:
        print("No segments found in RTTM files!")
        return
    
    print(f"\nFound diarization results for {len(all_segments)} file(s)\n")
    
    # Process each file
    print("=" * 60)
    print("Chopping audio files...")
    print("=" * 60)
    
    for filename, segments in all_segments.items():
        print(f"\nProcessing: {filename} ({len(segments)} segments)")
        
        # Find the corresponding WAV file
        wav_path = os.path.join(wav_folder, f"{filename}.wav")
        
        # Clear output folder before writing chopped files (only once)
        if os.path.exists(output_folder):
            try:
                for child in Path(output_folder).iterdir():
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    elif child.is_dir():
                        shutil.rmtree(child)
            except Exception as e:
                print(f"Warning: failed to clear output folder '{output_folder}': {e}")
        else:
            os.makedirs(output_folder, exist_ok=True)

        # Chop the audio
        chop_audio_file(wav_path, segments, output_folder, padding_ms)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Chop audio files based on RTTM diarization results"
    )
    parser.add_argument(
        "--rttm-path",
        type=str,
        default="demo/output/pred_rttms",
        help="Path to RTTM file or folder containing RTTM files (default: demo/output/pred_rttms)"
    )
    parser.add_argument(
        "--wav-folder",
        type=str,
        default="demo/phone_recordings",
        help="Path to folder containing WAV files (default: demo/phone_recordings)"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="demo/output/chopped_audios",
        help="Path to folder for saving chopped audio files (default: demo/output/chopped_audios)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="test",
        help="Filename (without extension) to process. Use 'all' to process all files (default: test)"
    )
    parser.add_argument(
        "--padding-ms",
        type=int,
        default=0,
        help="Padding in milliseconds to add before/after each segment (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Configuration from arguments
    RTTM_PATH = args.rttm_path
    WAV_FOLDER = args.wav_folder
    OUTPUT_FOLDER = args.output_folder
    PADDING_MS = args.padding_ms
    filename = None if args.filename.lower() == 'all' else args.filename
    
    # Process audio files
    chop_all_audio_files(
        wav_folder=WAV_FOLDER,
        rttm_path=RTTM_PATH,
        output_folder=OUTPUT_FOLDER,
        filename=filename,  # None means process all files
        padding_ms=PADDING_MS
    )