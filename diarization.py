import os
import json
import tempfile
import argparse
import shutil
import platform
import urllib.request
from pathlib import Path
from omegaconf import OmegaConf
import torch

# Try to import NVIDIA NeMo collections; provide clear instructions if missing
try:
    from nemo.collections.asr.models import ClusteringDiarizer
    from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
except ModuleNotFoundError:
    print("\nModule 'nemo' is not installed in the active environment.")
    print("Install a compatible PyTorch first (CPU example):")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("Then install NeMo (example, may require NVIDIA index):")
    print("  pip install nemo_toolkit[all] --extra-index-url https://pypi.ngc.nvidia.com")
    print("After installing, re-run this script.")
    raise


def download_config(output_dir: str, domain_type: str = "meeting"):
    """
    Download the official NeMo configuration file.
    
    Args:
        output_dir: Directory to store config file
        domain_type: Type of audio domain ('meeting' or 'telephonic')
    
    Returns:
        Path to the downloaded config file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config_file_name = f"diar_infer_{domain_type}.yaml"
    config_path = os.path.join(output_dir, config_file_name)
    
    # Download config if not exists
    if not os.path.exists(config_path):
        config_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{config_file_name}"
        print(f"Downloading configuration from: {config_url}")
        try:
            urllib.request.urlretrieve(config_url, config_path)
            print(f"Configuration downloaded to: {config_path}")
        except Exception as e:
            print(f"Error downloading config: {e}")
            raise
    else:
        print(f"Using existing configuration: {config_path}")
    
    return config_path


def setup_config(audio_file: str, output_dir: str, domain_type: str = "meeting", num_speakers: int = None):
    """
    Setup configuration for speaker diarization using official NeMo config.
    
    Args:
        audio_file: Path to the input audio file
        output_dir: Directory to store output files
        domain_type: Type of audio domain ('meeting' or 'telephonic')
        num_speakers: Number of speakers if known (optional)
    
    Returns:
        OmegaConf configuration object
    """
    # Download official config
    config_path = download_config(output_dir, domain_type)
    
    # Load configuration
    cfg = OmegaConf.load(config_path)
    
    # Create manifest path
    manifest_path = os.path.join(output_dir, 'manifest.json')
    
    # Update paths
    cfg.diarizer.manifest_filepath = manifest_path
    cfg.diarizer.out_dir = output_dir
    
    # Fix for Windows multiprocessing issues
    # Windows uses 'spawn' instead of 'fork', which requires all objects to be pickleable
    # Setting num_workers to 0 disables multiprocessing and avoids pickle errors
    if platform.system() == 'Windows':
        cfg.num_workers = 0
        print("Windows detected: Setting num_workers=0 to avoid multiprocessing issues")
    
    # Update config with known number of speakers if provided
    if num_speakers is not None:
        cfg.diarizer.clustering.parameters.oracle_num_speakers = True
        cfg.diarizer.clustering.parameters.max_num_speakers = num_speakers
        print(f"Using oracle mode with {num_speakers} speakers")
    
    return cfg


def diarize_audio(audio_filepath, out_dir, num_speakers=2, domain_type="meeting"):
    """
    Perform speaker diarization on an audio file using official NeMo configs.
    
    Args:
        audio_filepath (str): Path to the audio file to diarize
        out_dir (str): Output directory for diarization results
        num_speakers (int, optional): Number of speakers if known (default: 2)
        domain_type (str): Type of audio domain - 'meeting' or 'telephonic' (default: 'meeting')
    
    Returns:
        str: Content of the .rttm file containing diarization results
    """
    # Delete output directory if it exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"Deleted existing output directory: {out_dir}")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starting speaker diarization for: {audio_filepath}")
    print(f"Output directory: {out_dir}")
    print(f"Domain type: {domain_type}")
    
    # Create manifest file
    audio_path = Path(audio_filepath).resolve()
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_filepath}")
    
    manifest_path = Path(out_dir) / "manifest.json"
    
    # Create manifest entry
    manifest_entry = {
        "audio_filepath": str(audio_path),
        "offset": 0,
        "duration": None,  # Will be calculated automatically
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers,
        "rttm_filepath": None,
        "uem_filepath": None
    }
    
    # Write manifest file
    with open(manifest_path, 'w') as f:
        json.dump(manifest_entry, f)
        f.write('\n')
    
    print(f"Created manifest file: {manifest_path}")
    
    try:
        # Setup configuration using official NeMo config
        cfg = setup_config(audio_filepath, out_dir, domain_type, num_speakers)
        
        # Initialize the diarization model
        print("Initializing ClusteringDiarizer model...")
        print("Downloading pretrained models (this may take a while on first run)...")
        
        # Create diarizer
        diarizer = ClusteringDiarizer(cfg=cfg)
        
        # Run diarization
        print("Performing diarization...")
        diarizer.diarize()
        
        print(f"\n{'='*60}")
        print("Diarization complete!")
        print(f"{'='*60}")
        print(f"\nResults saved to: {out_dir}")
        
        # Print RTTM file location
        audio_basename = Path(audio_filepath).stem
        rttm_file = Path(out_dir) / "pred_rttms" / f"{audio_basename}.rttm"
        
        if rttm_file.exists():
            print(f"RTTM file: {rttm_file}")
            with open(rttm_file, 'r') as f:
                rttm_content = f.read()
            return rttm_content
        else:
            # Fallback: try to find any .rttm file
            rttm_dir = os.path.join(out_dir, 'pred_rttms')
            if os.path.exists(rttm_dir):
                rttm_files = [f for f in os.listdir(rttm_dir) if f.endswith('.rttm')]
                if rttm_files:
                    rttm_filepath = os.path.join(rttm_dir, rttm_files[0])
                    print(f"RTTM file: {rttm_filepath}")
                    with open(rttm_filepath, 'r') as f:
                        rttm_content = f.read()
                    return rttm_content
            
            raise FileNotFoundError("RTTM file not found. Check the output directory for results.")
        
    except Exception as e:
        print(f"\nError during diarization: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your audio file is in a supported format (WAV, FLAC, MP3)")
        print("2. Check that the audio file is not corrupted")
        print("3. Ensure you have sufficient disk space for model downloads")
        print("4. Try updating nemo_toolkit: pip install --upgrade nemo_toolkit[asr]")
        raise


# Example usage (can be commented out or removed when used as a module)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform speaker diarization on an audio file using official NeMo configs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diarization.py ./test.wav
  python diarization.py ./test.wav --out_dir results
  python diarization.py ./test.wav --num_speakers 3
  python diarization.py ./test.wav --domain_type telephonic
        """
    )
    parser.add_argument('audio_file', type=str, nargs='?', default='./test.wav',
                        help='Path to the audio file to diarize (default: ./test.wav)')
    parser.add_argument('--out_dir', type=str, default='./diarization_output',
                        help='Output directory for diarization results (default: ./diarization_output)')
    parser.add_argument('--num_speakers', type=int, default=2,
                        help='Number of speakers (default: 2)')
    parser.add_argument('--domain_type', type=str, choices=['meeting', 'telephonic'], default='meeting',
                        help='Audio domain type: "meeting" for general audio or "telephonic" for phone calls (default: meeting)')
    
    args = parser.parse_args()
    
    if os.path.exists(args.audio_file):
        rttm_content = diarize_audio(
            audio_filepath=args.audio_file,
            out_dir=args.out_dir,
            num_speakers=args.num_speakers,
            domain_type=args.domain_type
        )
        print("\n=== RTTM Content ===")
        print(rttm_content)
    else:
        print(f"Audio file not found: {args.audio_file}")
        print("Please provide a valid audio file path.")