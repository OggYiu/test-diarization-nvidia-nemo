"""
Pure speaker diarization module using NVIDIA NeMo.
Simplified version without caching - just performs diarization and returns results.
"""

import os
import json
import shutil
import platform
import urllib.request
from pathlib import Path
from omegaconf import OmegaConf

# Try to import NVIDIA NeMo collections
try:
    from nemo.collections.asr.models import ClusteringDiarizer
except ModuleNotFoundError:
    print("\nModule 'nemo' is not installed in the active environment.")
    print("Install a compatible PyTorch first (CPU example):")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("Then install NeMo (example, may require NVIDIA index):")
    print("  pip install nemo_toolkit[all] --extra-index-url https://pypi.ngc.nvidia.com")
    print("After installing, re-run this script.")
    raise


def download_config(output_dir: str, domain_type: str = "telephonic") -> str:
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
        print(f"📥 Downloading configuration from: {config_url}")
        try:
            urllib.request.urlretrieve(config_url, config_path)
            print(f"✅ Configuration downloaded to: {config_path}")
        except Exception as e:
            print(f"❌ Error downloading config: {e}")
            raise
    else:
        print(f"📄 Using existing configuration: {config_path}")
    
    return config_path


def setup_config(audio_file: str, output_dir: str, domain_type: str = "telephonic", num_speakers: int = 2) -> OmegaConf:
    """
    Setup configuration for speaker diarization using official NeMo config.
    
    Args:
        audio_file: Path to the input audio file
        output_dir: Directory to store output files
        domain_type: Type of audio domain ('meeting' or 'telephonic')
        num_speakers: Number of speakers (default: 2)
    
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
    if platform.system() == 'Windows':
        cfg.num_workers = 0
        print("🪟 Windows detected: Setting num_workers=0 to avoid multiprocessing issues")
    
    # Update config with known number of speakers
    cfg.diarizer.clustering.parameters.oracle_num_speakers = True
    cfg.diarizer.clustering.parameters.max_num_speakers = num_speakers
    print(f"👥 Using oracle mode with {num_speakers} speakers")
    
    return cfg


def diarize(audio_filepath: str, output_dir: str, num_speakers: int = 2, domain_type: str = "telephonic") -> str:
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_filepath: Path to the audio file to diarize
        output_dir: Output directory for diarization results
        num_speakers: Number of speakers (default: 2)
        domain_type: Type of audio domain - 'meeting' or 'telephonic' (default: 'telephonic')
    
    Returns:
        str: Content of the .rttm file containing diarization results
        
    Raises:
        FileNotFoundError: If audio file or RTTM output is not found
        Exception: If diarization fails
    """
    # Clean output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"🧹 Cleaned existing output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🎙️  Starting speaker diarization")
    print(f"{'='*60}")
    print(f"📁 Audio file: {audio_filepath}")
    print(f"📂 Output directory: {output_dir}")
    print(f"🎯 Domain type: {domain_type}")
    print(f"👥 Number of speakers: {num_speakers}")
    print(f"{'='*60}\n")
    
    # Verify audio file exists
    audio_path = Path(audio_filepath).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_filepath}")
    
    # Create manifest file
    manifest_path = Path(output_dir) / "manifest.json"
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
    
    print(f"📝 Created manifest file: {manifest_path}\n")
    
    try:
        # Setup configuration
        cfg = setup_config(audio_filepath, output_dir, domain_type, num_speakers)
        
        # Initialize the diarization model
        print("🤖 Initializing ClusteringDiarizer model...")
        print("⬇️  Downloading pretrained models (this may take a while on first run)...\n")
        
        diarizer = ClusteringDiarizer(cfg=cfg)
        
        # Run diarization
        print("🔄 Performing diarization...\n")
        diarizer.diarize()
        
        print(f"\n{'='*60}")
        print("✅ Diarization complete!")
        print(f"{'='*60}\n")
        
        # Find and read RTTM file
        audio_basename = Path(audio_filepath).stem
        rttm_file = Path(output_dir) / "pred_rttms" / f"{audio_basename}.rttm"
        
        if rttm_file.exists():
            print(f"📄 RTTM file: {rttm_file}")
            with open(rttm_file, 'r') as f:
                rttm_content = f.read()
            
            # Print summary
            lines = rttm_content.strip().split('\n')
            speakers = set()
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 8:
                        speakers.add(parts[7])
            
            print(f"📊 Detected {len(lines)} segments from {len(speakers)} speakers: {', '.join(sorted(speakers))}\n")
            
            return rttm_content
        else:
            # Fallback: try to find any .rttm file
            rttm_dir = os.path.join(output_dir, 'pred_rttms')
            if os.path.exists(rttm_dir):
                rttm_files = [f for f in os.listdir(rttm_dir) if f.endswith('.rttm')]
                if rttm_files:
                    rttm_filepath = os.path.join(rttm_dir, rttm_files[0])
                    print(f"📄 RTTM file: {rttm_filepath}")
                    with open(rttm_filepath, 'r') as f:
                        rttm_content = f.read()
                    return rttm_content
            
            raise FileNotFoundError("❌ RTTM file not found. Check the output directory for results.")
        
    except Exception as e:
        print(f"\n❌ Error during diarization: {e}")
        print("\n💡 Troubleshooting tips:")
        print("  1. Make sure your audio file is in a supported format (WAV, FLAC, MP3)")
        print("  2. Check that the audio file is not corrupted")
        print("  3. Ensure you have sufficient disk space for model downloads")
        print("  4. Try updating nemo_toolkit: pip install --upgrade nemo_toolkit[asr]")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Perform speaker diarization on an audio file using NVIDIA NeMo.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diarize.py audio.wav
  python diarize.py audio.wav --output results
  python diarize.py audio.wav --num_speakers 3
  python diarize.py audio.wav --domain meeting
        """
    )
    parser.add_argument('audio_file', type=str,
                        help='Path to the audio file to diarize')
    parser.add_argument('--output', type=str, default='./diarization_output',
                        help='Output directory for results (default: ./diarization_output)')
    parser.add_argument('--num_speakers', type=int, default=2,
                        help='Number of speakers (default: 2)')
    parser.add_argument('--domain', type=str, choices=['meeting', 'telephonic'], 
                        default='telephonic',
                        help='Audio domain: "meeting" or "telephonic" (default: telephonic)')
    
    args = parser.parse_args()
    
    if os.path.exists(args.audio_file):
        try:
            rttm_content = diarize(
                audio_filepath=args.audio_file,
                output_dir=args.output,
                num_speakers=args.num_speakers,
                domain_type=args.domain
            )
            print("\n" + "="*60)
            print("📋 RTTM Content")
            print("="*60)
            print(rttm_content)
            print("="*60 + "\n")
        except Exception as e:
            print(f"\n❌ Failed: {e}")
            exit(1)
    else:
        print(f"❌ Audio file not found: {args.audio_file}")
        print("Please provide a valid audio file path.")
        exit(1)

