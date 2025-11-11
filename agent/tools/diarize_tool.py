from langchain.tools import tool
import os
import sys
import shutil
import json
import re
import tempfile
from pathlib import Path
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
import urllib.request
import platform

# Add parent directory to path to import diarize
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def has_special_chars(filename: str) -> bool:
    """
    Check if filename contains characters that may cause issues with NeMo.
    
    Args:
        filename: The filename to check
        
    Returns:
        True if special characters are found, False otherwise
    """
    # Characters that can cause issues: brackets, parentheses in some contexts
    # We're being conservative and checking for common problematic characters
    special_chars = r'[\[\]]'
    return bool(re.search(special_chars, filename))


def create_temp_audio_copy(audio_filepath: str, temp_dir: str) -> tuple[str, str]:
    """
    Create a temporary copy of the audio file with a sanitized filename.
    
    Args:
        audio_filepath: Original audio file path
        temp_dir: Directory to store temporary file
        
    Returns:
        Tuple of (temp_filepath, original_basename)
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    original_path = Path(audio_filepath)
    original_basename = original_path.stem
    extension = original_path.suffix
    
    # Create a sanitized filename using timestamp
    import time
    sanitized_name = f"temp_audio_{int(time.time() * 1000)}{extension}"
    temp_filepath = os.path.join(temp_dir, sanitized_name)
    
    # Copy the file
    shutil.copy2(audio_filepath, temp_filepath)
    print(f"🔧 Created temporary copy with sanitized filename: {sanitized_name}")
    
    return temp_filepath, original_basename


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
    # Always clean the output directory to ensure fresh results
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
    
    # Always use a temporary file for NeMo processing to avoid any filename-related issues
    # but keep output folder and other references with the original filename
    original_filename = audio_path.name
    print(f"🔧 Creating temporary copy with sanitized filename for NeMo processing...")
    temp_dir = os.path.join(output_dir, "temp_audio")
    temp_file_path, _ = create_temp_audio_copy(str(audio_path), temp_dir)
    audio_path_for_nemo = Path(temp_file_path)
    using_temp_file = True
    print(f"✅ Using temporary file for NeMo: {audio_path_for_nemo.name}\n")
    
    # Create manifest file (use temp file path for NeMo)
    manifest_path = Path(output_dir) / "manifest.json"
    manifest_entry = {
        "audio_filepath": str(audio_path_for_nemo),
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
        
        # Find any RTTM file generated by NeMo in the pred_rttms directory
        rttm_dir = Path(output_dir) / "pred_rttms"
        final_rttm_file = rttm_dir / "diarization.rttm"
        
        if not rttm_dir.exists():
            raise FileNotFoundError(f"❌ RTTM directory not found: {rttm_dir}")
        
        # Find any .rttm file in the directory
        rttm_files = list(rttm_dir.glob("*.rttm"))
        
        if not rttm_files:
            raise FileNotFoundError(f"❌ No RTTM files found in {rttm_dir}. Check the output directory for results.")
        
        # Use the first (and likely only) RTTM file found
        generated_rttm_file = rttm_files[0]
        print(f"📄 Found generated RTTM file: {generated_rttm_file}")
        
        # Rename to standard name to avoid long filename issues
        if generated_rttm_file != final_rttm_file:
            shutil.move(str(generated_rttm_file), str(final_rttm_file))
            print(f"📄 Renamed RTTM file to: {final_rttm_file}")
        
        with open(final_rttm_file, 'r') as f:
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
        
        # Clean up temporary file if used
        if using_temp_file and temp_file_path and os.path.exists(temp_file_path):
            try:
                temp_dir = os.path.dirname(temp_file_path)
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temporary audio file\n")
            except Exception as cleanup_error:
                print(f"⚠️  Could not clean up temporary file: {cleanup_error}\n")
        
        return rttm_content
        
    except Exception as e:
        # Clean up temporary file if used
        if using_temp_file and temp_file_path and os.path.exists(temp_file_path):
            try:
                temp_dir = os.path.dirname(temp_file_path)
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temporary audio file\n")
            except Exception as cleanup_error:
                print(f"⚠️  Could not clean up temporary file: {cleanup_error}\n")
        
        print(f"\n❌ Error during diarization: {e}")
        print("\n💡 Troubleshooting tips:")
        print("  1. Make sure your audio file is in a supported format (WAV, FLAC, MP3)")
        print("  2. Check that the audio file is not corrupted")
        print("  3. Ensure you have sufficient disk space for model downloads")
        print("  4. Try updating nemo_toolkit: pip install --upgrade nemo_toolkit[asr]")
        raise


def sanitize_directory_name(name: str) -> str:
    """
    Sanitize a directory name by removing problematic characters.
    
    Args:
        name: The directory name to sanitize
        
    Returns:
        Sanitized directory name safe for all filesystems
    """
    # Replace brackets and other problematic characters with underscores
    sanitized = re.sub(r'[\[\]<>:"|?*]', '_', name)
    return sanitized


@tool
def diarize_audio(audio_filepath: str, num_speakers: int = 2, domain_type: str = "telephonic") -> dict:
    """Perform speaker diarization on an audio file to identify who spoke when.
    
    This tool processes an audio file and returns speaker diarization results showing
    when each speaker was talking. This is the first step for audio analysis workflows.

    Args:
        audio_filepath: Path to the audio file (WAV, FLAC, or MP3)
        num_speakers: Number of speakers in the audio (default: 2)
        domain_type: Type of audio - "telephonic" for phone calls or "meeting" for meetings (default: "telephonic")
    
    Returns:
        dict: Dictionary containing:
            - success: Whether diarization succeeded
            - rttm_content: Raw RTTM format results
            - output_dir: Directory with all output files
            - segments: List of diarization segments with speaker, start time, and duration
            - error: Error message if failed
    """
    try:
        # Verify audio file exists
        if not os.path.exists(audio_filepath):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_filepath}"
            }
        
        # Create output directory path with absolute path to avoid issues
        # Get the agent directory
        agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        audio_basename = os.path.splitext(os.path.basename(audio_filepath))[0]
        
        # Sanitize the directory name to avoid special character issues
        output_dir = os.path.join(agent_dir, "output", "diarization", audio_basename)
        
        print(f"📂 Output directory: {output_dir}")
        
        # Perform diarization
        rttm_content = diarize(
            audio_filepath=audio_filepath,
            output_dir=output_dir,
            num_speakers=num_speakers,
            domain_type=domain_type
        )
        
        # Parse RTTM content into structured segments
        segments = []
        for line in rttm_content.strip().split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 8:
                    segments.append({
                        "speaker": parts[7],
                        "start_time": float(parts[3]),
                        "duration": float(parts[4])
                    })
        
        return {
            "success": True,
            "rttm_content": rttm_content,
            "output_dir": output_dir,
            "segments": segments,
            "num_segments": len(segments),
            "speakers": list(set(seg["speaker"] for seg in segments))
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

