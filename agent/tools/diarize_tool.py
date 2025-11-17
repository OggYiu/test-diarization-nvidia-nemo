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

# Import settings
import settings


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
        
    Raises:
        FileNotFoundError: If source file doesn't exist or is empty
        IOError: If copy operation fails or destination file is invalid
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    original_path = Path(audio_filepath)
    
    # Check if source file exists
    if not original_path.exists():
        raise FileNotFoundError(f"Source audio file does not exist: {audio_filepath}")
    
    # Check if source file is not empty
    source_size = original_path.stat().st_size
    if source_size == 0:
        raise IOError(f"Source audio file is empty (0 bytes): {audio_filepath}")
    
    print(f"📊 Source file size: {source_size:,} bytes")
    
    original_basename = original_path.stem
    extension = original_path.suffix
    
    # Create a sanitized filename using timestamp
    import time
    sanitized_name = f"temp_audio_{int(time.time() * 1000)}{extension}"
    temp_filepath = os.path.join(temp_dir, sanitized_name)
    
    # Copy the file
    try:
        shutil.copy2(audio_filepath, temp_filepath)
    except Exception as e:
        raise IOError(f"Failed to copy audio file: {e}")
    
    # Verify the temp file was created
    temp_path = Path(temp_filepath)
    if not temp_path.exists():
        raise IOError(f"Temporary file was not created: {temp_filepath}")
    
    # Verify the temp file is not empty
    temp_size = temp_path.stat().st_size
    if temp_size == 0:
        raise IOError(f"Temporary file is empty after copy: {temp_filepath}")
    
    # Verify the file size matches the source (or is at least reasonable)
    if temp_size != source_size:
        raise IOError(
            f"File size mismatch after copy. Source: {source_size} bytes, "
            f"Temp: {temp_size} bytes"
        )
    
    print(f"✅ Created temporary copy with sanitized filename: {sanitized_name}")
    print(f"✅ Verified temp file size: {temp_size:,} bytes")
    
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


def setup_config(audio_file: str, output_dir: str, temp_dir: str, domain_type: str = "telephonic", num_speakers: int = 2) -> OmegaConf:
    """
    Setup configuration for speaker diarization using official NeMo config.
    
    Args:
        audio_file: Path to the input audio file
        output_dir: Directory to store output files
        temp_dir: Temporary directory for manifest (to avoid special character issues)
        domain_type: Type of audio domain ('meeting' or 'telephonic')
        num_speakers: Number of speakers (default: 2)
    
    Returns:
        OmegaConf configuration object
    """
    # Download official config to temp directory to avoid special character issues
    config_path = download_config(temp_dir, domain_type)
    
    # Load configuration
    cfg = OmegaConf.load(config_path)
    
    # Create manifest path in temp directory to avoid special character issues
    manifest_path = os.path.join(temp_dir, 'manifest.json')
    
    # Update paths - use temp_dir for ALL NeMo operations to avoid special character issues
    cfg.diarizer.manifest_filepath = manifest_path
    cfg.diarizer.out_dir = temp_dir  # NeMo will write all intermediate files here
    
    # Fix for Windows multiprocessing issues
    if platform.system() == 'Windows':
        cfg.num_workers = 0
        print("🪟 Windows detected: Setting num_workers=0 to avoid multiprocessing issues")
    
    # Update config with known number of speakers
    cfg.diarizer.clustering.parameters.oracle_num_speakers = True
    cfg.diarizer.clustering.parameters.max_num_speakers = num_speakers
    print(f"👥 Using oracle mode with {num_speakers} speakers")
    
    return cfg


def diarize(audio_filepath: str, output_dir: str, num_speakers: int = 2, domain_type: str = "telephonic", overwrite: bool = False) -> str:
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_filepath: Path to the audio file to diarize
        output_dir: Output directory for diarization results
        num_speakers: Number of speakers (default: 2)
        domain_type: Type of audio domain - 'meeting' or 'telephonic' (default: 'telephonic')
        overwrite: If True, re-run diarization even if RTTM file exists (default: False)
    
    Returns:
        str: Path to the .rttm file containing diarization results
        
    Raises:
        FileNotFoundError: If audio file or RTTM output is not found
        Exception: If diarization fails
    """
    # Check if RTTM file already exists
    expected_rttm_file = Path(output_dir) / "pred_rttms" / "diarization.rttm"
    
    if not overwrite and expected_rttm_file.exists():
        # Check if the file is not empty
        if expected_rttm_file.stat().st_size > 0:
            print(f"\n{'='*60}")
            print(f"✅ Found existing RTTM file (not empty)")
            print(f"{'='*60}")
            print(f"📄 Using existing RTTM file: {expected_rttm_file}")
            print(f"💡 To re-run diarization, set overwrite=True")
            print(f"{'='*60}\n")
            return str(expected_rttm_file)
        else:
            print(f"⚠️  Existing RTTM file is empty, will re-run diarization")
    
    # Clean the output directory only if overwrite is True
    if overwrite and os.path.exists(output_dir):
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
    # Use centralized temp directory to avoid path issues
    agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(agent_dir, "output", "temp")
    temp_file_path, _ = create_temp_audio_copy(str(audio_path), temp_dir)
    
    audio_path_for_nemo = Path(temp_file_path)
    
    print(f"✅ Using temporary file for NeMo: {audio_path_for_nemo}")
    print(f"🐛 DEBUG: Temp audio file path: {temp_file_path}")
    print(f"🐛 DEBUG: Temp directory: {temp_dir}\n")
    
    # Create manifest file in temp folder to avoid special character issues
    # NeMo has trouble when the manifest is in a folder with special characters
    manifest_path = Path(temp_dir) / "manifest.json"
    # Convert Windows path to forward slashes for better compatibility with NeMo
    audio_path_str = str(audio_path_for_nemo).replace('\\', '/')
    manifest_entry = {
        "audio_filepath": audio_path_str,
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
    
    print(f"📝 Created manifest file: {manifest_path}")
    print(f"🐛 DEBUG: Manifest audio_filepath: {audio_path_str}")
    
    # Read and print manifest content for debugging
    with open(manifest_path, 'r') as f:
        manifest_content = f.read()
        print(f"🐛 DEBUG: Manifest content:\n{manifest_content}")
    
    # Verify temp audio file exists and is readable
    print(f"🐛 DEBUG: Verifying temp audio file...")
    print(f"🐛 DEBUG: Temp file exists: {os.path.exists(temp_file_path)}")
    if os.path.exists(temp_file_path):
        import wave
        try:
            with wave.open(temp_file_path, 'rb') as w:
                print(f"🐛 DEBUG: Temp file - Channels: {w.getnchannels()}, Framerate: {w.getframerate()}, Frames: {w.getnframes()}, Duration: {w.getnframes() / w.getframerate():.2f}s")
        except Exception as e:
            print(f"🐛 DEBUG: Error reading temp file with wave: {e}")
    print()
    
    try:
        # Setup configuration
        cfg = setup_config(audio_filepath, output_dir, temp_dir, domain_type, num_speakers)
        
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
        
        # Find RTTM file in temp directory (where NeMo wrote it)
        temp_rttm_dir = Path(temp_dir) / "pred_rttms"
        
        if not temp_rttm_dir.exists():
            raise FileNotFoundError(f"❌ RTTM directory not found: {temp_rttm_dir}")
        
        # Find any .rttm file in the directory
        rttm_files = list(temp_rttm_dir.glob("*.rttm"))
        
        if not rttm_files:
            raise FileNotFoundError(f"❌ No RTTM files found in {temp_rttm_dir}. Check the temp directory for results.")
        
        # Use the first (and likely only) RTTM file found
        generated_rttm_file = rttm_files[0]
        print(f"📄 Found generated RTTM file in temp: {generated_rttm_file}")
        
        # Read RTTM content
        with open(generated_rttm_file, 'r') as f:
            rttm_content = f.read()
        
        # Copy ALL NeMo outputs from temp to final output directory
        print(f"\n📦 Copying all NeMo outputs to final location...")
        
        # Copy pred_rttms directory
        output_rttm_dir = Path(output_dir) / "pred_rttms"
        output_rttm_dir.mkdir(parents=True, exist_ok=True)
        final_rttm_file = output_rttm_dir / "diarization.rttm"
        with open(final_rttm_file, 'w') as f:
            f.write(rttm_content)
        print(f"  ✅ Copied RTTM to: {final_rttm_file}")
        
        # Copy speaker_outputs directory
        temp_speaker_outputs = Path(temp_dir) / "speaker_outputs"
        if temp_speaker_outputs.exists():
            final_speaker_outputs = Path(output_dir) / "speaker_outputs"
            if final_speaker_outputs.exists():
                shutil.rmtree(final_speaker_outputs)
            shutil.copytree(temp_speaker_outputs, final_speaker_outputs)
            print(f"  ✅ Copied speaker_outputs to: {final_speaker_outputs}")
        
        # Copy vad_outputs directory
        temp_vad_outputs = Path(temp_dir) / "vad_outputs"
        if temp_vad_outputs.exists():
            final_vad_outputs = Path(output_dir) / "vad_outputs"
            if final_vad_outputs.exists():
                shutil.rmtree(final_vad_outputs)
            shutil.copytree(temp_vad_outputs, final_vad_outputs)
            print(f"  ✅ Copied vad_outputs to: {final_vad_outputs}")
        
        # Copy manifest files
        temp_manifest = Path(temp_dir) / "manifest.json"
        if temp_manifest.exists():
            final_manifest = Path(output_dir) / "manifest.json"
            shutil.copy2(temp_manifest, final_manifest)
            print(f"  ✅ Copied manifest.json to: {final_manifest}")
        
        temp_manifest_vad = Path(temp_dir) / "manifest_vad_input.json"
        if temp_manifest_vad.exists():
            final_manifest_vad = Path(output_dir) / "manifest_vad_input.json"
            shutil.copy2(temp_manifest_vad, final_manifest_vad)
            print(f"  ✅ Copied manifest_vad_input.json to: {final_manifest_vad}")
        
        # Copy config file to final output directory for reference
        temp_config = Path(temp_dir) / f"diar_infer_{domain_type}.yaml"
        if temp_config.exists():
            final_config = Path(output_dir) / f"diar_infer_{domain_type}.yaml"
            shutil.copy2(str(temp_config), str(final_config))
            print(f"  ✅ Copied config to: {final_config}")
        
        print(f"📦 All NeMo outputs copied successfully!\n")
        
        # Print summary
        lines = rttm_content.strip().split('\n')
        speakers = set()
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 8:
                    speakers.add(parts[7])
        
        print(f"📊 Detected {len(lines)} segments from {len(speakers)} speakers: {', '.join(sorted(speakers))}\n")
        
        # Clean up temporary directory after all files have been copied
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temporary directory: {temp_dir}\n")
            except Exception as cleanup_error:
                print(f"⚠️  Warning: Could not clean up temporary directory: {cleanup_error}\n")
        
        return str(final_rttm_file)
        
    except Exception as e:
        # Clean up temporary directory even if error occurred
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temporary directory after error: {temp_dir}\n")
            except Exception as cleanup_error:
                print(f"⚠️  Warning: Could not clean up temporary directory: {cleanup_error}\n")
        
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
            - audio_filepath: Path to the original audio file that was diarized
            - rttm_filepath: Path to the RTTM file containing diarization results (or error dict if failed)
    """
    try:
        # Read overwrite setting from settings file
        overwrite = settings.DIARIZATION_OVERWRITE
        
        # Verify audio file exists
        if not os.path.exists(audio_filepath):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_filepath}"
            }
        
        # Get the agent directory
        agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        audio_basename = os.path.splitext(os.path.basename(audio_filepath))[0]
        
        # ALWAYS use the agent/output/diarization directory structure
        # This prevents accidentally creating folders in the source directory
        output_dir = os.path.join(agent_dir, "output", "diarization", audio_basename)
        
        # Validate that output_dir is NOT in the source audio directory
        audio_dir = os.path.dirname(os.path.abspath(audio_filepath))
        output_dir_abs = os.path.abspath(output_dir)
        if output_dir_abs.startswith(audio_dir):
            # This would create output in the source folder - prevent it!
            output_dir = os.path.join(agent_dir, "output", "diarization", audio_basename)
            print(f"⚠️  Prevented creating output in source directory")
        
        print(f"📂 Diarization output directory: {output_dir}")
        
        # Perform diarization
        rttm_filepath = diarize(
            audio_filepath=audio_filepath,
            output_dir=output_dir,
            num_speakers=num_speakers,
            domain_type=domain_type,
            overwrite=overwrite
        )
        
        # Return only the file paths - no content to avoid confusing the LLM
        return {
            "audio_filepath": audio_filepath,
            "rttm_filepath": rttm_filepath
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

