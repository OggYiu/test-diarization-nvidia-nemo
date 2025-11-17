from langchain.tools import tool
import os
import tempfile
import sys
import shutil
import re
import time
from pydub import AudioSegment
from pathlib import Path

# Add parent directory to path to import audio_chopper
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import settings
import settings

# Import path normalization utilities
from .path_utils import normalize_path_for_llm, normalize_path_from_llm


def read_rttm_file(rttm_path):
    """
    Read an RTTM file and extract speaker segments.
    
    Args:
        rttm_path: Path to the RTTM file
        
    Returns:
        List of dictionaries containing segment information:
        [{'filename': str, 'speaker': str, 'start': float, 'duration': float, 'end': float}, ...]
    """
    trace_start = time.time()
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Reading RTTM file: {os.path.basename(rttm_path)}")
    segments = []
    
    with open(rttm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            # RTTM lines are space-separated but filenames may contain spaces
            # Strategy: find the first index `i` such that parts[i] and parts[i+1]
            # can both be parsed as floats — these correspond to start and duration.
            # The channel token is then at i-1, and the filename is everything
            # between parts[1] and parts[i-1] (exclusive of channel token).
            start_idx = None
            for i in range(2, len(parts) - 1):
                try:
                    # require parts[i] and parts[i+1] be numeric
                    _ = float(parts[i])
                    _ = float(parts[i + 1])
                except Exception:
                    continue

                # prefer the pair where the token after duration is non-numeric
                # (RTTM usually has '<NA>' after duration)
                next_is_numeric = False
                if i + 2 < len(parts):
                    try:
                        _ = float(parts[i + 2])
                        next_is_numeric = True
                    except Exception:
                        next_is_numeric = False

                if not next_is_numeric:
                    start_idx = i
                    break

                # otherwise keep searching (avoid selecting channel as start)

            if start_idx is None:
                # Couldn't detect start/duration pair; skip line
                continue

            channel_idx = start_idx - 1
            # filename spans parts[1:channel_idx]
            filename = ' '.join(parts[1:channel_idx]) if channel_idx > 1 else parts[1]

            try:
                start = float(parts[start_idx])
                duration = float(parts[start_idx + 1])
            except Exception:
                # fallback if parsing failed
                continue

            # speaker id is typically at offset start_idx + 4 in RTTM
            spk_idx = start_idx + 4
            if spk_idx < len(parts):
                speaker = parts[spk_idx]
            else:
                # fallback: try last token
                speaker = parts[-1]

            segment = {
                'filename': filename,
                'start': start,
                'duration': duration,
                'speaker': speaker,
            }
            segment['end'] = segment['start'] + segment['duration']
            segments.append(segment)
    
    trace_elapsed = time.time() - trace_start
    print(f"[TRACE {time.strftime('%H:%M:%S')}] RTTM file read completed in {trace_elapsed:.4f}s - Found {len(segments)} segments")
    return segments


def chop_audio_file(wav_path, segments, output_folder, padding_ms=100):
    """
    Chop a WAV file into smaller segments based on diarization results.
    
    Args:
        wav_path: Path to the WAV file
        segments: List of segment dictionaries with 'start', 'duration', 'speaker'
        output_folder: Folder to save chopped audio files
        padding_ms: Padding in milliseconds to add before/after each segment (default: 100ms)
    """
    if not os.path.exists(wav_path):
        error_msg = f"❌ WAV file not found: {wav_path}"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load audio file using pydub
    print(f"📂 Loading audio file: {wav_path}")
    print(f"📊 File size: {os.path.getsize(wav_path) / (1024*1024):.2f} MB")
    
    trace_start = time.time()
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Starting audio file loading...")
    try:
        audio = AudioSegment.from_wav(wav_path)
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Audio file loading completed in {trace_elapsed:.2f}s")
        print(f"✅ Audio loaded successfully - Duration: {len(audio)/1000:.2f}s")
    except Exception as e:
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Audio file loading failed after {trace_elapsed:.2f}s: {e}")
        error_msg = f"❌ Failed to load audio file with pydub: {e}"
        print(error_msg)
        raise
    audio_duration_sec = len(audio) / 1000.0
    
    base_filename = Path(wav_path).stem
    
    # Process each segment
    trace_start = time.time()
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Starting audio chopping for {len(segments)} segments...")
    for i, segment in enumerate(segments, 1):
        start_ms = max(0, segment['start'] * 1000 - padding_ms)
        end_ms = min(len(audio), segment['end'] * 1000 + padding_ms)
        
        # Extract the segment
        segment_audio = audio[start_ms:end_ms]
        
        # Create output filename with original filename, start time, duration, and speaker
        speaker = segment['speaker']
        start_time_formatted = int(segment['start'] * 1000)  # Convert to milliseconds as integer
        duration_formatted = int(segment['duration'] * 1000)  # Convert to milliseconds as integer
        output_filename = f"{base_filename}_segment_{i:03d}_{start_time_formatted:04d}_{duration_formatted:04d}_{speaker}.wav"
        output_path = os.path.join(output_folder, output_filename)
        
        # Save the segment
        segment_audio.export(output_path, format="wav")
        
        duration_sec = segment['duration']
        print(f"  Saved segment {i}: {output_filename} "
              f"({segment['start']:.2f}s - {segment['end']:.2f}s, "
              f"duration: {duration_sec:.2f}s, speaker: {speaker})")
    
    trace_elapsed = time.time() - trace_start
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Audio chopping completed in {trace_elapsed:.2f}s for {len(segments)} segments")



@tool
def chop_audio_by_rttm(audio_filepath: str, rttm_content: str = None, rttm_filepath: str = None) -> str:
    """Chop an audio file into speaker segments based on RTTM diarization data.
    
    This tool splits an audio file into separate segments for each speaker turn based on
    diarization results (RTTM format). Use this after running diarization to get individual
    speaker segments that can be transcribed separately.
    
    The output will be saved to: agent/output/chopped_segments/[audio_filename]/
    
    Args:
        audio_filepath: Path to the audio file to chop (WAV, FLAC, or MP3)
        rttm_content: RTTM content as a string (provide either this or rttm_filepath)
        rttm_filepath: Path to RTTM file (provide either this or rttm_content)
    
    Returns:
        str: Path to the directory containing the chopped audio segments
    """

    try:
        # Normalize input paths to handle any LLM path manipulation issues
        audio_filepath = normalize_path_from_llm(audio_filepath)
        if rttm_filepath:
            rttm_filepath = normalize_path_from_llm(rttm_filepath)
        
        # Read overwrite setting from settings file
        overwrite = settings.AUDIO_CHOPPER_OVERWRITE
        
        # Verify audio file exists
        if not os.path.exists(audio_filepath):
            return f"❌ Error: Audio file not found: {audio_filepath}"
        
        # Verify we have RTTM data
        if not rttm_content and not rttm_filepath:
            return "❌ Error: Must provide either rttm_content or rttm_filepath"
        
        # Get the agent directory
        agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        audio_basename = os.path.splitext(os.path.basename(audio_filepath))[0]
        
        # ALWAYS use the agent/output/chopped_segments directory structure
        # This prevents accidentally creating folders in the source directory
        output_dir = os.path.join(agent_dir, "output", "chopped_segments", audio_basename)
        
        # Validate that output_dir is NOT in the source audio directory
        audio_dir = os.path.dirname(os.path.abspath(audio_filepath))
        output_dir_abs = os.path.abspath(output_dir)
        if output_dir_abs.startswith(audio_dir):
            # This would create output in the source folder - prevent it!
            output_dir = os.path.join(agent_dir, "output", "chopped_segments", audio_basename)
            print(f"⚠️  Prevented creating output in source directory")
        
        print(f"📂 Chopping output directory: {output_dir}")
        
        # Check if files are already chopped
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            # Check if directory has any WAV files
            existing_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
            if existing_files and not overwrite:
                print(f"✅ Files already chopped ({len(existing_files)} segments found). Skipping chopping (overwrite=False).")
                print(f"📂 Existing segments in: {output_dir}\n")
                # Normalize path for LLM consumption (use forward slashes)
                return normalize_path_for_llm(output_dir)
            elif existing_files and overwrite:
                print(f"⚠️  Overwriting existing {len(existing_files)} segments (overwrite=True)")
                shutil.rmtree(output_dir)
                print(f"🧹 Cleaned existing output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle RTTM data
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Processing RTTM data...")
        if rttm_filepath:
            # Read from file
            if not os.path.exists(rttm_filepath):
                return f"❌ Error: RTTM file not found: {rttm_filepath}"
            segments = read_rttm_file(rttm_filepath)
        else:
            # Save RTTM content to temporary file
            temp_rttm = tempfile.NamedTemporaryFile(mode='w', suffix='.rttm', delete=False, encoding='utf-8')
            temp_rttm.write(rttm_content)
            temp_rttm.close()
            
            try:
                segments = read_rttm_file(temp_rttm.name)
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_rttm.name)
                except:
                    pass
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] RTTM data processing completed in {trace_elapsed:.4f}s")
        
        # Verify we have segments
        if not segments:
            return "❌ Error: No segments found in RTTM data"
        
        # Chop the audio file
        padding_ms = 0
        chop_audio_file(audio_filepath, segments, output_dir, padding_ms)
        
        print(f"\n✅ Audio chopping complete!")
        print(f"📊 Created {len(segments)} segments from {len(set(seg['speaker'] for seg in segments))} speakers")
        print(f"📂 Segments saved to: {output_dir}\n")
        
        # Normalize path for LLM consumption (use forward slashes)
        output_dir_llm = normalize_path_for_llm(output_dir)
        
        # Format return message with continuation instruction
        message = f"\n{'='*80}\n"
        message += f"✅ Audio Chopping Complete\n"
        message += f"{'='*80}\n\n"
        message += f"📂 Output directory: {output_dir_llm}\n"
        message += f"🔊 Segments created: {len(segments)}\n\n"
        message += f"{'='*80}\n"
        message += "✅ Audio chopping complete. Continue with the next step in the pipeline.\n"
        message += f"   Use segments_directory: {output_dir_llm}\n"
        message += f"{'='*80}\n"
        
        return message
        
    except Exception as e:
        return f"❌ Error during audio chopping: {str(e)}"

