from langchain.tools import tool
import os
import tempfile
import sys
from pydub import AudioSegment
from pathlib import Path

# Add parent directory to path to import audio_chopper
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def read_rttm_file(rttm_path):
    """
    Read an RTTM file and extract speaker segments.
    
    Args:
        rttm_path: Path to the RTTM file
        
    Returns:
        List of dictionaries containing segment information:
        [{'filename': str, 'speaker': str, 'start': float, 'duration': float, 'end': float}, ...]
    """
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
        print(f"Warning: WAV file not found: {wav_path}")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load audio file using pydub
    print(f"Loading audio file: {wav_path}")
    audio = AudioSegment.from_wav(wav_path)
    audio_duration_sec = len(audio) / 1000.0
    
    base_filename = Path(wav_path).stem
    
    # Process each segment
    for i, segment in enumerate(segments, 1):
        start_ms = max(0, segment['start'] * 1000 - padding_ms)
        end_ms = min(len(audio), segment['end'] * 1000 + padding_ms)
        
        # Extract the segment
        segment_audio = audio[start_ms:end_ms]
        
        # Create output filename with start time, duration, and speaker
        speaker = segment['speaker']
        start_time_formatted = int(segment['start'] * 1000)  # Convert to milliseconds as integer
        duration_formatted = int(segment['duration'] * 1000)  # Convert to milliseconds as integer
        output_filename = f"segment_{i:03d}_{start_time_formatted:04d}_{duration_formatted:04d}_{speaker}.wav"
        output_path = os.path.join(output_folder, output_filename)
        
        # Save the segment
        segment_audio.export(output_path, format="wav")
        
        duration_sec = segment['duration']
        print(f"  Saved segment {i}: {output_filename} "
              f"({segment['start']:.2f}s - {segment['end']:.2f}s, "
              f"duration: {duration_sec:.2f}s, speaker: {speaker})")


@tool
def chop_audio_by_rttm(audio_filepath: str, rttm_content: str = None, rttm_filepath: str = None, 
                        output_dir: str = None, padding_ms: int = 100) -> dict:
    """Chop an audio file into speaker segments based on RTTM diarization data.
    
    This tool splits an audio file into separate segments for each speaker turn based on
    diarization results (RTTM format). Use this after running diarization to get individual
    speaker segments that can be transcribed separately.
    
    Args:
        audio_filepath: Path to the audio file to chop (WAV, FLAC, or MP3)
        rttm_content: RTTM content as a string (provide either this or rttm_filepath)
        rttm_filepath: Path to RTTM file (provide either this or rttm_content)
        output_dir: Directory to save chopped segments (default: creates temp directory)
        padding_ms: Padding in milliseconds to add before/after each segment (default: 100)
    
    Returns:
        dict: Dictionary containing:
            - success: Whether chopping succeeded
            - output_dir: Directory containing chopped segments
            - segments: List of segment information with file paths
            - num_segments: Total number of segments created
            - error: Error message if failed
    """
    try:
        # Verify audio file exists
        if not os.path.exists(audio_filepath):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_filepath}"
            }
        
        # Verify we have RTTM data
        if not rttm_content and not rttm_filepath:
            return {
                "success": False,
                "error": "Must provide either rttm_content or rttm_filepath"
            }
        
        # Create output directory if not provided, using consistent structure: output/chopped_segments/file_name
        if output_dir is None:
            audio_basename = os.path.splitext(os.path.basename(audio_filepath))[0]
            output_dir = os.path.join("output", "chopped_segments", audio_basename)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle RTTM data
        if rttm_filepath:
            # Read from file
            if not os.path.exists(rttm_filepath):
                return {
                    "success": False,
                    "error": f"RTTM file not found: {rttm_filepath}"
                }
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
        
        # Verify we have segments
        if not segments:
            return {
                "success": False,
                "error": "No segments found in RTTM data"
            }
        
        # Chop the audio file
        chop_audio_file(audio_filepath, segments, output_dir, padding_ms)
        
        # Get list of chopped files
        chopped_files = sorted([
            os.path.join(output_dir, f) 
            for f in os.listdir(output_dir) 
            if f.endswith('.wav')
        ])
        
        # Build segment info list
        segment_info = []
        for idx, seg in enumerate(segments):
            # Match the filename pattern from chop_audio_file
            start_ms = int(seg['start'] * 1000)
            duration_ms = int(seg['duration'] * 1000)
            expected_filename = f"segment_{idx+1:03d}_{start_ms:04d}_{duration_ms:04d}_{seg['speaker']}.wav"
            filepath = os.path.join(output_dir, expected_filename)
            
            segment_info.append({
                "segment_number": idx + 1,
                "filepath": filepath,
                "speaker": seg['speaker'],
                "start_time": seg['start'],
                "duration": seg['duration'],
                "end_time": seg['end']
            })
        
        return {
            "success": True,
            "output_dir": output_dir,
            "segments": segment_info,
            "num_segments": len(segment_info),
            "speakers": list(set(seg['speaker'] for seg in segments))
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

