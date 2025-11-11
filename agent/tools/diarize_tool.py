from langchain.tools import tool
import os
import sys

# Add parent directory to path to import diarize
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from diarize import diarize


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
        
        # Create output directory path with new structure: output/diarization/file_name
        audio_basename = os.path.splitext(os.path.basename(audio_filepath))[0]
        output_dir = os.path.join("output", "diarization", audio_basename)
        
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

