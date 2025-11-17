"""
STT Tool for Transcribing Audio Segments with SenseVoiceSmall
Copied from unified_gui.py (tabs/tab_stt.py) to avoid code duplication
"""

import os
import time
import json
from typing import Annotated
from langchain.tools import tool
from funasr import AutoModel
import torch
from opencc import OpenCC

# Import helper functions from batch_stt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from batch_stt import format_str_v3, load_audio

# Import settings
agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if agent_dir not in sys.path:
    sys.path.insert(0, agent_dir)
import settings

# Import path normalization utilities
from .path_utils import normalize_path_for_llm, normalize_path_from_llm

# Global variables for model management
sensevoice_model = None
current_sensevoice_loaded = False

# Initialize OpenCC converter for Traditional Chinese
opencc_converter = OpenCC('s2hk')  # Simplified to Hong Kong Traditional


def get_device_info():
    """
    Detect and return information about available compute device (GPU/CPU).
    
    Returns:
        tuple: (device, device_name, device_info_str)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        device_info = f"🚀 GPU: {device_name}"
        device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device_info += f" ({device_memory:.1f} GB)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        device_info = f"💻 CPU: {torch.get_num_threads()} threads"
    
    return device, device_name, device_info


# Get device once at module load
DEVICE, DEVICE_NAME, DEVICE_INFO = get_device_info()


def initialize_sensevoice_model(vad_model="fsmn-vad", max_single_segment_time=30000):
    """Initialize the SenseVoice model.
    
    Args:
        vad_model: VAD model to use (default: "fsmn-vad")
        max_single_segment_time: Maximum segment time in milliseconds (default: 30000)
    """
    global sensevoice_model, current_sensevoice_loaded
    
    # Only reload if not already loaded
    if current_sensevoice_loaded and sensevoice_model is not None:
        print(f"[TRACE {time.strftime('%H:%M:%S')}] SenseVoiceSmall already loaded, skipping initialization")
        return f"✅ SenseVoiceSmall already loaded"
    
    status = f"🔄 Loading SenseVoiceSmall model...\n"
    status += f"  ⚙️ Device: {DEVICE_INFO}\n"
    status += f"  ⚙️ VAD Model: {vad_model}\n"
    status += f"  ⚙️ Max Segment Time: {max_single_segment_time}ms\n"
    
    try:
        # Determine device for FunASR (use 'cuda' or 'cpu')
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device_str}")
        print(f"Using VAD model: {vad_model}")
        print(f"Using max segment time: {max_single_segment_time}ms")
        
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Starting SenseVoiceSmall model initialization (this may take a while)...")
        
        sensevoice_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model=vad_model,
            vad_kwargs={"max_single_segment_time": max_single_segment_time},
            trust_remote_code=False,
            disable_update=True,
            ban_emo_unk=True,
            device=device_str
        )
        
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] SenseVoiceSmall model initialization completed in {trace_elapsed:.2f}s ({trace_elapsed/60:.2f} minutes)")
        
        current_sensevoice_loaded = True
        status += f"✅ SenseVoiceSmall loaded successfully on {DEVICE_NAME}!"
        return status
    except Exception as e:
        trace_elapsed = time.time() - trace_start if 'trace_start' in locals() else 0
        print(f"[TRACE {time.strftime('%H:%M:%S')}] SenseVoiceSmall model initialization failed after {trace_elapsed:.2f}s: {str(e)}")
        status += f"❌ Failed to load SenseVoiceSmall: {str(e)}"
        return status


def transcribe_single_audio_sensevoice(audio_path, language="yue"):
    """
    Transcribe a single audio file using SenseVoiceSmall model.
    
    Args:
        audio_path: Path to audio file
        language: Language code for transcription (default: "yue" for Cantonese)
        
    Returns:
        dict: Transcription result with file, path, transcription, raw_transcription
    """
    global sensevoice_model
    
    if sensevoice_model is None:
        return None
    
    filename = os.path.basename(audio_path)
    
    # Load audio
    trace_start = time.time()
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Loading audio file: {filename}")
    audio_array, sample_rate = load_audio(audio_path)
    if audio_array is None:
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Audio loading failed after {trace_elapsed:.2f}s")
        return None
    
    trace_elapsed = time.time() - trace_start
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Audio loading completed in {trace_elapsed:.2f}s")
    
    # Run inference
    try:
        start_time = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Starting transcription inference for: {filename}")
        
        # Prepare generate parameters
        generate_params = {
            "input": audio_array,
            "cache": {},
            "language": language,
            "use_itn": True,
            "batch_size_s": 60,
            "merge_vad": True
        }
        
        result = sensevoice_model.generate(**generate_params)
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Transcription inference completed in {processing_time:.2f}s for: {filename}")
        
        # Extract and format text
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Post-processing transcription for: {filename}")
        raw_text = result[0]["text"]
        formatted_text = format_str_v3(raw_text)
        
        # Convert to Traditional Chinese using OpenCC
        formatted_text = opencc_converter.convert(formatted_text)
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Post-processing completed in {trace_elapsed:.4f}s")
        
        return {
            "file": filename,
            "path": audio_path,
            "transcription": formatted_text,
            "raw_transcription": raw_text,
            "processing_time": processing_time
        }
    except Exception as e:
        print(f"Error transcribing with SenseVoice {audio_path}: {e}")
        return None


@tool
def transcribe_audio_segments(
    segments_directory: Annotated[str, "Directory containing chopped audio segments to transcribe"],
    language: Annotated[str, "Language code (default: 'yue' for Cantonese)"] = "yue",
    vad_model: Annotated[str, "VAD model to use"] = "fsmn-vad",
    max_single_segment_time: Annotated[int, "Maximum segment time in milliseconds"] = 30000
) -> str:
    """
    Transcribe all audio segments in a directory using SenseVoiceSmall model.
    This tool reads all audio files (.wav, .mp3, .flac, .m4a, .ogg) from the given directory
    and transcribes them. Use this after chop_audio_by_rttm has created speaker segments.
    
    If a transcription file already exists and overwrite=False (from settings), the existing file will be
    returned without re-processing the audio segments.
    
    Args:
        segments_directory: Path to directory containing chopped audio segments
        language: Language code for transcription (default: "yue" for Cantonese)
        vad_model: VAD model to use (default: "fsmn-vad")
        max_single_segment_time: Maximum segment time in milliseconds (default: 30000)
    
    Returns:
        str: Path to the transcriptions_text.txt file containing all transcription results
    """
    try:
        # Normalize the input path to handle any LLM path manipulation issues
        segments_directory = normalize_path_from_llm(segments_directory)
        
        # Read overwrite setting from settings file
        overwrite = settings.STT_OVERWRITE
        
        if not os.path.exists(segments_directory):
            return f"❌ Error: Segments directory not found: {segments_directory}"
        
        # Get all audio files in the directory
        audio_files = []
        for file in os.listdir(segments_directory):
            if file.endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                audio_files.append(os.path.join(segments_directory, file))
        
        if not audio_files:
            return f"❌ Error: No audio files found in {segments_directory}"
        
        # Sort files to maintain order
        audio_files.sort()
        
        # Determine output directory path BEFORE processing to check for existing files
        # Extract filename from segments_directory path
        segments_folder_name = os.path.basename(os.path.normpath(segments_directory))
        
        # Clean up the folder name by removing common suffixes that shouldn't be there
        # This handles cases where segments_directory has unwanted suffixes like:
        # - "[filename].wav_segments" -> "[filename]"
        # - "chopped" -> "chopped" (will be handled as-is for backward compat)
        # - "chopped_segments" -> "chopped_segments" (will be handled as-is)
        suffixes_to_remove = ['.wav_segments', '.mp3_segments', '.flac_segments', '.m4a_segments', '.ogg_segments']
        for suffix in suffixes_to_remove:
            if segments_folder_name.endswith(suffix):
                segments_folder_name = segments_folder_name[:-len(suffix)]
                print(f"🔧 Cleaned folder name suffix: {suffix}")
                break
        
        # Create output/transcriptions/filename directory using absolute path
        # Get the agent directory (same pattern as diarize_tool and audio_chopper_tool)
        agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # ALWAYS use the agent/output/transcriptions directory structure
        # This prevents accidentally creating folders in the source directory
        output_dir = os.path.join(agent_dir, "output", "transcriptions", segments_folder_name)
        
        # Validate that output_dir is NOT in the segments directory (source)
        segments_dir_abs = os.path.abspath(segments_directory)
        output_dir_abs = os.path.abspath(output_dir)
        if output_dir_abs.startswith(segments_dir_abs):
            # This would create output in the segments folder - prevent it!
            print(f"⚠️  Prevented creating output in segments directory")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename: transcriptions.json
        output_json_filename = "transcriptions.json"
        output_json_path = os.path.join(output_dir, output_json_filename)
        
        # Also save as simple text format: transcriptions_text.txt
        output_text_filename = "transcriptions_text.txt"
        output_text_path = os.path.join(output_dir, output_text_filename)
        
        # Check if transcription files already exist BEFORE processing
        # If file exists and overwrite=False, skip initialization and return early
        if os.path.exists(output_text_path) and not overwrite:
            print(f"\n{'='*80}")
            print(f"✅ Found existing transcription file: {output_text_path}")
            print(f"📄 Using existing file (overwrite=False)")
            print(f"{'='*80}\n")
            # Normalize path for LLM consumption (use forward slashes)
            return normalize_path_for_llm(output_text_path)
        
        # Initialize model only if we need to transcribe (file doesn't exist or overwrite=True)
        init_status = initialize_sensevoice_model(vad_model, max_single_segment_time)
        print(init_status)
        
        # If overwrite=True or file doesn't exist, proceed with transcription
        if os.path.exists(output_text_path) and overwrite:
            print(f"\n⚠️  Existing transcription file found, overwriting (overwrite=True)")
        
        print(f"\n{'='*80}")
        print(f"🎙️ Transcribing {len(audio_files)} audio segments...")
        print(f"{'='*80}\n")
        
        results = []
        total_time = 0
        
        transcription_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Starting batch transcription of {len(audio_files)} segments...")
        
        for idx, audio_file in enumerate(audio_files, 1):
            print(f"📝 Processing segment {idx}/{len(audio_files)}: {os.path.basename(audio_file)}")
            segment_start = time.time()
            
            result = transcribe_single_audio_sensevoice(
                audio_path=audio_file,
                language=language,
            )
            
            segment_elapsed = time.time() - segment_start
            if result:
                results.append(result)
                total_time += result['processing_time']
                print(f"   ✅ Transcribed in {result['processing_time']:.2f}s (total segment time: {segment_elapsed:.2f}s)")
                print(f"   📄 Text: {result['transcription'][:100]}...")
            else:
                print(f"   ❌ Failed to transcribe (segment time: {segment_elapsed:.2f}s)")
        
        transcription_elapsed = time.time() - transcription_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Batch transcription completed in {transcription_elapsed:.2f}s ({transcription_elapsed/60:.2f} minutes)")
        
        # Format results summary
        if not results:
            return "❌ Failed to transcribe any audio segments"
        
        # Save transcriptions to JSON file (output paths already determined above)
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Saving transcriptions to JSON file...")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json_data = {
                'total_segments': len(results),
                'total_processing_time': total_time,
                'language': language,
                'transcriptions': results
            }
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] JSON file save completed in {trace_elapsed:.4f}s")
        
        # Save as simple text format: transcriptions_text.txt
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Saving transcriptions to text file...")
        with open(output_text_path, 'w', encoding='utf-8') as f:
            for result in results:
                # Extract speaker from filename (e.g., segment_001_0220_0270_speaker_0.wav -> speaker_0)
                filename = result['file']
                speaker = "unknown"
                if 'speaker_' in filename:
                    # Extract speaker_X from filename
                    speaker_part = filename.split('speaker_')[-1]
                    speaker_num = speaker_part.split('.')[0].split('_')[0]
                    speaker = f"speaker_{speaker_num}"
                
                # Write in format: speaker_0:transcription text
                f.write(f"{speaker}:{result['transcription']}\n")
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Text file save completed in {trace_elapsed:.4f}s")
        
        print(f"\n✅ Successfully transcribed {len(results)}/{len(audio_files)} segments")
        print(f"⏱️ Total processing time: {total_time:.2f}s")
        print(f"💾 Transcriptions saved to: {output_json_path}")
        print(f"💾 Text format saved to: {output_text_path}\n")
        
        # Normalize path for LLM consumption (use forward slashes)
        output_text_path_llm = normalize_path_for_llm(output_text_path)
        
        # Format return message with continuation instruction
        message = f"\n{'='*80}\n"
        message += f"✅ Transcription Complete\n"
        message += f"{'='*80}\n\n"
        message += f"📊 Transcribed segments: {len(results)}/{len(audio_files)}\n"
        message += f"⏱️  Processing time: {total_time:.2f}s\n"
        message += f"💾 Text file: {output_text_path_llm}\n\n"
        message += f"{'='*80}\n"
        message += "✅ Transcription complete. Continue with the next step in the pipeline.\n"
        message += f"   Use transcriptions_text_path: {output_text_path_llm}\n"
        message += f"{'='*80}\n"
        
        return message
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error during transcription: {str(e)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        return error_msg

