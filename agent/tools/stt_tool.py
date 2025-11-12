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
        
        sensevoice_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model=vad_model,
            vad_kwargs={"max_single_segment_time": max_single_segment_time},
            trust_remote_code=False,
            disable_update=True,
            ban_emo_unk=True,
            device=device_str
        )
        current_sensevoice_loaded = True
        status += f"✅ SenseVoiceSmall loaded successfully on {DEVICE_NAME}!"
        return status
    except Exception as e:
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
    audio_array, sample_rate = load_audio(audio_path)
    if audio_array is None:
        return None
    
    # Run inference
    try:
        start_time = time.time()
        
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
        
        # Extract and format text
        raw_text = result[0]["text"]
        formatted_text = format_str_v3(raw_text)
        
        # Convert to Traditional Chinese using OpenCC
        formatted_text = opencc_converter.convert(formatted_text)
        
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
    
    Args:
        segments_directory: Path to directory containing chopped audio segments
        language: Language code for transcription (default: "yue" for Cantonese)
        vad_model: VAD model to use (default: "fsmn-vad")
        max_single_segment_time: Maximum segment time in milliseconds (default: 30000)
    
    Returns:
        str: Path to the JSON file containing all transcription results
    """
    try:
        # Initialize model if not already loaded
        init_status = initialize_sensevoice_model(vad_model, max_single_segment_time)
        print(init_status)
        
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
        
        print(f"\n{'='*80}")
        print(f"🎙️ Transcribing {len(audio_files)} audio segments...")
        print(f"{'='*80}\n")
        
        results = []
        total_time = 0
        
        for audio_file in audio_files:
            print(f"📝 Processing: {os.path.basename(audio_file)}")
            
            result = transcribe_single_audio_sensevoice(
                audio_path=audio_file,
                language=language,
            )
            
            if result:
                results.append(result)
                total_time += result['processing_time']
                print(f"   ✅ Transcribed in {result['processing_time']:.2f}s")
                print(f"   📄 Text: {result['transcription'][:100]}...")
            else:
                print(f"   ❌ Failed to transcribe")
        
        # Format results summary
        if not results:
            return "❌ Failed to transcribe any audio segments"
        
        # Save transcriptions to JSON file in output/transcriptions/filename/
        # Extract filename from segments_directory path
        segments_folder_name = os.path.basename(os.path.normpath(segments_directory))
        
        # Create output/transcriptions/filename directory
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(segments_directory)), 
            "transcriptions",
            segments_folder_name
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename: transcriptions.json
        output_json_filename = "transcriptions.json"
        output_json_path = os.path.join(output_dir, output_json_filename)
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json_data = {
                'total_segments': len(results),
                'total_processing_time': total_time,
                'language': language,
                'transcriptions': results
            }
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Also save as simple text format: transcriptions_text.txt
        output_text_filename = "transcriptions_text.txt"
        output_text_path = os.path.join(output_dir, output_text_filename)
        
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
        
        print(f"\n✅ Successfully transcribed {len(results)}/{len(audio_files)} segments")
        print(f"⏱️ Total processing time: {total_time:.2f}s")
        print(f"💾 Transcriptions saved to: {output_json_path}")
        print(f"💾 Text format saved to: {output_text_path}\n")
        
        # return output_json_path
        return output_text_path
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error during transcription: {str(e)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        return error_msg

