"""
Unified Gradio GUI combining all phone call analysis tools:
1. Speaker Diarization
2. Audio Chopper
3. Batch Speech-to-Text
4. LLM Analysis
5. Speaker Separation
6. Audio Enhancement
7. LLM Comparison
"""

import os
import gradio as gr
import tempfile
import shutil
import time
import json
from pathlib import Path
import zipfile
import re
import traceback
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import csv

# Import from individual modules
from diarization import diarize_audio
from audio_chopper import chop_audio_file, read_rttm_file
from batch_stt import transcribe_audio, format_str_v3, load_audio
from speaker_separation import perform_diarization, separate_speakers
from audio_enhancement import AudioEnhancer, transcribe_enhanced_audio, FUNASR_AVAILABLE
from funasr import AutoModel
from langchain_ollama import ChatOllama
import torchaudio

# ============================================================================
# Global variables for batch STT
# ============================================================================
model = None
current_model_name = None

# ============================================================================
# 1. SPEAKER DIARIZATION FUNCTIONS
# ============================================================================

# CSV file to cache diarization results
DIARIZATION_CSV = "diarization.csv"

def load_diarization_cache():
    """
    Load cached diarization results from CSV file.
    
    Returns:
        dict: Dictionary mapping filename to cached results
    """
    cache = {}
    if os.path.exists(DIARIZATION_CSV):
        try:
            with open(DIARIZATION_CSV, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cache[row['filename']] = {
                        'rttm_content': row['rttm_content'],
                        'processing_time': float(row['processing_time']),
                        'num_segments': int(row['num_segments']),
                        'num_speakers': int(row['num_speakers']),
                        'speaker_ids': row['speaker_ids'],
                        'timestamp': row['timestamp']
                    }
        except Exception as e:
            print(f"Warning: Could not load diarization cache: {e}")
    return cache

def save_diarization_to_cache(filename, rttm_content, processing_time, num_segments, num_speakers, speaker_ids):
    """
    Save diarization result to CSV cache.
    
    Args:
        filename: Name of the audio file
        rttm_content: RTTM content string
        processing_time: Time taken to process
        num_segments: Number of segments detected
        num_speakers: Number of speakers detected
        speaker_ids: Comma-separated speaker IDs
    """
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(DIARIZATION_CSV)
    
    try:
        with open(DIARIZATION_CSV, 'a', encoding='utf-8', newline='') as f:
            fieldnames = ['filename', 'rttm_content', 'processing_time', 'num_segments', 
                         'num_speakers', 'speaker_ids', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write the data
            writer.writerow({
                'filename': filename,
                'rttm_content': rttm_content,
                'processing_time': processing_time,
                'num_segments': num_segments,
                'num_speakers': num_speakers,
                'speaker_ids': speaker_ids,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    except Exception as e:
        print(f"Warning: Could not save to diarization cache: {e}")

def process_audio(audio_file):
    """
    Process audio file through diarization and return results.
    Checks cache first to avoid reprocessing.
    
    Args:
        audio_file: Audio file from Gradio interface
    
    Returns:
        tuple: (rttm_content, status_message, output_directory_path)
    """
    if audio_file is None:
        return "Please upload an audio file.", "❌ No file uploaded", ""
    
    try:
        filename = os.path.basename(audio_file)
        
        # Check if this file has been processed before
        cache = load_diarization_cache()
        if filename in cache:
            cached = cache[filename]
            
            # Create status message from cached data
            status = f"💾 Loading cached results for: {filename}\n"
            status += f"📅 Previously processed: {cached['timestamp']}\n\n"
            status += "✅ Results loaded from cache!"
            
            summary = f"📈 Summary:\n"
            summary += f"  • Total segments: {cached['num_segments']}\n"
            summary += f"  • Detected speakers: {cached['num_speakers']}\n"
            summary += f"  • Speaker IDs: {cached['speaker_ids']}\n"
            summary += f"  • Original processing time: {cached['processing_time']:.2f} seconds ({cached['processing_time']/60:.2f} minutes)\n"
            summary += f"  • ⚡ Cache hit - instant retrieval!\n\n"
            
            # Create a temporary output directory for consistency
            temp_out_dir = tempfile.mkdtemp(prefix="diarization_cached_")
            
            return cached['rttm_content'], summary + status, temp_out_dir
        
        # Not in cache, proceed with normal processing
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="diarization_")
        
        # Run diarization with timing
        status = f"🔄 Processing audio file: {filename}\n"
        status += f"📁 Output directory: {temp_out_dir}\n\n"
        
        start_time = time.time()
        rttm_content = diarize_audio(
            audio_filepath=audio_file,
            out_dir=temp_out_dir,
            num_speakers=2
        )
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        status += "✅ Diarization completed successfully!"
        
        # Create a summary
        lines = rttm_content.strip().split('\n')
        num_segments = len(lines)
        speakers = set()
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 8:
                    speakers.add(parts[7])
        
        speaker_ids_str = ', '.join(sorted(speakers))
        
        summary = f"📈 Summary:\n"
        summary += f"  • Total segments: {num_segments}\n"
        summary += f"  • Detected speakers: {len(speakers)}\n"
        summary += f"  • Speaker IDs: {speaker_ids_str}\n"
        summary += f"  • Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)\n"
        summary += f"  • 💾 Results saved to cache for future use\n\n"
        
        # Save to cache
        save_diarization_to_cache(
            filename=filename,
            rttm_content=rttm_content,
            processing_time=processing_time,
            num_segments=num_segments,
            num_speakers=len(speakers),
            speaker_ids=speaker_ids_str
        )
        
        return rttm_content, summary + status, temp_out_dir
        
    except Exception as e:
        error_msg = f"❌ Error during diarization: {str(e)}"
        return error_msg, error_msg, ""


# ============================================================================
# 2. AUDIO CHOPPER FUNCTIONS
# ============================================================================

def process_audio_chopping(audio_file, rttm_file, rttm_text):
    padding_ms = 100
    """
    Process audio file and RTTM file to chop audio into segments.
    
    Args:
        audio_file: Audio file from Gradio interface
        rttm_file: RTTM file from Gradio interface
        rttm_text: RTTM text string pasted by user
    
    Returns:
        tuple: (zip_file_path, status_message, segment_details, output_path)
    """
    if audio_file is None:
        return None, "❌ No audio file uploaded", "", ""
    
    if rttm_file is None and (not rttm_text or not rttm_text.strip()):
        return None, "❌ No RTTM file uploaded or RTTM text provided", "", ""
    
    try:
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="chopped_audio_")
        
        status = f"🔄 Processing audio file: {os.path.basename(audio_file)}\n"
        
        # Handle RTTM input - prioritize file over text
        rttm_source = None
        if rttm_file is not None:
            status += f"📄 RTTM file: {os.path.basename(rttm_file)}\n"
            rttm_source = rttm_file
        elif rttm_text and rttm_text.strip():
            status += f"📄 RTTM source: Pasted text\n"
            # Save text to a temporary file
            temp_rttm_file = os.path.join(temp_out_dir, "temp_rttm.rttm")
            with open(temp_rttm_file, 'w', encoding='utf-8') as f:
                f.write(rttm_text.strip())
            rttm_source = temp_rttm_file
        
        status += f"⏱️ Padding: {padding_ms} ms\n"
        status += f"📁 Output directory: {temp_out_dir}\n\n"
        
        start_time = time.time()
        
        # Read RTTM file to get segments
        status += "📖 Reading RTTM data...\n"
        segments = read_rttm_file(rttm_source)
        
        if not segments:
            return None, "❌ No segments found in RTTM file!", ""
        
        status += f"✅ Found {len(segments)} segments\n\n"
        
        # Count speakers
        speakers = set(seg['speaker'] for seg in segments)
        status += f"👥 Detected speakers: {len(speakers)} ({', '.join(sorted(speakers))})\n\n"
        
        # Chop audio file
        status += "✂️ Chopping audio into segments...\n"
        chop_audio_file(audio_file, segments, temp_out_dir, padding_ms)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        status += f"\n✅ Audio chopping completed successfully!\n"
        status += f"⏱️ Processing time: {processing_time:.2f} seconds\n\n"
        
        # List generated files
        output_files = sorted(os.listdir(temp_out_dir))
        file_list = "📁 Generated files:\n"
        for i, fname in enumerate(output_files, 1):
            file_path = os.path.join(temp_out_dir, fname)
            file_size = os.path.getsize(file_path)
            file_list += f"  {i}. {fname} ({file_size:,} bytes)\n"
        
        status += file_list
        
        # Create a zip file of all segments
        zip_path = os.path.join(temp_out_dir, "chopped_segments.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for fname in output_files:
                file_path = os.path.join(temp_out_dir, fname)
                zipf.write(file_path, arcname=fname)
        
        status += f"\n📦 All segments packaged into: chopped_segments.zip\n"
        
        # Create segment details table
        segment_details = "📊 Segment Details:\n\n"
        segment_details += "Segment | Speaker | Start (s) | End (s) | Duration (s)\n"
        segment_details += "--------|---------|-----------|---------|-------------\n"
        for i, seg in enumerate(segments, 1):
            segment_details += f"segment_{i:03d} | {seg['speaker']} | {seg['start']:.2f} | {seg['end']:.2f} | {seg['duration']:.2f}\n"
        
        return zip_path, status, segment_details, zip_path
        
    except Exception as e:
        error_msg = f"❌ Error during audio chopping: {str(e)}"
        error_msg += f"\n\n{traceback.format_exc()}"
        return None, error_msg, "", ""


# ============================================================================
# 3. BATCH SPEECH-TO-TEXT FUNCTIONS
# ============================================================================

def initialize_model():
    """Initialize the SenseVoice model."""
    global model, current_model_name
    
    model_name = "iic/SenseVoiceSmall"
    
    # Only reload if different model
    if current_model_name == model_name and model is not None:
        return f"✅ Model already loaded: {model_name}"
    
    status = f"🔄 Loading SenseVoice model: {model_name}...\n"
    
    try:
        model = AutoModel(
            model=model_name,
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=False,
            disable_update=True,
        )
        current_model_name = model_name
        status += "✅ Model loaded successfully!"
        return status
    except Exception as e:
        status += f"❌ Failed to load model: {str(e)}"
        return status


def transcribe_single_audio(audio_path, language="auto"):
    """Transcribe a single audio file using the global model."""
    global model
    
    if model is None:
        return None
    
    # Load audio
    audio_array, sample_rate = load_audio(audio_path)
    if audio_array is None:
        return None
    
    # Run inference
    try:
        result = model.generate(
            input=audio_array,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True
        )
        
        # Extract and format text
        raw_text = result[0]["text"]
        formatted_text = format_str_v3(raw_text)
        
        return {
            "file": os.path.basename(audio_path),
            "path": audio_path,
            "transcription": formatted_text,
            "raw_transcription": raw_text
        }
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return None


def process_batch_transcription(audio_files, zip_file, link_or_path, language, progress=gr.Progress()):
    """
    Process multiple audio files for transcription.
    
    Args:
        audio_files: List of audio files from Gradio interface
        zip_file: Zip file containing audio files
        link_or_path: URL to a zip file or local folder path
        language: Language code for transcription
    
    Returns:
        tuple: (json_file, txt_file, zip_file, status_message, conversation_content)
    """
    if (not audio_files or len(audio_files) == 0) and not zip_file and not link_or_path:
        return None, None, None, "❌ No audio files uploaded", ""
    
    try:
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="batch_stt_")
        audio_dir = os.path.join(temp_out_dir, "audio_files")
        os.makedirs(audio_dir, exist_ok=True)
        
        status = f"🔄 Starting batch transcription...\n"
        
        # Initialize model
        progress(0, desc="Loading model...")
        model_status = initialize_model()
        status += model_status + "\n\n"
        
        if model is None:
            return None, None, None, status + "\n❌ Failed to initialize model", ""
        
        # Copy uploaded files to temp directory and sort them
        progress(0.1, desc="Preparing files...")
        audio_paths = []
        
        # Handle zip file if provided
        if zip_file is not None:
            status += "📦 Processing zip file...\n"
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Extract all audio files
                    for file_info in zip_ref.filelist:
                        filename = file_info.filename
                        # Skip directories and hidden files
                        if not filename.endswith('/') and not os.path.basename(filename).startswith('.'):
                            # Check if it's an audio file
                            ext = os.path.splitext(filename)[1].lower()
                            if ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus']:
                                # Extract to audio_dir
                                extracted_path = zip_ref.extract(file_info, audio_dir)
                                audio_paths.append(extracted_path)
                                status += f"  ✅ Extracted: {os.path.basename(filename)}\n"
                status += f"\n📊 Total files extracted from zip: {len(audio_paths)}\n\n"
            except Exception as e:
                return None, None, None, f"❌ Error extracting zip file: {str(e)}", ""
        
        # Handle individual audio files
        if audio_files and len(audio_files) > 0:
            for audio_file in audio_files:
                if audio_file is not None:
                    # Get original filename
                    original_name = os.path.basename(audio_file)
                    dest_path = os.path.join(audio_dir, original_name)
                    shutil.copy2(audio_file, dest_path)
                    audio_paths.append(dest_path)
        
        # Handle link or path
        if link_or_path and link_or_path.strip():
            link_or_path = link_or_path.strip()
            
            # Check if it's a URL
            if link_or_path.startswith('http://') or link_or_path.startswith('https://'):
                status += "🔗 Downloading zip file from URL...\n"
                try:
                    # Download the file
                    temp_zip = os.path.join(temp_out_dir, "downloaded.zip")
                    urllib.request.urlretrieve(link_or_path, temp_zip)
                    status += f"  ✅ Downloaded successfully\n"
                    
                    # Extract the zip file
                    status += "📦 Extracting zip file...\n"
                    with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                        for file_info in zip_ref.filelist:
                            filename = file_info.filename
                            if not filename.endswith('/') and not os.path.basename(filename).startswith('.'):
                                ext = os.path.splitext(filename)[1].lower()
                                if ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus']:
                                    extracted_path = zip_ref.extract(file_info, audio_dir)
                                    audio_paths.append(extracted_path)
                                    status += f"  ✅ Extracted: {os.path.basename(filename)}\n"
                    status += f"\n📊 Total files extracted from URL: {len(audio_paths)}\n\n"
                except Exception as e:
                    return None, None, None, f"❌ Error downloading/extracting from URL: {str(e)}", ""
            
            # Check if it's a local folder path
            elif os.path.isdir(link_or_path):
                status += f"📁 Reading audio files from folder: {link_or_path}\n"
                try:
                    # Get all audio files from the folder
                    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus']
                    for root, dirs, files in os.walk(link_or_path):
                        for file in files:
                            ext = os.path.splitext(file)[1].lower()
                            if ext in audio_extensions:
                                source_path = os.path.join(root, file)
                                dest_path = os.path.join(audio_dir, file)
                                shutil.copy2(source_path, dest_path)
                                audio_paths.append(dest_path)
                                status += f"  ✅ Copied: {file}\n"
                    status += f"\n📊 Total files from folder: {len(audio_paths)}\n\n"
                except Exception as e:
                    return None, None, None, f"❌ Error reading from folder: {str(e)}", ""
            
            # Check if it's a local zip file path
            elif os.path.isfile(link_or_path) and link_or_path.lower().endswith('.zip'):
                status += f"📦 Processing local zip file: {link_or_path}\n"
                try:
                    with zipfile.ZipFile(link_or_path, 'r') as zip_ref:
                        for file_info in zip_ref.filelist:
                            filename = file_info.filename
                            if not filename.endswith('/') and not os.path.basename(filename).startswith('.'):
                                ext = os.path.splitext(filename)[1].lower()
                                if ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus']:
                                    extracted_path = zip_ref.extract(file_info, audio_dir)
                                    audio_paths.append(extracted_path)
                                    status += f"  ✅ Extracted: {os.path.basename(filename)}\n"
                    status += f"\n📊 Total files extracted from local zip: {len(audio_paths)}\n\n"
                except Exception as e:
                    return None, None, None, f"❌ Error extracting local zip file: {str(e)}", ""
            else:
                return None, None, None, f"❌ Invalid path or URL: {link_or_path}\nPlease provide a valid URL, folder path, or zip file path.", ""
        
        if len(audio_paths) == 0:
            return None, None, None, "❌ No valid audio files found", ""
        
        status += f"📊 Total files: {len(audio_paths)}\n"
        status += f"🌐 Language: {language}\n"
        status += f"🤖 Model: SenseVoiceSmall\n\n"
        
        # Sort files by name (for segment_001.wav, segment_002.wav ordering)
        audio_paths.sort(key=lambda p: os.path.basename(p))
        
        start_time = time.time()
        
        # Process each file
        results = []
        total_files = len(audio_paths)
        
        status += f"📝 Processing {total_files} audio file(s)...\n\n"
        
        for i, audio_path in enumerate(audio_paths):
            progress((0.1 + 0.7 * (i / total_files)), desc=f"Transcribing {i+1}/{total_files}...")
            
            filename = os.path.basename(audio_path)
            status += f"[{i+1}/{total_files}] {filename}\n"
            
            result = transcribe_single_audio(audio_path, language)
            if result:
                results.append(result)
                status += f"  ✅ {result['transcription'][:100]}{'...' if len(result['transcription']) > 100 else ''}\n\n"
            else:
                status += f"  ❌ Failed to transcribe\n\n"
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        progress(0.9, desc="Saving results...")
        
        # Save results to JSON
        json_path = os.path.join(temp_out_dir, "transcriptions.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save conversation.txt
        txt_path = os.path.join(temp_out_dir, "conversation.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            for r in results:
                fname = r.get('file', '')
                speaker = None
                m = re.search(r"speaker_(\d+)", fname)
                if m:
                    speaker = f"speaker_{m.group(1)}"
                else:
                    speaker = Path(fname).stem if fname else 'unknown'
                f.write(f"{speaker}: {r.get('transcription', '')}\n")
        
        # Create a zip file with all results
        zip_path = os.path.join(temp_out_dir, "batch_transcription_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(json_path, arcname="transcriptions.json")
            zipf.write(txt_path, arcname="conversation.txt")
        
        progress(1.0, desc="Complete!")
        
        status += f"\n{'='*60}\n"
        status += f"✅ Batch transcription completed!\n"
        status += f"⏱️ Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)\n"
        status += f"📊 Successfully processed: {len(results)}/{total_files} files\n"
        status += f"📁 Results saved to:\n"
        status += f"   • transcriptions.json\n"
        status += f"   • conversation.txt\n"
        status += f"   • batch_transcription_results.zip\n"
        status += f"{'='*60}\n"
        
        # Read conversation.txt content
        conversation_content = ""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                conversation_content = f.read()
        except Exception as e:
            conversation_content = f"Error reading conversation.txt: {str(e)}"
        
        return json_path, txt_path, zip_path, status, conversation_content
        
    except Exception as e:
        error_msg = f"❌ Error during batch transcription: {str(e)}"
        error_msg += f"\n\n{traceback.format_exc()}"
        return None, None, None, error_msg, ""


# ============================================================================
# 4. LLM ANALYSIS FUNCTIONS
# ============================================================================

# Common model options
MODEL_OPTIONS = [
    "qwen3:30b",
    "gpt-oss:20b",
    "gemma3-27b",
    "deepseek-r1:14b",
    "deepseek-r1:32b",
]

DEFAULT_MODEL = MODEL_OPTIONS[0]
DEFAULT_OLLAMA_URL = "http://192.168.61.2:11434"
DEFAULT_SYSTEM_MESSAGE = (
    "你是一位精通粵語以及香港股市的分析師。請用繁體中文回應，"
    "並從下方對話中判斷誰是券商、誰是客戶，整理最終下單（股票代號、買/賣、價格、數量），"
)


def parse_metadata(metadata_text: str) -> dict[str, str]:
    """
    Parse metadata from pasted text format.
    
    Expected format:
        Broker Name: Dickson Lau
        Broker Id: 0489
        Client Number: 97501167
        Client Name: CHAN CHO WING and CHAN MAN LEE
        Client Id: P77751
        UTC: 2025-10-10T01:45:10
        HKT: 2025-10-10T09:45:10
    
    Returns:
        dict: Dictionary with keys: broker_name, broker_id, client_number, 
              client_id, client_name, utc_time, hkt_time
    """
    result = {
        "broker_name": "",
        "broker_id": "",
        "client_number": "",
        "client_id": "",
        "client_name": "",
        "utc_time": "",
        "hkt_time": ""
    }
    
    if not metadata_text or not metadata_text.strip():
        return result
    
    # Parse line by line
    lines = metadata_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if ':' not in line:
            continue
            
        # Split only on first colon to handle values that contain colons (like times)
        key, value = line.split(':', 1)
        key = key.strip().lower()
        value = value.strip()
        
        # Map the keys to result dictionary
        if 'broker name' in key:
            result['broker_name'] = value
        elif 'broker id' in key:
            result['broker_id'] = value
        elif 'client number' in key:
            result['client_number'] = value
        elif 'client id' in key:
            result['client_id'] = value
        elif 'client name' in key:
            result['client_name'] = value
        elif key == 'utc':
            result['utc_time'] = value
        elif key == 'hkt':
            result['hkt_time'] = value
    
    return result


def analyze_with_llm(
    prompt_text: str,
    prompt_file,
    model: str,
    ollama_url: str,
    system_message: str,
    temperature: float,
    metadata_text: str = "",
) -> tuple[str, str]:
    """
    Analyze text with LLM
    
    Returns:
        tuple: (status_message, response_text)
    """
    try:
        # Determine the prompt source
        final_prompt = None
        
        if prompt_file is not None:
            # Read from uploaded file
            try:
                file_path = Path(prompt_file.name)
                final_prompt = file_path.read_text(encoding="utf-8")
                status = f"✓ Loaded prompt from file: {file_path.name}"
            except Exception as e:
                return f"❌ Error reading file: {str(e)}", ""
        elif prompt_text and prompt_text.strip():
            # Use text input
            final_prompt = prompt_text.strip()
            status = "✓ Using text input"
        else:
            return "❌ Error: Please provide either text input or upload a file", ""
        
        # Validate inputs
        if not model or not model.strip():
            return "❌ Error: Please specify a model name", ""
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL", ""
        
        # Parse metadata from the pasted text
        metadata_dict = parse_metadata(metadata_text)
        
        # Build metadata context if any fields are provided
        metadata_lines = []
        if metadata_dict['broker_name']:
            metadata_lines.append(f"Broker Name: {metadata_dict['broker_name']}")
        if metadata_dict['broker_id']:
            metadata_lines.append(f"Broker Id: {metadata_dict['broker_id']}")
        if metadata_dict['client_number']:
            metadata_lines.append(f"Client Number: {metadata_dict['client_number']}")
        if metadata_dict['client_name']:
            metadata_lines.append(f"Client Name: {metadata_dict['client_name']}")
        if metadata_dict['client_id']:
            metadata_lines.append(f"Client Id: {metadata_dict['client_id']}")
        if metadata_dict['utc_time']:
            metadata_lines.append(f"UTC: {metadata_dict['utc_time']}")
        if metadata_dict['hkt_time']:
            metadata_lines.append(f"HKT: {metadata_dict['hkt_time']}")
        
        # Prepend metadata to system message if available
        final_system_message = system_message
        if metadata_lines:
            metadata_context = "\n".join(metadata_lines)
            final_system_message = f"{system_message}\n\n以下是對話的資料背景, 可能會幫助你分析對話:\n{metadata_context}"
        
        print(f"{'*' * 30} final_system_message: {final_system_message} {'*' * 30}")
        
        # Initialize the LLM
        status += f"\n✓ Connecting to Ollama at: {ollama_url}"
        status += f"\n✓ Using model: {model}"
        status += f"\n✓ Temperature: {temperature}"
        
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=temperature,
        )
        
        # Prepare messages
        messages = [
            ("system", final_system_message),
            ("human", final_prompt),
        ]
        
        status += "\n✓ Sending request to LLM..."
        
        # Get response
        resp = chat_llm.invoke(messages)
        
        # Extract content
        try:
            response_content = getattr(resp, "content", str(resp))
        except Exception:
            response_content = str(resp)
        
        status += "\n✓ Analysis complete!"
        
        # return status, response_content
        return response_content
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, ""


def load_example_file():
    """Load an example transcription file if available"""
    example_paths = [
        Path("demo/transcriptions/conversation.txt"),
        Path("output/transcriptions/conversation.txt"),
    ]
    
    for path in example_paths:
        if path.exists():
            return path.read_text(encoding="utf-8")
    
    return "請輸入電話對話記錄文本，或上傳文件。"


# ============================================================================
# 5. SPEAKER SEPARATION FUNCTIONS
# ============================================================================

def create_speaker_audio_separation(audio_tensor, sample_rate, segments, target_speaker):
    """
    Create audio with only the target speaker, silencing others.
    
    Args:
        audio_tensor: torch.Tensor - audio waveform [channels, samples]
        sample_rate: int - sample rate of audio
        segments: list - speaker segments from RTTM
        target_speaker: str - speaker to keep (e.g., 'speaker_0')
    
    Returns:
        torch.Tensor - modified audio with only target speaker
    """
    # Clone the audio tensor
    output_audio = audio_tensor.clone()
    
    # Get total duration
    total_samples = audio_tensor.shape[-1]
    
    # Create a mask for silence (1 = keep, 0 = silence)
    mask = torch.zeros(total_samples)
    
    # Mark regions where target speaker is talking
    for segment in segments:
        if segment['speaker'] == target_speaker:
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            # Ensure we don't exceed boundaries
            start_sample = max(0, start_sample)
            end_sample = min(total_samples, end_sample)
            mask[start_sample:end_sample] = 1.0
    
    # Apply mask to audio
    output_audio = output_audio * mask.unsqueeze(0)
    
    return output_audio


def parse_rttm_for_separation(rttm_path):
    """Parse RTTM file to extract speaker segments."""
    segments = []
    
    with open(rttm_path, 'r') as f:
        for line in f:
            if line.startswith('SPEAKER'):
                parts = line.strip().split()
                speaker = parts[7]
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                
                segments.append({
                    'speaker': speaker,
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })
    
    return segments


def process_speaker_separation(audio_file, progress=gr.Progress()):
    """
    Perform speaker separation on audio file.
    Reuses logic from speaker_separation.py for reliability.
    
    Args:
        audio_file: Audio file from Gradio interface
    
    Returns:
        tuple: (speaker_0_file, speaker_1_file, status_message)
    """
    if audio_file is None:
        return None, None, "[ERROR] No audio file uploaded"
    
    try:
        import soundfile as sf
        
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="speaker_separated_")
        
        status = "=" * 60 + "\n"
        status += "SPEAKER SEPARATION TOOL\n"
        status += "=" * 60 + "\n"
        status += f"Input audio: {os.path.basename(audio_file)}\n"
        status += f"Output directory: {temp_out_dir}\n\n"
        
        overall_start_time = time.time()
        
        # ================================================================
        # STEP 1: Perform diarization
        # ================================================================
        progress(0.1, desc="Starting diarization...")
        
        status += "=" * 60 + "\n"
        status += "STEP 1: Speaker Diarization with NVIDIA NeMo\n"
        status += "=" * 60 + "\n\n"
        
        try:
            # Use the perform_diarization function from speaker_separation module
            rttm_file = perform_diarization(audio_file, temp_out_dir)
            status += f"[OK] Diarization complete!\n"
            status += f"[OK] RTTM file generated: {os.path.basename(rttm_file)}\n\n"
        except Exception as e:
            error_msg = f"[ERROR] Error during diarization: {str(e)}\n\n{traceback.format_exc()}"
            return None, None, status + error_msg
        
        # ================================================================
        # STEP 2: Separate speakers
        # ================================================================
        progress(0.4, desc="Loading audio...")
        
        status += "=" * 60 + "\n"
        status += "STEP 2: Speaker Separation\n"
        status += "=" * 60 + "\n\n"
        
        # Load audio using soundfile
        status += f"[*] Loading audio: {os.path.basename(audio_file)}\n"
        try:
            audio_data, sample_rate = sf.read(str(audio_file), dtype='float32')
            
            # Convert to torch tensor and ensure correct shape [channels, samples]
            if audio_data.ndim == 1:
                # Mono audio
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                # Stereo or multi-channel: transpose to [channels, samples]
                waveform = torch.from_numpy(audio_data.T)
            
            status += f"[OK] Audio loaded: shape={waveform.shape}, sample_rate={sample_rate} Hz\n"
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                status += "[*] Converting stereo to mono...\n"
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                status += "[OK] Converted to mono\n"
            
            status += "\n"
            
        except Exception as e:
            error_msg = f"[ERROR] Error loading audio: {str(e)}\n{traceback.format_exc()}"
            return None, None, status + error_msg
        
        progress(0.5, desc="Parsing diarization results...")
        
        # Parse RTTM file
        status += f"[*] Parsing diarization results from: {os.path.basename(rttm_file)}\n"
        try:
            segments = parse_rttm_for_separation(rttm_file)
            
            if not segments:
                error_msg = "[ERROR] No speaker segments found in RTTM file!"
                return None, None, status + error_msg
            
            # Get unique speakers
            speakers = sorted(set(seg['speaker'] for seg in segments))
            status += f"[OK] Found {len(speakers)} speaker(s): {', '.join(speakers)}\n\n"
            
            # Print segment summary
            status += "[*] Segment Summary:\n"
            for speaker in speakers:
                speaker_segments = [s for s in segments if s['speaker'] == speaker]
                total_duration = sum(s['duration'] for s in speaker_segments)
                status += f"  {speaker}: {len(speaker_segments)} segments, {total_duration:.2f}s total\n"
            status += "\n"
            
        except Exception as e:
            error_msg = f"[ERROR] Error parsing RTTM: {str(e)}\n{traceback.format_exc()}"
            return None, None, status + error_msg
        
        progress(0.6, desc="Creating separated audio files...")
        
        # Separate each speaker
        status += "[*] Creating separated audio files...\n\n"
        speaker_files = []
        
        for i, speaker in enumerate(speakers):
            try:
                progress(0.6 + (0.3 * i / len(speakers)), desc=f"Processing speaker {i}...")
                
                status += f"  Processing speaker_{i} ({speaker})...\n"
                
                # Create audio with only this speaker
                speaker_audio = create_speaker_audio_separation(waveform, sample_rate, segments, speaker)
                
                # Save to file using soundfile
                output_file = os.path.join(temp_out_dir, f"speaker_{i}_only.wav")
                
                # Convert tensor to numpy
                audio_numpy = speaker_audio.numpy()
                
                # Ensure audio_numpy is in the correct format for soundfile
                # soundfile expects [samples] for mono or [samples, channels] for multi-channel
                if audio_numpy.ndim == 2:
                    # If shape is [1, samples], transpose to [samples, 1]
                    # If shape is [channels, samples], transpose to [samples, channels]
                    if audio_numpy.shape[0] <= 2:  # Likely [channels, samples] format
                        audio_numpy = audio_numpy.T
                
                # If we have [samples, 1] format, convert to [samples] for mono
                if audio_numpy.ndim == 2 and audio_numpy.shape[1] == 1:
                    audio_numpy = audio_numpy.squeeze()
                
                status += f"    - Audio shape for saving: {audio_numpy.shape}\n"
                sf.write(str(output_file), audio_numpy, sample_rate)
                
                speaker_files.append(output_file)
                
                # Calculate non-silent duration
                speaker_segments = [s for s in segments if s['speaker'] == speaker]
                active_duration = sum(s['duration'] for s in speaker_segments)
                total_duration = waveform.shape[-1] / sample_rate
                
                status += f"  [OK] Saved: speaker_{i}_only.wav\n"
                status += f"    - Total duration: {total_duration:.2f}s\n"
                status += f"    - Active speech: {active_duration:.2f}s ({active_duration/total_duration*100:.1f}%)\n"
                status += f"    - Silenced: {total_duration - active_duration:.2f}s\n\n"
                
            except Exception as e:
                status += f"  [ERROR] Error processing speaker_{i}: {str(e)}\n"
                status += f"  {traceback.format_exc()}\n\n"
                # Continue to next speaker instead of stopping completely
                continue
        
        # ================================================================
        # Final summary
        # ================================================================
        progress(1.0, desc="Complete!")
        
        overall_end_time = time.time()
        processing_time = overall_end_time - overall_start_time
        
        status += "=" * 60 + "\n"
        status += "[SUCCESS] COMPLETE!\n"
        status += "=" * 60 + "\n"
        status += f"[*] Total processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)\n"
        status += f"[*] Separated files created: {len(speaker_files)}\n\n"
        
        status += "[*] Files created:\n"
        for speaker_file in speaker_files:
            file_size = os.path.getsize(speaker_file)
            status += f"  - {os.path.basename(speaker_file)} ({file_size:,} bytes)\n"
        
        # Return files (up to 2 speakers for now)
        speaker_0 = speaker_files[0] if len(speaker_files) > 0 else None
        speaker_1 = speaker_files[1] if len(speaker_files) > 1 else None
        
        # Add note if more than 2 speakers
        if len(speaker_files) > 2:
            status += f"\n[WARNING] Note: {len(speaker_files)} speakers detected. Only first 2 available for download in GUI.\n"
            status += f"   All files saved in: {temp_out_dir}\n"
        
        status += "\n"
        
        return speaker_0, speaker_1, status
        
    except Exception as e:
        error_msg = f"[ERROR] Error during speaker separation: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, error_msg


# ============================================================================
# 6. AUDIO ENHANCEMENT FUNCTIONS
# ============================================================================

def process_audio_enhancement(audio_file, enable_transcription=False, language="yue", progress=gr.Progress()):
    """
    Process audio enhancement.
    
    Args:
        audio_file: Audio file from Gradio interface
        enable_transcription: Whether to transcribe after enhancement
        language: Language for transcription
        
    Returns:
        tuple: (enhanced_audio_path, transcription_text, status_message)
    """
    if audio_file is None:
        return None, "", "❌ No audio file uploaded"
    
    try:
        # Create temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="audio_enhanced_")
        
        status = f"🔄 Processing audio file: {os.path.basename(audio_file)}\n"
        status += f"📁 Output directory: {temp_out_dir}\n\n"
        
        progress(0.1, desc="Initializing...")
        
        # Create enhancer
        enhancer = AudioEnhancer(target_sr=16000)
        
        # Set output path
        output_path = os.path.join(temp_out_dir, "enhanced_audio.wav")
        
        progress(0.2, desc="Enhancing audio...")
        
        # Enhance audio
        status += "=" * 60 + "\n"
        status += "🎵 Audio Enhancement Pipeline\n"
        status += "=" * 60 + "\n\n"
        
        try:
            # Run enhancement with detailed steps
            status += "📂 Loading audio...\n"
            waveform, sr = enhancer.load_audio(audio_file)
            status += f"✓ Loaded: {waveform.shape}, {sr} Hz\n\n"
            
            progress(0.3, desc="High-pass filtering...")
            status += "🔧 Step 1: High-pass filter (remove rumble)...\n"
            waveform = enhancer.apply_highpass_filter(waveform, sr, cutoff=80)
            status += "✓ Applied high-pass filter at 80 Hz\n\n"
            
            progress(0.4, desc="Low-pass filtering...")
            status += "🔧 Step 2: Low-pass filter (remove hiss)...\n"
            waveform = enhancer.apply_lowpass_filter(waveform, sr, cutoff=8000)
            status += "✓ Applied low-pass filter at 8000 Hz\n\n"
            
            progress(0.5, desc="Noise reduction...")
            status += "🔧 Step 3: Noise reduction...\n"
            waveform = enhancer.reduce_noise(waveform, sr)
            status += "✓ Applied spectral noise reduction\n\n"
            
            progress(0.6, desc="Speech enhancement...")
            status += "🔧 Step 4: Speech frequency enhancement...\n"
            waveform = enhancer.apply_speech_enhancement(waveform, sr)
            status += "✓ Enhanced speech frequencies (300-3400 Hz)\n\n"
            
            progress(0.7, desc="Compression...")
            status += "🔧 Step 5: Dynamic range compression...\n"
            waveform = enhancer.apply_dynamic_range_compression(waveform)
            status += "✓ Applied compression (helps with volume variations)\n\n"
            
            progress(0.8, desc="Normalizing...")
            status += "🔧 Step 6: Final normalization...\n"
            waveform = enhancer.normalize_audio(waveform, target_level=-20.0)
            status += "✓ Normalized to -20 dBFS\n\n"
            
            # Save
            status += "💾 Saving enhanced audio...\n"
            torchaudio.save(output_path, waveform, sr)
            status += f"✓ Saved: {output_path}\n\n"
            
            # Calculate sizes
            input_size = os.path.getsize(audio_file) / 1024
            output_size = os.path.getsize(output_path) / 1024
            status += "📊 Summary:\n"
            status += f"  Input size:  {input_size:.2f} KB\n"
            status += f"  Output size: {output_size:.2f} KB\n"
            status += f"  Sample rate: {sr} Hz\n"
            status += "=" * 60 + "\n\n"
            
        except Exception as e:
            error_msg = f"❌ Error during enhancement: {str(e)}\n{traceback.format_exc()}"
            return None, "", status + error_msg
        
        # Transcribe if requested
        transcription = ""
        if enable_transcription:
            progress(0.9, desc="Transcribing...")
            status += "=" * 60 + "\n"
            status += "📝 Transcription with SenseVoice\n"
            status += "=" * 60 + "\n"
            status += f"Language: {language}\n\n"
            
            try:
                transcription = transcribe_enhanced_audio(
                    output_path,
                    language=language
                )
                status += "✓ Transcription complete\n\n"
                
                # Save transcription
                transcript_path = os.path.join(temp_out_dir, "transcription.txt")
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                status += f"💾 Transcription saved: {transcript_path}\n"
                
            except Exception as e:
                status += f"❌ Error during transcription: {str(e)}\n"
                transcription = f"Error: {str(e)}"
        
        progress(1.0, desc="Complete!")
        
        status += "\n" + "=" * 60 + "\n"
        status += "✅ COMPLETE!\n"
        status += "=" * 60 + "\n"
        
        return output_path, transcription, status
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        error_msg += f"\n\n{traceback.format_exc()}"
        return None, "", error_msg


# ============================================================================
# 7. LLM COMPARISON FUNCTIONS
# ============================================================================

def analyze_single_model(
    model: str,
    prompt: str,
    ollama_url: str,
    system_message: str,
    temperature: float,
) -> Tuple[str, str, float]:
    """
    Analyze text with a single LLM model
    
    Returns:
        tuple: (model_name, response_text, elapsed_time)
    """
    start_time = time.time()
    
    try:
        # Initialize the LLM
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=temperature,
        )
        
        # Prepare messages
        messages = [
            ("system", system_message),
            ("human", prompt),
        ]
        
        # Get response
        resp = chat_llm.invoke(messages)
        
        # Extract content
        try:
            response_content = getattr(resp, "content", str(resp))
        except Exception:
            response_content = str(resp)
        
        elapsed_time = time.time() - start_time
        
        return model, response_content, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return model, error_msg, elapsed_time


def compare_models(
    prompt_text: str,
    prompt_file,
    selected_models: List[str],
    ollama_url: str,
    system_message: str,
    temperature: float,
) -> Tuple[str, Dict[str, Tuple[str, float]]]:
    """
    Analyze text with multiple LLM models in parallel
    
    Returns:
        tuple: (status_message, dict of {model_name: (response, elapsed_time)})
    """
    try:
        # Determine the prompt source
        final_prompt = None
        
        if prompt_file is not None:
            # Read from uploaded file
            try:
                file_path = Path(prompt_file.name)
                final_prompt = file_path.read_text(encoding="utf-8")
                status = f"✓ Loaded prompt from file: {file_path.name}\n"
            except Exception as e:
                return f"❌ Error reading file: {str(e)}", {}
        elif prompt_text and prompt_text.strip():
            # Use text input
            final_prompt = prompt_text.strip()
            status = "✓ Using text input\n"
        else:
            return "❌ Error: Please provide either text input or upload a file", {}
        
        # Validate inputs
        if not selected_models or len(selected_models) == 0:
            return "❌ Error: Please select at least one model", {}
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL", {}
        
        status += f"✓ Running {len(selected_models)} model(s) in parallel...\n"
        status += f"✓ Ollama URL: {ollama_url}\n"
        status += f"✓ Temperature: {temperature}\n"
        status += f"✓ Models: {', '.join(selected_models)}\n"
        status += "\n" + "="*50 + "\n"
        
        # Run models in parallel
        results = {}
        total_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(
                    analyze_single_model,
                    model,
                    final_prompt,
                    ollama_url,
                    system_message,
                    temperature
                ): model for model in selected_models
            }
            
            # Process completed tasks
            for future in as_completed(future_to_model):
                model_name, response, elapsed = future.result()
                results[model_name] = (response, elapsed)
                status += f"✓ {model_name} completed in {elapsed:.2f}s\n"
        
        total_elapsed = time.time() - total_start_time
        status += "="*50 + "\n"
        status += f"✓ All models completed in {total_elapsed:.2f}s\n"
        status += f"✓ Average time per model: {total_elapsed/len(selected_models):.2f}s\n"
        
        return status, results
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, {}


def format_comparison_results(results: Dict[str, Tuple[str, float]]) -> List[Tuple[str, str, str]]:
    """
    Format results for display in comparison boxes
    
    Returns:
        list: List of tuples (model_name, time_info, response), one per model
    """
    if not results:
        return []
    
    formatted = []
    for model, (response, elapsed) in results.items():
        model_name = f"🤖 {model}"
        time_info = f"⏱️ {elapsed:.2f} 秒"
        formatted.append((model_name, time_info, response))
    
    return formatted


# ============================================================================
# 8. FILE METADATA EXTRACTION FUNCTIONS
# ============================================================================

def parse_filename_metadata(filename: str, csv_path: str = "client.csv") -> str:
    """
    Parse audio filename to extract metadata and format output.
    
    Expected filename format:
    [Broker Name Broker_ID]_Unknown1-ClientPhone_YYYYMMDDHHMMSS(Unknown2).wav
    or
    [Broker Name]_Unknown1-ClientPhone_YYYYMMDDHHMMSS(Unknown2).wav
    
    Args:
        filename: The audio filename to parse
        csv_path: Path to client.csv file for name lookup
        
    Returns:
        Formatted string with extracted metadata
    """
    try:
        # Remove extension
        base_name = os.path.splitext(filename)[0]
        
        # Parse filename using regex
        # Pattern 1 (with brackets and parentheses): [Broker Name Optional_ID]_Unknown1-ClientPhone_DateTime(Unknown2)
        pattern1 = r'\[(.*?)\]_(\d+)-(\d+)_(\d{14})\((\d+)\)'
        match = re.match(pattern1, base_name)
        
        # Pattern 2 (sanitized by Gradio - no brackets or parentheses): Broker Name Optional_ID_Unknown1-ClientPhone_DateTime Unknown2
        # Example: "Dickson Lau 0489_8330-97501167_2025101001451020981"
        if not match:
            pattern2 = r'(.*?)_(\d+)-(\d+)_(\d{14})(\d+)'
            match = re.match(pattern2, base_name)
        
        if not match:
            return f"""
            ❌ Error: Filename does not match expected pattern.
            
            Expected format:
            [Broker Name ID]_8330-97501167_20251010014510(20981).wav
            
            Received:
            {filename}
            
            Note: The filename may have been sanitized by the system. 
            Please ensure special characters are preserved or manually enter the correct format.
            """
        
        broker_info = match.group(1)  # e.g., "Dickson Lau 0489" or "Dickson Lau"
        # unknown_1 = match.group(2)     # e.g., "8330"
        client_number = match.group(3) # e.g., "97501167"
        datetime_str = match.group(4)  # e.g., "20251010014510"
        # unknown_2 = match.group(5)     # e.g., "20981"
        
        # Parse broker name and ID
        broker_parts = broker_info.rsplit(' ', 1)  # Split from right to handle multi-word names
        if len(broker_parts) == 2 and broker_parts[1].isdigit():
            broker_name = broker_parts[0]
            broker_id = broker_parts[1]
        else:
            broker_name = broker_info
            broker_id = "N/A"
        
        # Parse datetime (UTC)
        try:
            utc_dt = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
            # Convert to HKT (UTC+8)
            hkt_dt = utc_dt + timedelta(hours=8)
            
            utc_formatted = utc_dt.strftime("%Y-%m-%dT%H:%M:%S")
            hkt_formatted = hkt_dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError as e:
            return f"❌ Error parsing datetime: {str(e)}"
        
        # Look up client name and ID in CSV
        client_name = "Not found"
        client_id = "Not found"
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as csvfile:
                    csv_reader = csv.DictReader(csvfile)
                    for row in csv_reader:
                        # Find name column
                        name_col = None
                        for col in row.keys():
                            if 'name' in col.lower() or 'client' in col.lower():
                                name_col = col
                                break
                        
                        # Check if phone number matches any of the three phone columns
                        # CSV format: AE,acctno,name,mobile,home,office
                        phone_columns = ['mobile', 'home', 'office']
                        for phone_col in phone_columns:
                            if phone_col in row and row[phone_col].strip() == client_number:
                                if name_col:
                                    client_name = row[name_col].strip()
                                # Get client ID (acctno column)
                                if 'acctno' in row:
                                    client_id = row['acctno'].strip()
                                break
                        
                        # If we found the client, exit the loop
                        if client_name != "Not found":
                            break
            except Exception as e:
                client_name = f"Error reading CSV: {str(e)}"
                client_id = "Error reading CSV"
        else:
            client_name = "client.csv not found"
            client_id = "client.csv not found"
        
        # Format output as JSON
        # result_dict = {
        #     "broker_name": broker_name,
        #     "broker_id": broker_id,
        #     "client_number": client_number,
        #     "client_name": client_name,
        #     "client_id": client_id,
        #     "utc": utc_formatted,
        #     "hkt": hkt_formatted
        # }
        # output = json.dumps(result_dict, indent=2, ensure_ascii=False)
        
        # Format output
        output = f"""
Broker Name: {broker_name}
Broker Id: {broker_id}
Client Number: {client_number}
Client Name: {client_name}
Client Id: {client_id}
UTC: {utc_formatted}
HKT: {hkt_formatted}
"""
        
        return output
        
    except Exception as e:
        error_msg = f"❌ Error parsing filename: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg


def process_file_metadata(audio_file):
    """
    Process uploaded audio file and extract metadata from filename.
    
    Args:
        audio_file: Audio file from Gradio interface
        
    Returns:
        str: Formatted metadata string
    """
    if audio_file is None:
        return "❌ No file uploaded. Please drag and drop an audio file."
    
    try:
        # Get filename
        filename = os.path.basename(audio_file)
        
        # Parse metadata
        result = parse_filename_metadata(filename)
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg


# ============================================================================
# UNIFIED GRADIO INTERFACE
# ============================================================================

def create_unified_interface():
    """Create the unified Gradio interface with all tools in tabs."""
    
    with gr.Blocks(title="Phone Call Analysis Suite", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📞 Phone Call Analysis Suite
            ### All-in-one tool for speaker diarization, audio processing, transcription, and analysis
            """
        )
        
        with gr.Tabs():
            
            # ================================================================
            # TAB 8: FILE METADATA EXTRACTION
            # ================================================================
            with gr.Tab("8️⃣ File Metadata"):
                gr.Markdown("### Extract metadata from audio filename")
                gr.Markdown("*Parse filename to extract broker info, client info, and timestamps*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Input")
                        meta_audio_input = gr.File(
                            label="Upload Audio File",
                            type="filepath",
                            file_types=["audio"]
                        )
                        meta_process_btn = gr.Button("📋 Extract Metadata", variant="primary", size="lg")
                        
                        gr.Markdown("""
                        #### Expected Filename Format:
                        ```
                        [Broker Name ID]_8330-97501167_20251010014510(20981).wav
                        ```
                        
                        #### Components:
                        - **[Broker Name ID]**: Broker's name and optional ID
                        - **8330**: Unknown field 1
                        - **97501167**: Client phone number
                        - **20251010014510**: UTC timestamp (YYYYMMDDHHMMSS)
                        - **(20981)**: Unknown field 2
                        
                        #### Example:
                        `[Dickson Lau 0489]_8330-97501167_20251010014510(20981).wav`
                        
                        #### Client Lookup:
                        Place a `client.csv` file in the same directory with columns for phone number and client name.
                        The system will automatically look up the client name based on the phone number.
                        """)
                        
                    with gr.Column(scale=2):
                        gr.Markdown("#### Extracted Metadata")
                        meta_output = gr.Textbox(
                            label="Results",
                            lines=25,
                            max_lines=35,
                            interactive=False,
                            show_copy_button=True
                        )
                
                meta_process_btn.click(
                    fn=process_file_metadata,
                    inputs=[meta_audio_input],
                    outputs=[meta_output]
                )

            # ================================================================
            # TAB 1: SPEAKER DIARIZATION
            # ================================================================
            with gr.Tab("1️⃣ Speaker Diarization"):
                gr.Markdown("### Identify and separate speakers in audio files")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Input")
                        diar_audio_input = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            sources=["upload"]
                        )
                        diar_process_btn = gr.Button("🚀 Start Diarization", variant="primary", size="lg")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("#### Results")
                        diar_status_output = gr.Textbox(
                            label="Status",
                            lines=8,
                            max_lines=15,
                            interactive=False
                        )
                        diar_rttm_output = gr.Textbox(
                            label="RTTM Output (Rich Time-Marked Text)",
                            lines=15,
                            max_lines=30,
                            interactive=False,
                            show_copy_button=True
                        )
                        diar_output_dir = gr.Textbox(
                            label="Output Directory Path",
                            interactive=False,
                            visible=True
                        )
                
                diar_process_btn.click(
                    fn=process_audio,
                    inputs=[diar_audio_input],
                    outputs=[diar_rttm_output, diar_status_output, diar_output_dir]
                )
            
            # ================================================================
            # TAB 2: AUDIO CHOPPER
            # ================================================================
            with gr.Tab("2️⃣ Audio Chopper"):
                gr.Markdown("### Chop audio files into speaker segments based on RTTM")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Input Files")
                        chop_audio_input = gr.Audio(
                            label="Audio File",
                            type="filepath",
                            sources=["upload"]
                        )
                        
                        gr.Markdown("#### RTTM Input (Choose one method)")
                        
                        with gr.Tab("📝 Paste RTTM Text"):
                            chop_rttm_text = gr.Textbox(
                                label="RTTM Content",
                                placeholder="Paste RTTM content here...\nExample:\nSPEAKER test 1 0.000 2.500 <NA> <NA> speaker_0 <NA> <NA>\nSPEAKER test 1 2.500 3.200 <NA> <NA> speaker_1 <NA> <NA>",
                                lines=10,
                                max_lines=20
                            )

                        with gr.Tab("📁 Upload RTTM File"):
                            chop_rttm_input = gr.File(
                                label="RTTM File",
                                file_types=[".rttm"],
                                file_count="single"
                            )
                        
                        chop_process_btn = gr.Button("✂️ Chop Audio", variant="primary", size="lg")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("#### Results")
                        chop_status_output = gr.Textbox(
                            label="Status",
                            lines=12,
                            max_lines=20,
                            interactive=False
                        )
                        chop_segment_details = gr.Textbox(
                            label="Segment Details",
                            lines=10,
                            max_lines=20,
                            interactive=False,
                            show_copy_button=True
                        )
                        chop_download_output = gr.File(
                            label="Download Chopped Segments (ZIP)",
                            interactive=False
                        )
                        chop_output_path = gr.Textbox(
                            label="Output File Path",
                            interactive=False,
                            show_copy_button=True
                        )
                
                chop_process_btn.click(
                    fn=process_audio_chopping,
                    inputs=[chop_audio_input, chop_rttm_input, chop_rttm_text],
                    outputs=[chop_download_output, chop_status_output, chop_segment_details, chop_output_path]
                )
            
            # ================================================================
            # TAB 3: BATCH SPEECH-TO-TEXT
            # ================================================================
            with gr.Tab("3️⃣ Batch Speech-to-Text"):
                gr.Markdown("### Transcribe multiple audio segments to text")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Input (Choose one or both methods)")
                        
                        with gr.Tab("🔗 Paste Link or Folder Path"):
                            stt_link_or_path = gr.Textbox(
                                label="Zip File URL or Folder Path",
                                placeholder="Paste a URL to a zip file (e.g., https://example.com/audio.zip)\nor a local folder path (e.g., C:/Users/me/audio_files)",
                                lines=2,
                                max_lines=3
                            )
                            gr.Markdown("*Provide either a direct URL to a zip file or a local folder path containing audio files*")
                        
                        with gr.Tab("📁 Upload Audio Files"):
                            stt_audio_files = gr.File(
                                label="Upload Audio Files",
                                file_count="multiple",
                                file_types=[".wav", ".mp3", ".flac", ".m4a"],
                                type="filepath"
                            )
                            
                        with gr.Tab("📦 Upload Zip File"):
                            stt_zip_file = gr.File(
                                label="Upload Zip File Containing Audio Files",
                                file_count="single",
                                file_types=[".zip"],
                                type="filepath"
                            )
                            gr.Markdown("*The zip file should contain audio files (.wav, .mp3, .flac, .m4a, .ogg, .opus)*")
                        
                        stt_language_dropdown = gr.Dropdown(
                            choices=["auto", "zh", "en", "yue", "ja", "ko"],
                            value="yue",
                            label="Language",
                            info="Select the language of the audio"
                        )
                        stt_process_btn = gr.Button("🚀 Start Transcription", variant="primary", size="lg")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("#### Results")
                        stt_status_output = gr.Textbox(
                            label="Processing Status",
                            lines=15,
                            max_lines=20,
                            interactive=False
                        )
                        stt_conversation_output = gr.Textbox(
                            label="Conversation (conversation.txt)",
                            lines=15,
                            max_lines=25,
                            interactive=False,
                            show_copy_button=True
                        )
                        stt_zip_download = gr.File(
                            label="Download All Results (ZIP)",
                            interactive=False
                        )
                
                stt_process_btn.click(
                    fn=process_batch_transcription,
                    inputs=[stt_audio_files, stt_zip_file, stt_link_or_path, stt_language_dropdown],
                    outputs=[gr.File(visible=False), gr.File(visible=False), stt_zip_download, stt_status_output, stt_conversation_output]
                )
            
            # ================================================================
            # TAB 4: LLM ANALYSIS
            # ================================================================
            with gr.Tab("4️⃣ LLM Analysis"):
                gr.Markdown("### Analyze transcriptions using Large Language Models")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### Input Settings")
                        
                        with gr.Tab("文本輸入"):
                            llm_prompt_textbox = gr.Textbox(
                                label="對話記錄",
                                placeholder="請輸入或粘貼電話對話記錄...",
                                lines=15,
                                value=load_example_file(),
                            )
                        
                        with gr.Tab("文件上傳"):
                            llm_prompt_file = gr.File(
                                label="上傳對話記錄文件 (.txt, .json)",
                                file_types=[".txt", ".json"],
                            )
                            gr.Markdown("*上傳文件將優先於文本輸入*")
                        
                        gr.Markdown("#### Context Information (Metadata)")
                        gr.Markdown("*This information will be included in the system prompt*")
                        
                        llm_metadata_textbox = gr.Textbox(
                            label="Paste Context Information",
                            placeholder="""""",
                            lines=8,
                            info="Paste all context information at once in the format shown above"
                        )
                        
                        gr.Markdown("#### LLM Settings")
                        
                        with gr.Row():
                            llm_model_dropdown = gr.Dropdown(
                                choices=MODEL_OPTIONS,
                                value=DEFAULT_MODEL,
                                label="模型",
                                allow_custom_value=True,
                            )
                            llm_temperature_slider = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature",
                            )
                        
                        llm_ollama_url = gr.Textbox(
                            label="Ollama URL",
                            value=DEFAULT_OLLAMA_URL,
                            placeholder="http://localhost:11434",
                        )
                        
                        llm_system_message = gr.Textbox(
                            label="系統訊息 (System Message)",
                            value=DEFAULT_SYSTEM_MESSAGE,
                            lines=3,
                        )
                        
                        llm_analyze_btn = gr.Button("🚀 開始分析", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Analysis Results")
                        
                        # llm_status_box = gr.Textbox(
                        #     label="狀態",
                        #     lines=6,
                        #     interactive=False,
                        # )
                        
                        llm_response_box = gr.Textbox(
                            label="LLM 回應",
                            lines=20,
                            interactive=False,
                        )
                
                llm_analyze_btn.click(
                    fn=analyze_with_llm,
                    inputs=[
                        llm_prompt_textbox,
                        llm_prompt_file,
                        llm_model_dropdown,
                        llm_ollama_url,
                        llm_system_message,
                        llm_temperature_slider,
                        llm_metadata_textbox,
                    ],
                    # outputs=[llm_status_box, llm_response_box],
                    outputs=[llm_response_box],
                )
            
            # ================================================================
            # TAB 5: SPEAKER SEPARATION
            # ================================================================
            with gr.Tab("5️⃣ Speaker Separation"):
                gr.Markdown("### Separate speakers into individual audio tracks")
                gr.Markdown("*This tool performs diarization and creates separate audio files with only one speaker in each*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Input")
                        sep_audio_input = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            sources=["upload"]
                        )
                        sep_process_btn = gr.Button("Separate Speakers", variant="primary", size="lg")
                        
                        gr.Markdown("""
                        #### How it works:
                        1. **Diarization**: Identifies who speaks when
                        2. **Separation**: Creates individual audio files per speaker
                        3. **Output**: Each speaker in a separate WAV file (others silenced)
                        """)
                        
                    with gr.Column(scale=2):
                        gr.Markdown("#### Results")
                        sep_status_output = gr.Textbox(
                            label="Processing Status",
                            lines=20,
                            max_lines=30,
                            interactive=False
                        )
                        
                        gr.Markdown("#### Download Separated Audio Files")
                        with gr.Row():
                            sep_speaker_0_output = gr.Audio(
                                label="Speaker 0 Only",
                                type="filepath",
                                interactive=False
                            )
                            sep_speaker_1_output = gr.Audio(
                                label="Speaker 1 Only",
                                type="filepath",
                                interactive=False
                            )
                
                sep_process_btn.click(
                    fn=process_speaker_separation,
                    inputs=[sep_audio_input],
                    outputs=[sep_speaker_0_output, sep_speaker_1_output, sep_status_output]
                )
            
            # ================================================================
            # TAB 6: AUDIO ENHANCEMENT
            # ================================================================
            with gr.Tab("6️⃣ Audio Enhancement"):
                gr.Markdown("### Enhance audio quality for better transcription")
                gr.Markdown("*Optimized for poor quality phone recordings, elderly speakers, and Cantonese speech*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Input")
                        enh_audio_input = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            sources=["upload"]
                        )
                        
                        gr.Markdown("#### Options")
                        enh_transcribe_checkbox = gr.Checkbox(
                            label="Transcribe Enhanced Audio",
                            value=False,
                            info="Automatically transcribe after enhancement"
                        )
                        
                        enh_language_dropdown = gr.Dropdown(
                            choices=["auto", "zh", "en", "yue", "ja", "ko"],
                            value="yue",
                            label="Transcription Language",
                            info="yue = Cantonese",
                            visible=False
                        )
                        
                        enh_process_btn = gr.Button("🎵 Enhance Audio", variant="primary", size="lg")
                        
                        gr.Markdown("""
                        #### Enhancement Pipeline:
                        1. **High-pass filter** - Remove low rumble
                        2. **Low-pass filter** - Remove high hiss
                        3. **Noise reduction** - Spectral gating
                        4. **Speech enhancement** - Boost 300-3400 Hz
                        5. **Compression** - Even out volume
                        6. **Normalization** - Optimize for STT
                        
                        #### Best for:
                        - Poor quality phone recordings
                        - Elderly speakers
                        - Background noise
                        - Volume variations
                        """)
                        
                    with gr.Column(scale=2):
                        gr.Markdown("#### Results")
                        enh_status_output = gr.Textbox(
                            label="Processing Status",
                            lines=25,
                            max_lines=35,
                            interactive=False
                        )
                        
                        gr.Markdown("#### Enhanced Audio")
                        enh_audio_output = gr.Audio(
                            label="Download Enhanced Audio",
                            type="filepath",
                            interactive=False
                        )
                        
                        enh_transcription_output = gr.Textbox(
                            label="Transcription (if enabled)",
                            lines=10,
                            max_lines=20,
                            interactive=False,
                            show_copy_button=True
                        )
                
                # Connect button
                enh_process_btn.click(
                    fn=process_audio_enhancement,
                    inputs=[enh_audio_input, enh_transcribe_checkbox, enh_language_dropdown],
                    outputs=[enh_audio_output, enh_transcription_output, enh_status_output]
                )
                
                # Show/hide language dropdown
                enh_transcribe_checkbox.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[enh_transcribe_checkbox],
                    outputs=[enh_language_dropdown]
                )
            
            # ================================================================
            # TAB 7: LLM COMPARISON
            # ================================================================
            with gr.Tab("7️⃣ LLM Comparison"):
                gr.Markdown("### Compare multiple LLM models simultaneously")
                gr.Markdown("*Run multiple models in parallel and compare their responses*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Input Settings")
                        
                        with gr.Tab("文本輸入"):
                            comp_prompt_textbox = gr.Textbox(
                                label="對話記錄",
                                placeholder="請輸入或粘貼電話對話記錄...",
                                lines=12,
                                value=load_example_file(),
                            )
                        
                        with gr.Tab("文件上傳"):
                            comp_prompt_file = gr.File(
                                label="上傳對話記錄文件 (.txt, .json)",
                                file_types=[".txt", ".json"],
                            )
                            gr.Markdown("*上傳文件將優先於文本輸入*")
                        
                        gr.Markdown("#### LLM Settings")
                        
                        comp_model_checkboxes = gr.CheckboxGroup(
                            choices=MODEL_OPTIONS,
                            value=[MODEL_OPTIONS[0], MODEL_OPTIONS[1]],
                            label="選擇模型 (可選多個)",
                            info="選擇要同時運行的模型",
                        )
                        
                        comp_temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                        )
                        
                        comp_ollama_url = gr.Textbox(
                            label="Ollama URL",
                            value=DEFAULT_OLLAMA_URL,
                            placeholder="http://localhost:11434",
                        )
                        
                        comp_system_message = gr.Textbox(
                            label="系統訊息 (System Message)",
                            value=DEFAULT_SYSTEM_MESSAGE,
                            lines=3,
                        )
                        
                        comp_compare_btn = gr.Button("🚀 開始比較", variant="primary", size="lg")
                        
                        comp_status_box = gr.Textbox(
                            label="執行狀態",
                            lines=12,
                            interactive=False,
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Comparison Results")
                        
                        # Store results in state
                        comp_results_state = gr.State({})
                        
                        # Create result boxes for up to 6 models
                        comp_result_boxes = []
                        
                        for i in range(6):
                            with gr.Row(visible=False) as row:
                                with gr.Column():
                                    model_label = gr.Markdown(f"", visible=True)
                                    time_label = gr.Markdown(f"", visible=True)
                                    result_text = gr.Textbox(
                                        label="",
                                        lines=12,
                                        interactive=False,
                                        show_label=False,
                                        show_copy_button=True,
                                    )
                            comp_result_boxes.append((row, model_label, time_label, result_text))
                
                # Helper function to update UI
                def update_comparison_ui(
                    prompt_text,
                    prompt_file,
                    selected_models,
                    ollama_url,
                    system_message,
                    temperature,
                ):
                    status, results = compare_models(
                        prompt_text,
                        prompt_file,
                        selected_models,
                        ollama_url,
                        system_message,
                        temperature,
                    )
                    
                    formatted_results = format_comparison_results(results)
                    
                    # Prepare outputs for all result boxes
                    outputs = [status, results]
                    
                    # Update each result box
                    for i in range(6):
                        if i < len(formatted_results):
                            model_name, time_info, response = formatted_results[i]
                            outputs.extend([
                                gr.Row(visible=True),      # row visibility
                                model_name,                  # model label
                                time_info,                   # time label
                                response,                    # result text
                            ])
                        else:
                            # Hide unused boxes
                            outputs.extend([
                                gr.Row(visible=False),       # row visibility
                                "",                          # model label
                                "",                          # time label
                                "",                          # result text
                            ])
                    
                    return outputs
                
                # Connect the button
                comp_outputs = [comp_status_box, comp_results_state]
                for row, model_label, time_label, result_text in comp_result_boxes:
                    comp_outputs.extend([row, model_label, time_label, result_text])
                
                comp_compare_btn.click(
                    fn=update_comparison_ui,
                    inputs=[
                        comp_prompt_textbox,
                        comp_prompt_file,
                        comp_model_checkboxes,
                        comp_ollama_url,
                        comp_system_message,
                        comp_temperature_slider,
                    ],
                    outputs=comp_outputs,
                )
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Unified Phone Call Analysis Suite...")
    print("📝 All tools available in one interface!")
    print("=" * 60)
    
    demo = create_unified_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

