"""
Tab 3: Batch Speech-to-Text
Transcribe multiple audio segments to text
"""

import os
import re
import time
import json
import csv
import shutil
import tempfile
import zipfile
import traceback
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
import gradio as gr

from funasr import AutoModel
from batch_stt import format_str_v3, load_audio
from audio_chopper import chop_audio_file, read_rttm_file

try:
    from diarization import diarize_audio
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    def diarize_audio(*args, **kwargs):
        raise RuntimeError("Diarization is not available. NeMo is required but not installed.")
from mongodb_utils import load_from_mongodb, save_to_mongodb, find_one_from_mongodb

# Import for LLM analysis
from langchain_ollama import ChatOllama
from model_config import DEFAULT_MODEL, DEFAULT_OLLAMA_URL

# Import for Whisper-v3-Cantonese model
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download
import librosa
import torch
import sys
from opencc import OpenCC

# MongoDB collection names
DIARIZATION_COLLECTION = "diarization_results"
METADATA_COLLECTION = "file_metadata"
TRANSCRIPTION_SENSEVOICE_COLLECTION = "transcriptions_sensevoice"
TRANSCRIPTION_WHISPERV3_COLLECTION = "transcriptions_whisperv3_cantonese"

# Global variables for model management
sensevoice_model = None
whisperv3_cantonese_model = None
whisperv3_cantonese_processor = None
current_sensevoice_loaded = False
current_whisperv3_cantonese_loaded = False

# Device management for GPU/CPU
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
# DEVICE = "cpu"

# Initialize OpenCC converter for Simplified to Traditional Chinese conversion
opencc_converter = OpenCC('s2t')  # Simplified to Traditional

def load_diarization_cache():
    """
    Load cached diarization results from MongoDB.
    
    Returns:
        dict: Dictionary mapping filename to cached results
    """
    cache = {}
    
    # Load all documents from MongoDB
    documents = load_from_mongodb(DIARIZATION_COLLECTION)
    
    for doc in documents:
        cache[doc['filename']] = {
            'rttm_content': doc['rttm_content'],
            'processing_time': float(doc['processing_time']),
            'num_segments': int(doc['num_segments']),
            'num_speakers': int(doc['num_speakers']),
            'speaker_ids': doc['speaker_ids'],
            'timestamp': doc['timestamp']
        }
    
    return cache


def save_diarization_to_cache(filename, rttm_content, processing_time, num_segments, num_speakers, speaker_ids):
    """
    Save diarization result to MongoDB cache.
    
    Args:
        filename: Name of the audio file
        rttm_content: RTTM content string
        processing_time: Time taken to process
        num_segments: Number of segments detected
        num_speakers: Number of speakers detected
        speaker_ids: Comma-separated speaker IDs
    """
    # Prepare document
    document = {
        'filename': filename,
        'rttm_content': rttm_content,
        'processing_time': processing_time,
        'num_segments': num_segments,
        'num_speakers': num_speakers,
        'speaker_ids': speaker_ids,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to MongoDB with upsert on filename
    save_to_mongodb(DIARIZATION_COLLECTION, document, unique_key='filename')


def load_transcription_cache_sensevoice():
    """
    Load cached SenseVoice transcription results from MongoDB.
    
    Returns:
        dict: Dictionary mapping filename to cached transcription results
    """
    cache = {}
    
    # Load all documents from MongoDB
    documents = load_from_mongodb(TRANSCRIPTION_SENSEVOICE_COLLECTION)
    
    for doc in documents:
        cache[doc['filename']] = {
            'transcription': doc['transcription'],
            'raw_transcription': doc['raw_transcription'],
            'language': doc.get('language', 'yue'),
            'processing_time': float(doc['processing_time']),
            'timestamp': doc['timestamp']
        }
    
    return cache


def save_transcription_to_cache_sensevoice(filename, transcription, raw_transcription, language, processing_time):
    """
    Save SenseVoice transcription result to MongoDB cache.
    
    Args:
        filename: Name of the audio file
        transcription: Formatted transcription text
        raw_transcription: Raw transcription text
        language: Language code used for transcription
        processing_time: Time taken to process
    """
    document = {
        'filename': filename,
        'transcription': transcription,
        'raw_transcription': raw_transcription,
        'language': language,
        'processing_time': processing_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'SenseVoiceSmall'
    }
    
    # Save to MongoDB with upsert on filename
    save_to_mongodb(TRANSCRIPTION_SENSEVOICE_COLLECTION, document, unique_key='filename')


def load_transcription_cache_whisperv3():
    """
    Load cached Whisper-v3-Cantonese transcription results from MongoDB.
    
    Returns:
        dict: Dictionary mapping filename to cached transcription results
    """
    cache = {}
    
    # Load all documents from MongoDB
    documents = load_from_mongodb(TRANSCRIPTION_WHISPERV3_COLLECTION)
    
    for doc in documents:
        cache[doc['filename']] = {
            'transcription': doc['transcription'],
            'raw_transcription': doc['raw_transcription'],
            'processing_time': float(doc['processing_time']),
            'timestamp': doc['timestamp']
        }
    
    return cache


def save_transcription_to_cache_whisperv3(filename, transcription, processing_time):
    """
    Save Whisper-v3-Cantonese transcription result to MongoDB cache.
    
    Args:
        filename: Name of the audio file
        transcription: Transcription text
        processing_time: Time taken to process
    """
    document = {
        'filename': filename,
        'transcription': transcription,
        'raw_transcription': transcription,  # Whisper doesn't have separate raw format
        'processing_time': processing_time,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'Whisper-v3-Cantonese'
    }
    
    # Save to MongoDB with upsert on filename
    save_to_mongodb(TRANSCRIPTION_WHISPERV3_COLLECTION, document, unique_key='filename')


def parse_filename_metadata(filename: str, csv_path: str = "client.csv") -> dict:
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
        dict: Dictionary with 'status' (success/error), 'formatted_output' (display string), 
              and 'data' (structured metadata dict)
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
            error_msg = f"""❌ Error: Filename does not match expected pattern.

Expected format:
[Broker Name ID]_8330-97501167_20251010014510(20981).wav

Received:
{filename}

Note: The filename may have been sanitized by the system. 
Please ensure special characters are preserved or manually enter the correct format.
"""
            return {
                'status': 'error',
                'formatted_output': error_msg,
                'data': None
            }
        
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
            error_msg = f"❌ Error parsing datetime: {str(e)}"
            return {
                'status': 'error',
                'formatted_output': error_msg,
                'data': None
            }
        
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
        
        # Format output for display
        formatted_output = f"""✅ Metadata extracted successfully

Broker Name: {broker_name}
Broker Id: {broker_id}
Client Number: {client_number}
Client Name: {client_name}
Client Id: {client_id}
UTC: {utc_formatted}
HKT: {hkt_formatted}

💾 Saved to MongoDB
"""
        
        # Prepare structured data
        metadata_dict = {
            'filename': filename,
            'broker_name': broker_name,
            'broker_id': broker_id,
            'client_number': client_number,
            'client_name': client_name,
            'client_id': client_id,
            'utc_datetime': utc_formatted,
            'hkt_datetime': hkt_formatted,
            'utc_datetime_obj': utc_dt,
            'hkt_datetime_obj': hkt_dt,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return {
            'status': 'success',
            'formatted_output': formatted_output,
            'data': metadata_dict
        }
        
    except Exception as e:
        error_msg = f"❌ Error parsing filename: {str(e)}\n\n{traceback.format_exc()}"
        return {
            'status': 'error',
            'formatted_output': error_msg,
            'data': None
        }


def process_file_metadata(audio_file):
    """
    Process uploaded audio file and extract metadata from filename.
    Also saves the metadata to MongoDB.
    
    Args:
        audio_file: Audio file from Gradio interface
        
    Returns:
        str: JSON formatted metadata string
    """
    if audio_file is None:
        return json.dumps({"error": "No file uploaded. Please drag and drop an audio file."}, indent=2)
    
    try:
        # Get filename
        filename = os.path.basename(audio_file)
        
        # Parse metadata
        result = parse_filename_metadata(filename)
        
        # If parsing was successful, save to MongoDB
        if result['status'] == 'success' and result['data']:
            save_to_mongodb(METADATA_COLLECTION, result['data'], unique_key='filename')
            # Return JSON format with status and data
            output_dict = {
                "metadata": {
                    "filename": result['data']['filename'],
                    "broker_name": result['data']['broker_name'],
                    "broker_id": result['data']['broker_id'],
                    "client_number": result['data']['client_number'],
                    "client_name": result['data']['client_name'],
                    "client_id": result['data']['client_id'],
                    "utc_datetime": result['data']['utc_datetime'],
                    "hkt_datetime": result['data']['hkt_datetime'],
                    "timestamp": result['data']['timestamp']
                }
            }
            return json.dumps(output_dict, indent=2, ensure_ascii=False)
        else:
            # Error case - return error in JSON format
            return json.dumps({
                "status": "error",
                "message": result['formatted_output']
            }, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return json.dumps({
            "status": "error",
            "message": error_msg
        }, indent=2, ensure_ascii=False)


def initialize_sensevoice_model():
    """Initialize the SenseVoice model."""
    global sensevoice_model, current_sensevoice_loaded
    
    # Only reload if not already loaded
    if current_sensevoice_loaded and sensevoice_model is not None:
        return f"✅ SenseVoiceSmall already loaded"
    
    status = f"🔄 Loading SenseVoiceSmall model...\n"
    status += f"  ⚙️ Device: {DEVICE_INFO}\n"
    
    try:
        sensevoice_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=False,
            disable_update=True,
        )
        current_sensevoice_loaded = True
        status += f"✅ SenseVoiceSmall loaded successfully! (FunASR handles device internally)"
        return status
    except Exception as e:
        status += f"❌ Failed to load SenseVoiceSmall: {str(e)}"
        return status


def initialize_whisperv3_cantonese_model():
    """Initialize the Whisper-v3-Cantonese model."""
    global whisperv3_cantonese_model, whisperv3_cantonese_processor, current_whisperv3_cantonese_loaded
    
    # Only reload if not already loaded
    if current_whisperv3_cantonese_loaded and whisperv3_cantonese_model is not None:
        return f"✅ Whisper-v3-Cantonese already loaded on {DEVICE_NAME}"
    
    status = f"🔄 Loading Whisper-v3-Cantonese model...\n"
    status += f"  ⚙️ Device: {DEVICE_INFO}\n"
    
    try:
        # Load the model and processor from Hugging Face
        whisperv3_cantonese_model = WhisperForConditionalGeneration.from_pretrained(
            "khleeloo/whisper-large-v3-cantonese"
        )
        whisperv3_cantonese_processor = WhisperProcessor.from_pretrained(
            "khleeloo/whisper-large-v3-cantonese"
        )
        status += f"  ✓ Model and processor loaded\n"
        
        # Move model to device (GPU if available)
        whisperv3_cantonese_model = whisperv3_cantonese_model.to(DEVICE)
        status += f"  ✓ Model moved to {DEVICE_NAME}\n"
        
        # Set model to evaluation mode
        whisperv3_cantonese_model.eval()
        current_whisperv3_cantonese_loaded = True
        
        status += f"✅ Whisper-v3-Cantonese loaded successfully on {DEVICE_NAME}!"
        return status
    except Exception as e:
        status += f"❌ Failed to load Whisper-v3-Cantonese: {str(e)}"
        return status


def transcribe_single_audio_sensevoice(audio_path, language="yue", use_cache=True):
    """
    Transcribe a single audio file using SenseVoiceSmall model.
    
    Args:
        audio_path: Path to audio file
        language: Language code for transcription
        use_cache: Whether to use cached results (default: True)
        
    Returns:
        dict: Transcription result with file, path, transcription, raw_transcription, and cache_hit
    """
    global sensevoice_model
    
    if sensevoice_model is None:
        return None
    
    filename = os.path.basename(audio_path)
    
    # Check cache first
    if use_cache:
        cache = load_transcription_cache_sensevoice()
        if filename in cache:
            cached = cache[filename]
            # Only use cache if language matches
            if cached.get('language', 'yue') == language:
                print(f"💾 Cache hit for SenseVoice: {filename}")
                return {
                    "file": filename,
                    "path": audio_path,
                    "transcription": cached['transcription'],
                    "raw_transcription": cached['raw_transcription'],
                    "cache_hit": True,
                    "cached_time": cached.get('processing_time', 0),
                    "cached_timestamp": cached.get('timestamp', '')
                }
    
    # Load audio
    audio_array, sample_rate = load_audio(audio_path)
    if audio_array is None:
        return None
    
    # Run inference
    try:
        start_time = time.time()
        result = sensevoice_model.generate(
            input=audio_array,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True
        )
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Extract and format text
        raw_text = result[0]["text"]
        formatted_text = format_str_v3(raw_text)
        
        # Convert to Traditional Chinese using OpenCC
        formatted_text = opencc_converter.convert(formatted_text)
        
        # Save to cache
        save_transcription_to_cache_sensevoice(
            filename=filename,
            transcription=formatted_text,
            raw_transcription=raw_text,
            language=language,
            processing_time=processing_time
        )
        
        return {
            "file": filename,
            "path": audio_path,
            "transcription": formatted_text,
            "raw_transcription": raw_text,
            "cache_hit": False,
            "processing_time": processing_time
        }
    except Exception as e:
        print(f"Error transcribing with SenseVoice {audio_path}: {e}")
        return None


def transcribe_single_audio_whisperv3_cantonese(audio_path, use_cache=True):
    """
    Transcribe a single audio file using Whisper-v3-Cantonese model.
    
    Args:
        audio_path: Path to audio file
        use_cache: Whether to use cached results (default: True)
        
    Returns:
        dict: Transcription result with file, path, transcription, raw_transcription, and cache_hit
    """
    global whisperv3_cantonese_model, whisperv3_cantonese_processor
    
    if whisperv3_cantonese_model is None or whisperv3_cantonese_processor is None:
        return None
    
    filename = os.path.basename(audio_path)
    
    # Check cache first
    if use_cache:
        cache = load_transcription_cache_whisperv3()
        if filename in cache:
            cached = cache[filename]
            print(f"💾 Cache hit for Whisper-v3-Cantonese: {filename}")
            return {
                "file": filename,
                "path": audio_path,
                "transcription": cached['transcription'],
                "raw_transcription": cached['raw_transcription'],
                "cache_hit": True,
                "cached_time": cached.get('processing_time', 0),
                "cached_timestamp": cached.get('timestamp', '')
            }
    
    # Load audio
    try:
        start_time = time.time()
        audio, sampling_rate = librosa.load(audio_path, sr=16000)
        
        # Process the audio
        input_features = whisperv3_cantonese_processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
        
        # Move input to device
        input_features = input_features.to(DEVICE)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = whisperv3_cantonese_model.generate(input_features, language="yue")
        
        # Decode the transcription
        transcription = whisperv3_cantonese_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcription_text = transcription[0]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Save to cache
        save_transcription_to_cache_whisperv3(
            filename=filename,
            transcription=transcription_text,
            processing_time=processing_time
        )
        
        return {
            "file": filename,
            "path": audio_path,
            "transcription": transcription_text,
            "raw_transcription": transcription_text,
            "cache_hit": False,
            "processing_time": processing_time
        }
    except Exception as e:
        print(f"Error transcribing with Whisper-v3-Cantonese {audio_path}: {e}")
        return None


def parse_rttm_timestamps(rttm_content: str) -> dict:
    """
    Parse RTTM content to extract start times for each speaker segment.
    
    Args:
        rttm_content: RTTM file content
    
    Returns:
        dict: Dictionary mapping segment index to (speaker_id, start_time)
              Format: {0: ('speaker_0', 0.380), 1: ('speaker_1', 1.255), ...}
    """
    timestamps = {}
    lines = rttm_content.strip().split('\n')
    
    for idx, line in enumerate(lines):
        if line.strip():
            parts = line.split()
            if len(parts) >= 10:
                # RTTM format: SPEAKER filename channel start_time duration <NA> <NA> speaker_id <NA> <NA>
                # Since filename can contain spaces, parse from the end (last 5 fields are always: <NA> <NA> speaker_id <NA> <NA>)
                # speaker_id is at parts[-3]
                # start_time and duration are at parts[-7] and parts[-6]
                try:
                    speaker_id = parts[-3]
                    start_time = float(parts[-7])
                    timestamps[idx] = (speaker_id, start_time)
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Warning: Could not parse RTTM line {idx}: {e}")
                    continue
    
    return timestamps


def add_timestamps_to_conversation(conversation: str, rttm_timestamps: dict, broker_name: str, client_name: str, broker_speaker_id: str) -> str:
    """
    Add timestamps from RTTM to each line of the conversation.
    
    Args:
        conversation: The labeled conversation text
        rttm_timestamps: Dictionary of timestamps from RTTM
        broker_name: Name of the broker
        client_name: Name of the client
        broker_speaker_id: Which speaker ID is the broker ("speaker_0" or "speaker_1")
    
    Returns:
        str: Conversation with timestamps added
    """
    if not conversation or not rttm_timestamps:
        return conversation
    
    lines = conversation.strip().split('\n')
    result_lines = []
    segment_idx = 0
    
    for line in lines:
        if line.strip():
            # Check if this line is a conversation line (starts with 經紀 or 客戶)
            if line.startswith(f"經紀 {broker_name}:") or line.startswith(f"客戶 {client_name}:"):
                # Get timestamp for this segment
                if segment_idx in rttm_timestamps:
                    speaker_id, start_time = rttm_timestamps[segment_idx]
                    
                    # Format: - 經紀 Dickson Lau (0.6): 请到时点啊。
                    if line.startswith(f"經紀 {broker_name}:"):
                        # Extract the text after the colon
                        text = line.split(':', 1)[1].strip() if ':' in line else ''
                        result_lines.append(f"經紀 {broker_name} (start_time: {start_time}): {text}")
                    else:
                        # Extract the text after the colon
                        text = line.split(':', 1)[1].strip() if ':' in line else ''
                        result_lines.append(f"客戶 {client_name} (start_time: {start_time}): {text}")
                    
                    segment_idx += 1
                else:
                    # No timestamp available, keep original format
                    result_lines.append(line)
            else:
                # Not a conversation line, keep as is
                result_lines.append(line)
        else:
            result_lines.append(line)
    
    return '\n'.join(result_lines)


def identify_speakers_with_llm(conversation_text: str, broker_name: str, client_name: str, 
                                model: str = DEFAULT_MODEL, ollama_url: str = DEFAULT_OLLAMA_URL) -> tuple[str, str, str]:
    """
    Use LLM to identify which speaker is the broker and which is the client,
    then replace speaker labels with proper names.
    
    Args:
        conversation_text: The conversation with speaker_0 and speaker_1 labels
        broker_name: Name of the broker from metadata
        client_name: Name of the client from metadata
        model: LLM model to use
        ollama_url: Ollama server URL
        
    Returns:
        tuple: (labeled_conversation, identification_log, broker_speaker_id)
               - labeled_conversation: Conversation with '經紀 {name}:' and '客戶 {name}:'
               - identification_log: LLM's reasoning for identification
               - broker_speaker_id: "speaker_0" or "speaker_1" indicating which one is the broker
    """
    try:
        # Prepare the prompt for LLM
        system_message = """你是一位專業的電話對話分析專員。你的任務是分析電話錄音的轉錄文字，識別對話中哪位說話者是經紀（券商），哪位是客戶。

請根據以下特徵判斷：
1. 經紀通常會：
   - 提供市場資訊和建議
   - 確認交易指示
   - 使用專業術語
   - 主動詢問客戶需求
   - 重複客戶的下單資料以確認

2. 客戶通常會：
   - 詢問股票資訊
   - 下達買賣指示
   - 回應經紀的確認

請以繁體中文回應，並且只需回答 "speaker_0" 或 "speaker_1" 來指出誰是經紀，然後在下一行簡短說明你的判斷理由（不超過3句話）。

回應格式：
經紀是: speaker_X
理由: [你的判斷理由]"""

        user_prompt = f"""以下是一段電話對話的轉錄文字：

{conversation_text}

已知資訊：
- 經紀姓名: {broker_name}
- 客戶姓名: {client_name}

請判斷 speaker_0 和 speaker_1 中，哪一位是經紀？"""

        # Initialize LLM
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=0.3,  # Lower temperature for more deterministic results
        )
        
        messages = [
            ("system", system_message),
            ("human", user_prompt),
        ]
        
        # Get response
        resp = chat_llm.invoke(messages)
        response_content = getattr(resp, "content", str(resp))
        
        # Parse the response to extract which speaker is the broker
        broker_speaker_id = None
        if "speaker_0" in response_content.lower():
            broker_speaker_id = "speaker_0"
        elif "speaker_1" in response_content.lower():
            broker_speaker_id = "speaker_1"
        
        if broker_speaker_id is None:
            # Default to speaker_0 if we can't determine
            broker_speaker_id = "speaker_0"
            response_content += "\n\n⚠️ 無法明確判斷，預設 speaker_0 為經紀"
        
        # Now replace the labels in the conversation
        client_speaker_id = "speaker_1" if broker_speaker_id == "speaker_0" else "speaker_0"
        
        # Replace speaker labels with proper names
        labeled_conversation = conversation_text
        labeled_conversation = labeled_conversation.replace(f"{broker_speaker_id}:", f"經紀 {broker_name}:")
        labeled_conversation = labeled_conversation.replace(f"{client_speaker_id}:", f"客戶 {client_name}:")
        
        return labeled_conversation, response_content, broker_speaker_id
        
    except Exception as e:
        error_msg = f"❌ LLM識別失敗: {str(e)}\n{traceback.format_exc()}"
        # Return original conversation if LLM fails
        return conversation_text, error_msg, "speaker_0"


def process_chop_and_transcribe(audio_file, language, use_sensevoice, use_whisperv3_cantonese, overwrite_diarization=False, padding_ms=100, use_enhanced_format=False, progress=gr.Progress()):
    """
    Integrated pipeline: Extract metadata, auto-diarize (with cache), chop audio based on RTTM, 
    transcribe the segments, and use LLM to identify speakers.
    
    Args:
        audio_file: Audio file from Gradio interface
        language: Language code for transcription
        use_sensevoice: Whether to use SenseVoiceSmall model
        use_whisperv3_cantonese: Whether to use Whisper-v3-Cantonese model
        overwrite_diarization: If True, reprocess diarization even if cached
        padding_ms: Padding in milliseconds for chopping (default: 100)
        use_enhanced_format: If True, add metadata header and timestamps to results
    
    Returns:
        tuple: (metadata_json, json_file, sensevoice_txt, whisperv3_txt, zip_file, 
                sensevoice_labeled, whisperv3_labeled, llm_log, status_message)
    """
    if audio_file is None:
        return "", None, None, None, None, "", "", "", "❌ No audio file uploaded"
    
    if not use_sensevoice and not use_whisperv3_cantonese:
        return "", None, None, None, None, "", "", "", "⚠️ Please select at least one model"
    
    try:
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="chop_transcribe_")
        chopped_audio_dir = os.path.join(temp_out_dir, "chopped_segments")
        os.makedirs(chopped_audio_dir, exist_ok=True)
        
        filename = os.path.basename(audio_file)
        
        status = f"🔄 Starting integrated pipeline (Auto-Diarize + Chop + Transcribe)...\n\n"
        status += f"📁 Audio file: {filename}\n"
        status += f"⏱️ Padding: {padding_ms} ms\n\n"
        
        # Step 0: Extract metadata from filename
        progress(0, desc="Extracting metadata...")
        status += "📋 Extracting metadata from filename...\n"
        metadata_result = parse_filename_metadata(filename)
        broker_name = "Unknown"
        client_name = "Unknown"
        metadata_json = ""
        
        if metadata_result['status'] == 'success' and metadata_result['data']:
            # Save to MongoDB
            save_to_mongodb(METADATA_COLLECTION, metadata_result['data'], unique_key='filename')
            broker_name = metadata_result['data'].get('broker_name', 'Unknown')
            client_name = metadata_result['data'].get('client_name', 'Unknown')
            status += f"✅ Metadata extracted successfully\n"
            status += f"  • 經紀: {broker_name}\n"
            status += f"  • 客戶: {client_name}\n"
            status += f"  • 💾 Saved to MongoDB\n\n"
            
            # Prepare JSON metadata for return
            metadata_json = json.dumps({
                "metadata": {
                    "filename": metadata_result['data']['filename'],
                    "broker_name": broker_name,
                    "broker_id": metadata_result['data']['broker_id'],
                    "client_number": metadata_result['data']['client_number'],
                    "client_name": client_name,
                    "client_id": metadata_result['data']['client_id'],
                    "utc_datetime": metadata_result['data']['utc_datetime'],
                    "hkt_datetime": metadata_result['data']['hkt_datetime'],
                    "timestamp": metadata_result['data']['timestamp']
                }
            }, indent=2, ensure_ascii=False)
        else:
            status += f"⚠️ Could not extract metadata from filename\n\n"
        
        # Step 1: Check MongoDB for cached RTTM or run diarization
        progress(0, desc="Checking for cached diarization...")
        cache = load_diarization_cache()
        
        rttm_content = None
        
        if filename in cache and not overwrite_diarization:
            # Load from cache
            cached = cache[filename]
            rttm_content = cached['rttm_content']
            
            status += f"💾 Loaded cached RTTM from MongoDB for: {filename}\n"
            status += f"📅 Previously processed: {cached['timestamp']}\n"
            status += f"  • Segments: {cached['num_segments']}\n"
            status += f"  • Speakers: {cached['num_speakers']} ({cached['speaker_ids']})\n"
            status += f"  • Original processing time: {cached['processing_time']:.2f}s\n"
            status += f"  • ⚡ Cache hit - instant retrieval!\n\n"
        else:
            # Run diarization
            if overwrite_diarization and filename in cache:
                status += f"♻️ Overwrite mode: Reprocessing diarization despite existing cache\n\n"
            else:
                status += f"🔍 No cached RTTM found. Running diarization...\n\n"
            
            progress(0.05, desc="Running diarization...")
            diarization_temp_dir = tempfile.mkdtemp(prefix="diarization_")
            
            diarization_start = time.time()
            rttm_content = diarize_audio(
                audio_filepath=audio_file,
                out_dir=diarization_temp_dir,
                num_speakers=2,
                domain_type="telephonic"
            )
            diarization_end = time.time()
            diarization_time = diarization_end - diarization_start
            
            # Parse RTTM to get statistics
            lines = rttm_content.strip().split('\n')
            num_segments = len(lines)
            speakers = set()
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 8:
                        speakers.add(parts[7])
            
            speaker_ids_str = ', '.join(sorted(speakers))
            
            # Save to MongoDB cache
            save_diarization_to_cache(
                filename=filename,
                rttm_content=rttm_content,
                processing_time=diarization_time,
                num_segments=num_segments,
                num_speakers=len(speakers),
                speaker_ids=speaker_ids_str
            )
            
            status += f"✅ Diarization completed!\n"
            status += f"  • Processing time: {diarization_time:.2f}s ({diarization_time/60:.2f} min)\n"
            status += f"  • Segments: {num_segments}\n"
            status += f"  • Speakers: {len(speakers)} ({speaker_ids_str})\n"
            status += f"  • 💾 Saved to MongoDB for future use\n\n"
            
            # Clean up diarization temp directory
            try:
                shutil.rmtree(diarization_temp_dir)
            except:
                pass
        
        # Step 2: Save RTTM text to a temporary file
        progress(0.1, desc="Preparing RTTM...")
        temp_rttm_file = os.path.join(temp_out_dir, "temp_rttm.rttm")
        with open(temp_rttm_file, 'w', encoding='utf-8') as f:
            f.write(rttm_content.strip())
        status += "✅ RTTM data prepared\n"
        
        # Step 3: Read RTTM segments
        progress(0.15, desc="Reading RTTM...")
        status += "📖 Reading RTTM segments...\n"
        segments = read_rttm_file(temp_rttm_file)
        
        if not segments:
            return "", None, None, None, None, "", "", "", "❌ No segments found in RTTM data!"
        
        status += f"✅ Found {len(segments)} segments\n\n"
        
        # Count speakers
        speakers = set(seg['speaker'] for seg in segments)
        status += f"👥 Detected speakers: {len(speakers)} ({', '.join(sorted(speakers))})\n\n"
        
        # Step 4: Chop audio file
        progress(0.2, desc="Chopping audio...")
        status += "✂️ Chopping audio into segments...\n"
        chop_audio_file(audio_file, segments, chopped_audio_dir, padding_ms)
        
        # Get list of chopped files
        chopped_files = sorted([
            os.path.join(chopped_audio_dir, f) 
            for f in os.listdir(chopped_audio_dir) 
            if f.endswith('.wav')
        ])
        
        status += f"✅ Audio chopped into {len(chopped_files)} segments\n\n"
        
        # Step 5: Initialize models
        progress(0.25, desc="Loading models...")
        if use_sensevoice:
            sensevoice_status = initialize_sensevoice_model()
            status += sensevoice_status + "\n"
        if use_whisperv3_cantonese:
            whisperv3_status = initialize_whisperv3_cantonese_model()
            status += whisperv3_status + "\n"
        status += "\n"
        
        if use_sensevoice and sensevoice_model is None:
            return "", None, None, None, None, "", "", "", status + "❌ Failed to load SenseVoiceSmall model"
        if use_whisperv3_cantonese and whisperv3_cantonese_model is None:
            return "", None, None, None, None, "", "", "", status + "❌ Failed to load Whisper-v3-Cantonese model"
        
        # Step 6: Transcribe chopped segments
        status += f"📝 Transcribing {len(chopped_files)} segment(s)...\n\n"
        start_time = time.time()
        
        sensevoice_results = []
        whisperv3_results = []
        total_files = len(chopped_files)
        
        # Process all files with SenseVoice first
        sensevoice_cache_hits = 0
        if use_sensevoice:
            status += f"🎙️ Processing with SenseVoiceSmall...\n\n"
            for i, audio_path in enumerate(chopped_files):
                progress((0.25 + 0.3 * (i / total_files)), desc=f"SenseVoice {i+1}/{total_files}...")
                
                filename = os.path.basename(audio_path)
                status += f"[{i+1}/{total_files}] {filename}\n"
                
                result = transcribe_single_audio_sensevoice(audio_path, language)
                if result:
                    sensevoice_results.append(result)
                    if result.get('cache_hit', False):
                        sensevoice_cache_hits += 1
                        status += f"  💾 SenseVoice (cached): {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                    else:
                        status += f"  ✅ SenseVoice: {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                else:
                    status += f"  ❌ SenseVoice: Failed\n"
                
                status += "\n"
            
            status += f"✅ SenseVoice completed: {len(sensevoice_results)}/{total_files} files ({sensevoice_cache_hits} from cache)\n\n"
        
        # Then process all files with Whisper-v3-Cantonese
        whisperv3_cache_hits = 0
        if use_whisperv3_cantonese:
            status += f"🎙️ Processing with Whisper-v3-Cantonese...\n\n"
            for i, audio_path in enumerate(chopped_files):
                progress((0.55 + 0.20 * (i / total_files)), desc=f"Whisper-v3 {i+1}/{total_files}...")
                
                filename = os.path.basename(audio_path)
                status += f"[{i+1}/{total_files}] {filename}\n"
                
                result = transcribe_single_audio_whisperv3_cantonese(audio_path)
                if result:
                    whisperv3_results.append(result)
                    if result.get('cache_hit', False):
                        whisperv3_cache_hits += 1
                        status += f"  💾 Whisper-v3 (cached): {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                    else:
                        status += f"  ✅ Whisper-v3: {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                else:
                    status += f"  ❌ Whisper-v3: Failed\n"
                
                status += "\n"
            
            status += f"✅ Whisper-v3-Cantonese completed: {len(whisperv3_results)}/{total_files} files ({whisperv3_cache_hits} from cache)\n\n"
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        progress(0.9, desc="Saving results...")
        
        # Helper function to format speaker name
        def get_speaker_name(fname):
            m = re.search(r"speaker_(\d+)", fname)
            if m:
                return f"speaker_{m.group(1)}"
            return Path(fname).stem if fname else 'unknown'
        
        # Save results to JSON files
        results_data = {
            "sensevoice": sensevoice_results if use_sensevoice else [],
            "whisperv3_cantonese": whisperv3_results if use_whisperv3_cantonese else []
        }
        json_path = os.path.join(temp_out_dir, "transcriptions.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        # Save SenseVoice conversation.txt
        sensevoice_txt_path = None
        sensevoice_conversation_content = ""
        if use_sensevoice and sensevoice_results:
            sensevoice_txt_path = os.path.join(temp_out_dir, "conversation_sensevoice.txt")
            with open(sensevoice_txt_path, 'w', encoding='utf-8') as f:
                for r in sensevoice_results:
                    fname = r.get('file', '')
                    speaker = get_speaker_name(fname)
                    f.write(f"{speaker}: {r.get('transcription', '')}\n")
            
            with open(sensevoice_txt_path, 'r', encoding='utf-8') as f:
                sensevoice_conversation_content = f.read()
        
        # Save Whisper-v3-Cantonese conversation.txt
        whisperv3_txt_path = None
        whisperv3_conversation_content = ""
        if use_whisperv3_cantonese and whisperv3_results:
            whisperv3_txt_path = os.path.join(temp_out_dir, "conversation_whisperv3_cantonese.txt")
            with open(whisperv3_txt_path, 'w', encoding='utf-8') as f:
                for r in whisperv3_results:
                    fname = r.get('file', '')
                    speaker = get_speaker_name(fname)
                    f.write(f"{speaker}: {r.get('transcription', '')}\n")
            
            with open(whisperv3_txt_path, 'r', encoding='utf-8') as f:
                whisperv3_conversation_content = f.read()
        
        # Step 6.5: Use LLM to identify speakers and create labeled conversations
        progress(0.92, desc="Identifying speakers with LLM...")
        status += "🤖 Using LLM to identify speakers (broker vs client)...\n"
        
        sensevoice_labeled_conversation = ""
        whisperv3_labeled_conversation = ""
        llm_identification_log = ""
        
        # Parse RTTM timestamps for enhanced format
        rttm_timestamps = {}
        if use_enhanced_format and rttm_content:
            rttm_timestamps = parse_rttm_timestamps(rttm_content)
        
        # Only run LLM identification if we have both broker and client names
        if broker_name != "Unknown" and client_name != "Unknown":
            try:
                # Use SenseVoice conversation if available, otherwise Whisper
                conversation_for_llm = sensevoice_conversation_content if sensevoice_conversation_content else whisperv3_conversation_content
                
                if conversation_for_llm:
                    labeled_conv, identification_log, broker_speaker_id = identify_speakers_with_llm(
                        conversation_for_llm, broker_name, client_name
                    )
                    
                    llm_identification_log = identification_log
                    
                    # Apply the same speaker identification to both models' results
                    if sensevoice_conversation_content:
                        sensevoice_labeled_conversation, _, _ = identify_speakers_with_llm(
                            sensevoice_conversation_content, broker_name, client_name
                        )
                    
                    if whisperv3_conversation_content:
                        whisperv3_labeled_conversation, _, _ = identify_speakers_with_llm(
                            whisperv3_conversation_content, broker_name, client_name
                        )
                    
                    # Apply enhanced format if enabled
                    if use_enhanced_format and metadata_result['status'] == 'success' and metadata_result['data']:
                        metadata_header = f"""對話時間: {metadata_result['data']['hkt_datetime']}
經紀: {broker_name}
broker_id: {metadata_result['data']['broker_id']}
客戶: {client_name}
client_id: {metadata_result['data']['client_id']}

"""
                        # Add timestamps to each line
                        sensevoice_labeled_conversation = add_timestamps_to_conversation(
                            sensevoice_labeled_conversation, rttm_timestamps, broker_name, client_name, broker_speaker_id
                        )
                        sensevoice_labeled_conversation = metadata_header + sensevoice_labeled_conversation
                        
                        whisperv3_labeled_conversation = add_timestamps_to_conversation(
                            whisperv3_labeled_conversation, rttm_timestamps, broker_name, client_name, broker_speaker_id
                        )
                        whisperv3_labeled_conversation = metadata_header + whisperv3_labeled_conversation
                    
                    # Convert SenseVoice results to Traditional Chinese
                    if sensevoice_labeled_conversation:
                        sensevoice_labeled_conversation = opencc_converter.convert(sensevoice_labeled_conversation)
                    
                    status += f"✅ Speaker identification completed\n"
                    status += f"  • 經紀識別為: {broker_speaker_id}\n\n"
                else:
                    status += "⚠️ No conversation available for speaker identification\n\n"
                    sensevoice_labeled_conversation = sensevoice_conversation_content
                    whisperv3_labeled_conversation = whisperv3_conversation_content
                    llm_identification_log = "⚠️ No conversation text available for identification"
                    # Convert SenseVoice results to Traditional Chinese
                    if sensevoice_labeled_conversation:
                        sensevoice_labeled_conversation = opencc_converter.convert(sensevoice_labeled_conversation)
            except Exception as e:
                status += f"⚠️ Speaker identification failed: {str(e)}\n\n"
                sensevoice_labeled_conversation = sensevoice_conversation_content
                whisperv3_labeled_conversation = whisperv3_conversation_content
                llm_identification_log = f"❌ Error: {str(e)}"
                # Convert SenseVoice results to Traditional Chinese
                if sensevoice_labeled_conversation:
                    sensevoice_labeled_conversation = opencc_converter.convert(sensevoice_labeled_conversation)
        else:
            status += "⚠️ Skipping speaker identification (metadata not available)\n\n"
            sensevoice_labeled_conversation = sensevoice_conversation_content
            whisperv3_labeled_conversation = whisperv3_conversation_content
            llm_identification_log = "⚠️ Metadata not available - cannot identify speakers"
            # Convert SenseVoice results to Traditional Chinese
            if sensevoice_labeled_conversation:
                sensevoice_labeled_conversation = opencc_converter.convert(sensevoice_labeled_conversation)
        
        # Create a zip file with all results
        zip_path = os.path.join(temp_out_dir, "transcription_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(json_path, arcname="transcriptions.json")
            if sensevoice_txt_path:
                zipf.write(sensevoice_txt_path, arcname="conversation_sensevoice.txt")
            if whisperv3_txt_path:
                zipf.write(whisperv3_txt_path, arcname="conversation_whisperv3_cantonese.txt")
        
        # Step 7: Clean up temporary chopped files
        progress(0.95, desc="Cleaning up...")
        try:
            shutil.rmtree(chopped_audio_dir)
            status += "🧹 Temporary chopped files cleaned up\n\n"
        except Exception as e:
            status += f"⚠️ Warning: Could not clean up temp files: {str(e)}\n\n"
        
        progress(1.0, desc="Complete!")
        
        status += f"\n{'='*60}\n"
        status += f"✅ Pipeline completed successfully!\n"
        status += f"⏱️ Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)\n"
        if use_sensevoice:
            status += f"📊 SenseVoice processed: {len(sensevoice_results)}/{total_files} segments\n"
        if use_whisperv3_cantonese:
            status += f"📊 Whisper-v3-Cantonese processed: {len(whisperv3_results)}/{total_files} segments\n"
        status += f"📁 Results saved to:\n"
        status += f"   • transcriptions.json\n"
        if sensevoice_txt_path:
            status += f"   • conversation_sensevoice.txt\n"
        if whisperv3_txt_path:
            status += f"   • conversation_whisperv3_cantonese.txt\n"
        status += f"   • transcription_results.zip\n"
        status += f"{'='*60}\n"
        
        return (metadata_json, json_path, sensevoice_txt_path, whisperv3_txt_path, zip_path, 
                sensevoice_labeled_conversation, whisperv3_labeled_conversation, 
                llm_identification_log, status)
        
    except Exception as e:
        error_msg = f"❌ Error during pipeline: {str(e)}"
        error_msg += f"\n\n{traceback.format_exc()}"
        return "", None, None, None, None, "", "", "", error_msg


def process_batch_transcription(audio_files, zip_file, link_or_path, language, use_sensevoice, use_whisperv3_cantonese, progress=gr.Progress()):
    """
    Process multiple audio files for transcription.
    
    Args:
        audio_files: List of audio files from Gradio interface
        zip_file: Zip file containing audio files
        link_or_path: URL to a zip file or local folder path
        language: Language code for transcription
        use_sensevoice: Whether to use SenseVoiceSmall model
        use_whisperv3_cantonese: Whether to use Whisper-v3-Cantonese model
    
    Returns:
        tuple: (json_file, sensevoice_txt_file, whisperv3_txt_file, zip_file, sensevoice_conversation, whisperv3_conversation)
    """
    if (not audio_files or len(audio_files) == 0) and not zip_file and not link_or_path:
        return None, None, None, None, "", ""
    
    if not use_sensevoice and not use_whisperv3_cantonese:
        return None, None, None, None, "⚠️ Please select at least one model", ""
    
    try:
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="batch_stt_")
        audio_dir = os.path.join(temp_out_dir, "audio_files")
        os.makedirs(audio_dir, exist_ok=True)
        
        status = f"🔄 Starting batch transcription...\n"
        
        # Initialize models based on checkboxes
        progress(0, desc="Loading models...")
        if use_sensevoice:
            sensevoice_status = initialize_sensevoice_model()
            status += sensevoice_status + "\n"
        if use_whisperv3_cantonese:
            whisperv3_status = initialize_whisperv3_cantonese_model()
            status += whisperv3_status + "\n"
        status += "\n"
        
        if use_sensevoice and sensevoice_model is None:
            return None, None, None, None, "❌ Failed to load SenseVoiceSmall model", ""
        if use_whisperv3_cantonese and whisperv3_cantonese_model is None:
            return None, None, None, None, "", "❌ Failed to load Whisper-v3-Cantonese model"
        
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
                return None, None, None, None, f"❌ Error extracting zip: {str(e)}", ""
        
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
                    return None, None, None, None, None, f"❌ Error downloading from URL: {str(e)}", "", ""
            
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
                    return None, None, None, None, None, f"❌ Error reading folder: {str(e)}", "", ""
            
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
                    return None, None, None, None, f"❌ Error processing local zip: {str(e)}", ""
            else:
                return None, None, None, None, "❌ Invalid path or URL provided", ""
        
        if len(audio_paths) == 0:
            return None, None, None, None, "❌ No audio files found", ""
        
        status += f"📊 Total files: {len(audio_paths)}\n"
        status += f"🌐 Language: {language}\n"
        models_used = []
        if use_sensevoice:
            models_used.append("SenseVoiceSmall")
        if use_whisperv3_cantonese:
            models_used.append("Whisper-v3-Cantonese")
        status += f"🤖 Models: {', '.join(models_used)}\n\n"
        
        # Sort files by name (for segment_001.wav, segment_002.wav ordering)
        audio_paths.sort(key=lambda p: os.path.basename(p))
        
        start_time = time.time()
        
        # Process each file with selected models
        sensevoice_results = []
        whisperv3_results = []
        total_files = len(audio_paths)
        
        status += f"📝 Processing {total_files} audio file(s)...\n\n"
        
        # Process all files with SenseVoice first (for better model caching)
        sensevoice_cache_hits = 0
        if use_sensevoice:
            status += f"🎙️ Processing with SenseVoiceSmall...\n\n"
            for i, audio_path in enumerate(audio_paths):
                progress((0.1 + 0.35 * (i / total_files)), desc=f"SenseVoice {i+1}/{total_files}...")
                
                filename = os.path.basename(audio_path)
                status += f"[{i+1}/{total_files}] {filename}\n"
                
                result = transcribe_single_audio_sensevoice(audio_path, language)
                if result:
                    sensevoice_results.append(result)
                    if result.get('cache_hit', False):
                        sensevoice_cache_hits += 1
                        status += f"  💾 SenseVoice (cached): {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                    else:
                        status += f"  ✅ SenseVoice: {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                else:
                    status += f"  ❌ SenseVoice: Failed\n"
                
                status += "\n"
            
            status += f"✅ SenseVoice completed: {len(sensevoice_results)}/{total_files} files ({sensevoice_cache_hits} from cache)\n\n"
        
        # Then process all files with Whisper-v3-Cantonese (for better model caching)
        whisperv3_cache_hits = 0
        if use_whisperv3_cantonese:
            status += f"🎙️ Processing with Whisper-v3-Cantonese...\n\n"
            for i, audio_path in enumerate(audio_paths):
                progress((0.45 + 0.30 * (i / total_files)), desc=f"Whisper-v3 {i+1}/{total_files}...")
                
                filename = os.path.basename(audio_path)
                status += f"[{i+1}/{total_files}] {filename}\n"
                
                result = transcribe_single_audio_whisperv3_cantonese(audio_path)
                if result:
                    whisperv3_results.append(result)
                    if result.get('cache_hit', False):
                        whisperv3_cache_hits += 1
                        status += f"  💾 Whisper-v3 (cached): {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                    else:
                        status += f"  ✅ Whisper-v3: {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                else:
                    status += f"  ❌ Whisper-v3: Failed\n"
                
                status += "\n"
            
            status += f"✅ Whisper-v3-Cantonese completed: {len(whisperv3_results)}/{total_files} files ({whisperv3_cache_hits} from cache)\n\n"
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        progress(0.9, desc="Saving results...")
        
        # Save results to JSON files
        results_data = {
            "sensevoice": sensevoice_results if use_sensevoice else [],
            "whisperv3_cantonese": whisperv3_results if use_whisperv3_cantonese else []
        }
        json_path = os.path.join(temp_out_dir, "transcriptions.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        # Helper function to format speaker name
        def get_speaker_name(fname):
            m = re.search(r"speaker_(\d+)", fname)
            if m:
                return f"speaker_{m.group(1)}"
            return Path(fname).stem if fname else 'unknown'
        
        # Save SenseVoice conversation.txt
        sensevoice_txt_path = None
        sensevoice_conversation_content = ""
        if use_sensevoice and sensevoice_results:
            sensevoice_txt_path = os.path.join(temp_out_dir, "conversation_sensevoice.txt")
            with open(sensevoice_txt_path, 'w', encoding='utf-8') as f:
                for r in sensevoice_results:
                    fname = r.get('file', '')
                    speaker = get_speaker_name(fname)
                    f.write(f"{speaker}: {r.get('transcription', '')}\n")
            
            with open(sensevoice_txt_path, 'r', encoding='utf-8') as f:
                sensevoice_conversation_content = f.read()
        
        # Save Whisper-v3-Cantonese conversation.txt
        whisperv3_txt_path = None
        whisperv3_conversation_content = ""
        if use_whisperv3_cantonese and whisperv3_results:
            whisperv3_txt_path = os.path.join(temp_out_dir, "conversation_whisperv3_cantonese.txt")
            with open(whisperv3_txt_path, 'w', encoding='utf-8') as f:
                for r in whisperv3_results:
                    fname = r.get('file', '')
                    speaker = get_speaker_name(fname)
                    f.write(f"{speaker}: {r.get('transcription', '')}\n")
            
            with open(whisperv3_txt_path, 'r', encoding='utf-8') as f:
                whisperv3_conversation_content = f.read()
        
        # Create a zip file with all results
        zip_path = os.path.join(temp_out_dir, "batch_transcription_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(json_path, arcname="transcriptions.json")
            if sensevoice_txt_path:
                zipf.write(sensevoice_txt_path, arcname="conversation_sensevoice.txt")
            if whisperv3_txt_path:
                zipf.write(whisperv3_txt_path, arcname="conversation_whisperv3_cantonese.txt")
        
        progress(1.0, desc="Complete!")
        
        status += f"\n{'='*60}\n"
        status += f"✅ Batch transcription completed!\n"
        status += f"⏱️ Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)\n"
        if use_sensevoice:
            status += f"📊 SenseVoice processed: {len(sensevoice_results)}/{total_files} files\n"
        if use_whisperv3_cantonese:
            status += f"📊 Whisper-v3-Cantonese processed: {len(whisperv3_results)}/{total_files} files\n"
        status += f"📁 Results saved to:\n"
        status += f"   • transcriptions.json\n"
        if sensevoice_txt_path:
            status += f"   • conversation_sensevoice.txt\n"
        if whisperv3_txt_path:
            status += f"   • conversation_whisperv3_cantonese.txt\n"
        status += f"   • batch_transcription_results.zip\n"
        status += f"{'='*60}\n"
        
        return json_path, sensevoice_txt_path, whisperv3_txt_path, zip_path, sensevoice_conversation_content, whisperv3_conversation_content
        
    except Exception as e:
        error_msg = f"❌ Error during batch transcription: {str(e)}"
        error_msg += f"\n\n{traceback.format_exc()}"
        return None, None, None, None, error_msg, ""


def create_stt_tab():
    """Create and return the Batch Speech-to-Text tab (with integrated chopping)"""
    with gr.Tab("3️⃣ Auto-Diarize & Transcribe"):
        gr.Markdown("### Automatically diarize, chop, and transcribe audio")
        gr.Markdown("*Upload an audio file. Diarization will be cached in MongoDB for future use.*")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Input")
                
                stt_audio_input = gr.Audio(
                    label="Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                
                stt_overwrite_diarization = gr.Checkbox(
                    label="🔄 Overwrite cached diarization",
                    value=False,
                    info="If checked, will re-run diarization even if MongoDB cache exists"
                )
                
                stt_use_enhanced_format = gr.Checkbox(
                    label="📋 Enhanced format (metadata + timestamps)",
                    value=True,
                    info="Add metadata header and RTTM timestamps to transcriptions"
                )
                
                gr.Markdown("#### Model Selection")
                with gr.Row():
                    stt_use_sensevoice = gr.Checkbox(
                        label="SenseVoiceSmall",
                        value=True,
                        info="Chinese/Multi-language ASR"
                    )
                    stt_use_whisperv3_cantonese = gr.Checkbox(
                        label="Whisper-v3-Cantonese",
                        value=True,
                        info="Large Whisper v3 Cantonese"
                    )
                
                stt_language_dropdown = gr.Dropdown(
                    choices=["auto", "zh", "en", "yue", "ja", "ko"],
                    value="yue",
                    label="Language (for SenseVoice)",
                    info="Select the language of the audio"
                )
                
                stt_process_btn = gr.Button("🎯 Auto-Diarize & Transcribe", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                # Metadata section
                gr.Markdown("#### 📋 File Metadata")
                stt_metadata_output = gr.Textbox(
                    label="Extracted Metadata (Auto-extracted during processing)",
                    lines=5,
                    max_lines=10,
                    interactive=False,
                    show_copy_button=True,
                    placeholder="Metadata will appear here after processing..."
                )
                
                # LLM-labeled transcriptions (with 經紀/客戶)
                gr.Markdown("#### 🤖 LLM-Labeled Transcriptions (經紀/客戶)")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("##### SenseVoiceSmall (Labeled)")
                        stt_sensevoice_labeled_output = gr.Textbox(
                            label="SenseVoiceSmall (經紀/客戶)",
                            lines=10,
                            max_lines=20,
                            interactive=False,
                            show_copy_button=True,
                            placeholder="LLM-labeled SenseVoiceSmall results will appear here..."
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("##### Whisper-v3-Cantonese (Labeled)")
                        stt_whisperv3_labeled_output = gr.Textbox(
                            label="Whisper-v3-Cantonese (經紀/客戶)",
                            lines=10,
                            max_lines=20,
                            interactive=False,
                            show_copy_button=True,
                            placeholder="LLM-labeled Whisper-v3 results will appear here..."
                        )
                
                # LLM identification log
                stt_llm_log_output = gr.Textbox(
                    label="🧠 LLM Speaker Identification Log",
                    lines=5,
                    max_lines=10,
                    interactive=False,
                    show_copy_button=True,
                    placeholder="LLM reasoning for speaker identification..."
                )
                
                stt_zip_download = gr.File(
                    label="Download All Results (ZIP)",
                    interactive=False
                )
                stt_status_output = gr.Textbox(
                    label="Status Log",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    show_copy_button=True
                )
        
        # Wire up the main transcription button
        stt_process_btn.click(
            fn=process_chop_and_transcribe,
            inputs=[stt_audio_input, stt_language_dropdown, stt_use_sensevoice, stt_use_whisperv3_cantonese, stt_overwrite_diarization, gr.Number(value=100, visible=False), stt_use_enhanced_format],
            outputs=[
                stt_metadata_output,  # metadata_json
                gr.File(visible=False),  # json_file
                gr.File(visible=False),  # sensevoice_txt
                gr.File(visible=False),  # whisperv3_txt
                stt_zip_download,  # zip_file
                stt_sensevoice_labeled_output,  # sensevoice_labeled
                stt_whisperv3_labeled_output,  # whisperv3_labeled
                stt_llm_log_output,  # llm_log
                stt_status_output  # status
            ]
        )

