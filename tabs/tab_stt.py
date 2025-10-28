"""
Tab 3: Batch Speech-to-Text
Transcribe multiple audio segments to text
"""

import os
import re
import time
import json
import shutil
import tempfile
import zipfile
import traceback
import urllib.request
from pathlib import Path
import gradio as gr

from funasr import AutoModel
from batch_stt import format_str_v3, load_audio
from audio_chopper import chop_audio_file, read_rttm_file

# Import for WSYue-ASR model
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download
import librosa
import torch
import sys

# Global variables for model management
sensevoice_model = None
wsyue_model = None
wsyue_processor = None
current_sensevoice_loaded = False
current_wsyue_loaded = False


def initialize_sensevoice_model():
    """Initialize the SenseVoice model."""
    global sensevoice_model, current_sensevoice_loaded
    
    # Only reload if not already loaded
    if current_sensevoice_loaded and sensevoice_model is not None:
        return f"✅ SenseVoiceSmall already loaded"
    
    status = f"🔄 Loading SenseVoiceSmall model...\n"
    
    try:
        sensevoice_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=False,
            disable_update=True,
        )
        current_sensevoice_loaded = True
        status += "✅ SenseVoiceSmall loaded successfully!"
        return status
    except Exception as e:
        status += f"❌ Failed to load SenseVoiceSmall: {str(e)}"
        return status


def initialize_wsyue_model():
    """Initialize the WSYue-ASR (Whisper Cantonese) model."""
    global wsyue_model, wsyue_processor, current_wsyue_loaded
    
    # Only reload if not already loaded
    if current_wsyue_loaded and wsyue_model is not None:
        return f"✅ WSYue-ASR already loaded"
    
    status = f"🔄 Loading WSYue-ASR (Whisper Cantonese) model...\n"
    
    try:
        # Download the model checkpoint from Hugging Face
        model_path = hf_hub_download(
            repo_id="ASLP-lab/WSYue-ASR",
            filename="whisper_medium_yue/whisper_medium_yue.pt",
            cache_dir="./model_cache"
        )
        status += f"  ✓ Model checkpoint downloaded\n"
        
        # Load the base Whisper medium model architecture
        wsyue_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
        wsyue_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        status += f"  ✓ Base Whisper architecture loaded\n"
        
        # Load the fine-tuned weights
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # The checkpoint might be stored in different formats
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load the state dict into the model
        try:
            wsyue_model.load_state_dict(state_dict, strict=False)
            status += f"  ✓ Fine-tuned weights loaded\n"
        except Exception as e:
            # Try to load compatible weights only
            model_dict = wsyue_model.state_dict()
            compatible_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(compatible_dict)
            wsyue_model.load_state_dict(model_dict)
            status += f"  ✓ Loaded {len(compatible_dict)}/{len(state_dict)} weight tensors\n"
        
        # Set model to evaluation mode
        wsyue_model.eval()
        current_wsyue_loaded = True
        
        status += "✅ WSYue-ASR loaded successfully!"
        return status
    except Exception as e:
        status += f"❌ Failed to load WSYue-ASR: {str(e)}"
        return status


def transcribe_single_audio_sensevoice(audio_path, language="auto"):
    """Transcribe a single audio file using SenseVoiceSmall model."""
    global sensevoice_model
    
    if sensevoice_model is None:
        return None
    
    # Load audio
    audio_array, sample_rate = load_audio(audio_path)
    if audio_array is None:
        return None
    
    # Run inference
    try:
        result = sensevoice_model.generate(
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
        print(f"Error transcribing with SenseVoice {audio_path}: {e}")
        return None


def transcribe_single_audio_wsyue(audio_path):
    """Transcribe a single audio file using WSYue-ASR model."""
    global wsyue_model, wsyue_processor
    
    if wsyue_model is None or wsyue_processor is None:
        return None
    
    # Load audio
    try:
        audio, sampling_rate = librosa.load(audio_path, sr=16000)
        
        # Process the audio
        input_features = wsyue_processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = wsyue_model.generate(
                input_features,
                language="zh",
                task="transcribe",
                max_length=448
            )
        
        # Decode the transcription
        transcription = wsyue_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcription_text = transcription[0]
        
        return {
            "file": os.path.basename(audio_path),
            "path": audio_path,
            "transcription": transcription_text,
            "raw_transcription": transcription_text
        }
    except Exception as e:
        print(f"Error transcribing with WSYue {audio_path}: {e}")
        return None


def process_chop_and_transcribe(audio_file, rttm_text, language, use_sensevoice, use_wsyue, padding_ms=100, progress=gr.Progress()):
    """
    Integrated pipeline: Chop audio based on RTTM, then transcribe the segments.
    
    Args:
        audio_file: Audio file from Gradio interface
        rttm_text: RTTM text string pasted by user
        language: Language code for transcription
        use_sensevoice: Whether to use SenseVoiceSmall model
        use_wsyue: Whether to use WSYue-ASR model
        padding_ms: Padding in milliseconds for chopping (default: 100)
    
    Returns:
        tuple: (json_file, sensevoice_txt, wsyue_txt, zip_file, sensevoice_conversation, wsyue_conversation, status_message)
    """
    if audio_file is None:
        return None, None, None, None, "", "", "❌ No audio file uploaded"
    
    if not rttm_text or not rttm_text.strip():
        return None, None, None, None, "", "", "❌ No RTTM text provided"
    
    if not use_sensevoice and not use_wsyue:
        return None, None, None, None, "", "", "⚠️ Please select at least one model"
    
    try:
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="chop_transcribe_")
        chopped_audio_dir = os.path.join(temp_out_dir, "chopped_segments")
        os.makedirs(chopped_audio_dir, exist_ok=True)
        
        status = f"🔄 Starting integrated pipeline (Chop + Transcribe)...\n\n"
        status += f"📁 Audio file: {os.path.basename(audio_file)}\n"
        status += f"⏱️ Padding: {padding_ms} ms\n\n"
        
        # Step 1: Save RTTM text to a temporary file
        progress(0, desc="Preparing RTTM...")
        temp_rttm_file = os.path.join(temp_out_dir, "temp_rttm.rttm")
        with open(temp_rttm_file, 'w', encoding='utf-8') as f:
            f.write(rttm_text.strip())
        status += "✅ RTTM text saved\n"
        
        # Step 2: Read RTTM segments
        progress(0.05, desc="Reading RTTM...")
        status += "📖 Reading RTTM data...\n"
        segments = read_rttm_file(temp_rttm_file)
        
        if not segments:
            return None, None, None, None, "", "", "❌ No segments found in RTTM text!"
        
        status += f"✅ Found {len(segments)} segments\n\n"
        
        # Count speakers
        speakers = set(seg['speaker'] for seg in segments)
        status += f"👥 Detected speakers: {len(speakers)} ({', '.join(sorted(speakers))})\n\n"
        
        # Step 3: Chop audio file
        progress(0.1, desc="Chopping audio...")
        status += "✂️ Chopping audio into segments...\n"
        chop_audio_file(audio_file, segments, chopped_audio_dir, padding_ms)
        
        # Get list of chopped files
        chopped_files = sorted([
            os.path.join(chopped_audio_dir, f) 
            for f in os.listdir(chopped_audio_dir) 
            if f.endswith('.wav')
        ])
        
        status += f"✅ Audio chopped into {len(chopped_files)} segments\n\n"
        
        # Step 4: Initialize models
        progress(0.2, desc="Loading models...")
        if use_sensevoice:
            sensevoice_status = initialize_sensevoice_model()
            status += sensevoice_status + "\n"
        if use_wsyue:
            wsyue_status = initialize_wsyue_model()
            status += wsyue_status + "\n"
        status += "\n"
        
        if use_sensevoice and sensevoice_model is None:
            return None, None, None, None, "", "", status + "❌ Failed to load SenseVoiceSmall model"
        if use_wsyue and wsyue_model is None:
            return None, None, None, None, "", "", status + "❌ Failed to load WSYue-ASR model"
        
        # Step 5: Transcribe chopped segments
        status += f"📝 Transcribing {len(chopped_files)} segment(s)...\n\n"
        start_time = time.time()
        
        sensevoice_results = []
        wsyue_results = []
        total_files = len(chopped_files)
        
        # Process all files with SenseVoice first
        if use_sensevoice:
            status += f"🎙️ Processing with SenseVoiceSmall...\n\n"
            for i, audio_path in enumerate(chopped_files):
                progress((0.2 + 0.35 * (i / total_files)), desc=f"SenseVoice {i+1}/{total_files}...")
                
                filename = os.path.basename(audio_path)
                status += f"[{i+1}/{total_files}] {filename}\n"
                
                result = transcribe_single_audio_sensevoice(audio_path, language)
                if result:
                    sensevoice_results.append(result)
                    status += f"  ✅ SenseVoice: {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                else:
                    status += f"  ❌ SenseVoice: Failed\n"
                
                status += "\n"
            
            status += f"✅ SenseVoice completed: {len(sensevoice_results)}/{total_files} files\n\n"
        
        # Then process all files with WSYue
        if use_wsyue:
            status += f"🎙️ Processing with WSYue-ASR...\n\n"
            for i, audio_path in enumerate(chopped_files):
                progress((0.55 + 0.35 * (i / total_files)), desc=f"WSYue {i+1}/{total_files}...")
                
                filename = os.path.basename(audio_path)
                status += f"[{i+1}/{total_files}] {filename}\n"
                
                result = transcribe_single_audio_wsyue(audio_path)
                if result:
                    wsyue_results.append(result)
                    status += f"  ✅ WSYue: {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                else:
                    status += f"  ❌ WSYue: Failed\n"
                
                status += "\n"
            
            status += f"✅ WSYue completed: {len(wsyue_results)}/{total_files} files\n\n"
        
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
            "wsyue": wsyue_results if use_wsyue else []
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
        
        # Save WSYue conversation.txt
        wsyue_txt_path = None
        wsyue_conversation_content = ""
        if use_wsyue and wsyue_results:
            wsyue_txt_path = os.path.join(temp_out_dir, "conversation_wsyue.txt")
            with open(wsyue_txt_path, 'w', encoding='utf-8') as f:
                for r in wsyue_results:
                    fname = r.get('file', '')
                    speaker = get_speaker_name(fname)
                    f.write(f"{speaker}: {r.get('transcription', '')}\n")
            
            with open(wsyue_txt_path, 'r', encoding='utf-8') as f:
                wsyue_conversation_content = f.read()
        
        # Create a zip file with all results
        zip_path = os.path.join(temp_out_dir, "transcription_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(json_path, arcname="transcriptions.json")
            if sensevoice_txt_path:
                zipf.write(sensevoice_txt_path, arcname="conversation_sensevoice.txt")
            if wsyue_txt_path:
                zipf.write(wsyue_txt_path, arcname="conversation_wsyue.txt")
        
        # Step 6: Clean up temporary chopped files
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
        if use_wsyue:
            status += f"📊 WSYue processed: {len(wsyue_results)}/{total_files} segments\n"
        status += f"📁 Results saved to:\n"
        status += f"   • transcriptions.json\n"
        if sensevoice_txt_path:
            status += f"   • conversation_sensevoice.txt\n"
        if wsyue_txt_path:
            status += f"   • conversation_wsyue.txt\n"
        status += f"   • transcription_results.zip\n"
        status += f"{'='*60}\n"
        
        return json_path, sensevoice_txt_path, wsyue_txt_path, zip_path, sensevoice_conversation_content, wsyue_conversation_content, status
        
    except Exception as e:
        error_msg = f"❌ Error during pipeline: {str(e)}"
        error_msg += f"\n\n{traceback.format_exc()}"
        return None, None, None, None, "", "", error_msg


def process_batch_transcription(audio_files, zip_file, link_or_path, language, use_sensevoice, use_wsyue, progress=gr.Progress()):
    """
    Process multiple audio files for transcription.
    
    Args:
        audio_files: List of audio files from Gradio interface
        zip_file: Zip file containing audio files
        link_or_path: URL to a zip file or local folder path
        language: Language code for transcription
        use_sensevoice: Whether to use SenseVoiceSmall model
        use_wsyue: Whether to use WSYue-ASR model
    
    Returns:
        tuple: (json_file, txt_file, zip_file, sensevoice_conversation, wsyue_conversation)
    """
    if (not audio_files or len(audio_files) == 0) and not zip_file and not link_or_path:
        return None, None, None, "", ""
    
    if not use_sensevoice and not use_wsyue:
        return None, None, None, "⚠️ Please select at least one model", ""
    
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
        if use_wsyue:
            wsyue_status = initialize_wsyue_model()
            status += wsyue_status + "\n"
        status += "\n"
        
        if use_sensevoice and sensevoice_model is None:
            return None, None, None, "❌ Failed to load SenseVoiceSmall model", ""
        if use_wsyue and wsyue_model is None:
            return None, None, None, "", "❌ Failed to load WSYue-ASR model"
        
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
                    return None, None, None, None, f"❌ Error downloading from URL: {str(e)}", ""
            
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
                    return None, None, None, None, f"❌ Error reading folder: {str(e)}", ""
            
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
        if use_wsyue:
            models_used.append("WSYue-ASR")
        status += f"🤖 Models: {', '.join(models_used)}\n\n"
        
        # Sort files by name (for segment_001.wav, segment_002.wav ordering)
        audio_paths.sort(key=lambda p: os.path.basename(p))
        
        start_time = time.time()
        
        # Process each file with selected models
        sensevoice_results = []
        wsyue_results = []
        total_files = len(audio_paths)
        
        status += f"📝 Processing {total_files} audio file(s)...\n\n"
        
        # Process all files with SenseVoice first (for better model caching)
        if use_sensevoice:
            status += f"🎙️ Processing with SenseVoiceSmall...\n\n"
            for i, audio_path in enumerate(audio_paths):
                progress((0.1 + 0.35 * (i / total_files)), desc=f"SenseVoice {i+1}/{total_files}...")
                
                filename = os.path.basename(audio_path)
                status += f"[{i+1}/{total_files}] {filename}\n"
                
                result = transcribe_single_audio_sensevoice(audio_path, language)
                if result:
                    sensevoice_results.append(result)
                    status += f"  ✅ SenseVoice: {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                else:
                    status += f"  ❌ SenseVoice: Failed\n"
                
                status += "\n"
            
            status += f"✅ SenseVoice completed: {len(sensevoice_results)}/{total_files} files\n\n"
        
        # Then process all files with WSYue (for better model caching)
        if use_wsyue:
            status += f"🎙️ Processing with WSYue-ASR...\n\n"
            for i, audio_path in enumerate(audio_paths):
                progress((0.45 + 0.35 * (i / total_files)), desc=f"WSYue {i+1}/{total_files}...")
                
                filename = os.path.basename(audio_path)
                status += f"[{i+1}/{total_files}] {filename}\n"
                
                result = transcribe_single_audio_wsyue(audio_path)
                if result:
                    wsyue_results.append(result)
                    status += f"  ✅ WSYue: {result['transcription'][:80]}{'...' if len(result['transcription']) > 80 else ''}\n"
                else:
                    status += f"  ❌ WSYue: Failed\n"
                
                status += "\n"
            
            status += f"✅ WSYue completed: {len(wsyue_results)}/{total_files} files\n\n"
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        progress(0.9, desc="Saving results...")
        
        # Save results to JSON files
        results_data = {
            "sensevoice": sensevoice_results if use_sensevoice else [],
            "wsyue": wsyue_results if use_wsyue else []
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
        
        # Save WSYue conversation.txt
        wsyue_txt_path = None
        wsyue_conversation_content = ""
        if use_wsyue and wsyue_results:
            wsyue_txt_path = os.path.join(temp_out_dir, "conversation_wsyue.txt")
            with open(wsyue_txt_path, 'w', encoding='utf-8') as f:
                for r in wsyue_results:
                    fname = r.get('file', '')
                    speaker = get_speaker_name(fname)
                    f.write(f"{speaker}: {r.get('transcription', '')}\n")
            
            with open(wsyue_txt_path, 'r', encoding='utf-8') as f:
                wsyue_conversation_content = f.read()
        
        # Create a zip file with all results
        zip_path = os.path.join(temp_out_dir, "batch_transcription_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(json_path, arcname="transcriptions.json")
            if sensevoice_txt_path:
                zipf.write(sensevoice_txt_path, arcname="conversation_sensevoice.txt")
            if wsyue_txt_path:
                zipf.write(wsyue_txt_path, arcname="conversation_wsyue.txt")
        
        progress(1.0, desc="Complete!")
        
        status += f"\n{'='*60}\n"
        status += f"✅ Batch transcription completed!\n"
        status += f"⏱️ Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)\n"
        if use_sensevoice:
            status += f"📊 SenseVoice processed: {len(sensevoice_results)}/{total_files} files\n"
        if use_wsyue:
            status += f"📊 WSYue processed: {len(wsyue_results)}/{total_files} files\n"
        status += f"📁 Results saved to:\n"
        status += f"   • transcriptions.json\n"
        if sensevoice_txt_path:
            status += f"   • conversation_sensevoice.txt\n"
        if wsyue_txt_path:
            status += f"   • conversation_wsyue.txt\n"
        status += f"   • batch_transcription_results.zip\n"
        status += f"{'='*60}\n"
        
        return json_path, sensevoice_txt_path, wsyue_txt_path, zip_path, sensevoice_conversation_content, wsyue_conversation_content
        
    except Exception as e:
        error_msg = f"❌ Error during batch transcription: {str(e)}"
        error_msg += f"\n\n{traceback.format_exc()}"
        return None, None, None, None, error_msg, ""


def create_stt_tab():
    """Create and return the Batch Speech-to-Text tab (with integrated chopping)"""
    with gr.Tab("3️⃣ Chop & Transcribe"):
        gr.Markdown("### Chop audio by RTTM and transcribe segments")
        gr.Markdown("*Upload an audio file and paste RTTM text to automatically chop and transcribe*")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Input")
                
                stt_audio_input = gr.Audio(
                    label="Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                
                stt_rttm_text = gr.Textbox(
                    label="RTTM Content",
                    placeholder="Paste RTTM content here...\n\nExample:\nSPEAKER test 1 0.000 2.500 <NA> <NA> speaker_0 <NA> <NA>\nSPEAKER test 1 2.500 3.200 <NA> <NA> speaker_1 <NA> <NA>",
                    lines=8,
                    max_lines=15
                )
                
                gr.Markdown("#### Model Selection")
                with gr.Row():
                    stt_use_sensevoice = gr.Checkbox(
                        label="SenseVoiceSmall",
                        value=True,
                        info="Chinese/Multi-language ASR"
                    )
                    stt_use_wsyue = gr.Checkbox(
                        label="WSYue-ASR (Whisper Cantonese)",
                        value=False,
                        info="Cantonese-optimized Whisper"
                    )
                
                stt_language_dropdown = gr.Dropdown(
                    choices=["auto", "zh", "en", "yue", "ja", "ko"],
                    value="yue",
                    label="Language (for SenseVoice)",
                    info="Select the language of the audio"
                )
                
                stt_process_btn = gr.Button("✂️🎙️ Chop & Transcribe", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("#### Processing Status")
                stt_status_output = gr.Textbox(
                    label="Status Log",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    show_copy_button=True
                )
                
                gr.Markdown("#### Transcription Results")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("##### 📝 SenseVoiceSmall")
                        stt_sensevoice_output = gr.Textbox(
                            label="SenseVoiceSmall Transcription",
                            lines=15,
                            max_lines=25,
                            interactive=False,
                            show_copy_button=True,
                            placeholder="SenseVoiceSmall results will appear here..."
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("##### 📝 WSYue-ASR (Whisper Cantonese)")
                        stt_wsyue_output = gr.Textbox(
                            label="WSYue-ASR Transcription",
                            lines=15,
                            max_lines=25,
                            interactive=False,
                            show_copy_button=True,
                            placeholder="WSYue-ASR results will appear here..."
                        )
                
                stt_zip_download = gr.File(
                    label="Download All Results (ZIP)",
                    interactive=False
                )
        
        stt_process_btn.click(
            fn=process_chop_and_transcribe,
            inputs=[stt_audio_input, stt_rttm_text, stt_language_dropdown, stt_use_sensevoice, stt_use_wsyue],
            outputs=[gr.File(visible=False), gr.File(visible=False), gr.File(visible=False), stt_zip_download, stt_sensevoice_output, stt_wsyue_output, stt_status_output]
        )

