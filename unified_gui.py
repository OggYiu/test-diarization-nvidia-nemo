"""
Unified Gradio GUI combining all phone call analysis tools:
1. Speaker Diarization
2. Audio Chopper
3. Batch Speech-to-Text
4. LLM Analysis
5. Speaker Separation
6. Audio Enhancement
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

def process_audio(audio_file):
    """
    Process audio file through diarization and return results.
    
    Args:
        audio_file: Audio file from Gradio interface
    
    Returns:
        tuple: (rttm_content, status_message, output_directory_path)
    """
    if audio_file is None:
        return "Please upload an audio file.", "❌ No file uploaded", ""
    
    try:
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="diarization_")
        
        # Run diarization with timing
        status = f"🔄 Processing audio file: {os.path.basename(audio_file)}\n"
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
        
        summary = f"📈 Summary:\n"
        summary += f"  • Total segments: {num_segments}\n"
        summary += f"  • Detected speakers: {len(speakers)}\n"
        summary += f"  • Speaker IDs: {', '.join(sorted(speakers))}\n"
        summary += f"  • Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)\n\n"
        
        return rttm_content, summary + status, temp_out_dir
        
    except Exception as e:
        error_msg = f"❌ Error during diarization: {str(e)}"
        return error_msg, error_msg, ""


# ============================================================================
# 2. AUDIO CHOPPER FUNCTIONS
# ============================================================================

def process_audio_chopping(audio_file, rttm_file, rttm_text):
    padding_ms = 200
    """
    Process audio file and RTTM file to chop audio into segments.
    
    Args:
        audio_file: Audio file from Gradio interface
        rttm_file: RTTM file from Gradio interface
        rttm_text: RTTM text string pasted by user
    
    Returns:
        tuple: (zip_file_path, status_message, segment_details)
    """
    if audio_file is None:
        return None, "❌ No audio file uploaded", ""
    
    if rttm_file is None and (not rttm_text or not rttm_text.strip()):
        return None, "❌ No RTTM file uploaded or RTTM text provided", ""
    
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
        
        return zip_path, status, segment_details
        
    except Exception as e:
        error_msg = f"❌ Error during audio chopping: {str(e)}"
        error_msg += f"\n\n{traceback.format_exc()}"
        return None, error_msg, ""


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


def process_batch_transcription(audio_files, zip_file, language, progress=gr.Progress()):
    """
    Process multiple audio files for transcription.
    
    Args:
        audio_files: List of audio files from Gradio interface
        zip_file: Zip file containing audio files
        language: Language code for transcription
    
    Returns:
        tuple: (json_file, txt_file, zip_file, status_message, conversation_content)
    """
    if (not audio_files or len(audio_files) == 0) and not zip_file:
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
]

DEFAULT_MODEL = MODEL_OPTIONS[0]
DEFAULT_OLLAMA_URL = "http://192.168.61.2:11434"
DEFAULT_SYSTEM_MESSAGE = (
    "你是一位精通粵語以及香港股市的分析師。請用繁體中文回應，"
    "並從下方對話中判斷誰是券商、誰是客戶，整理最終下單（股票代號、買/賣、價格、數量），"
)


def analyze_with_llm(
    prompt_text: str,
    prompt_file,
    model: str,
    ollama_url: str,
    system_message: str,
    temperature: float,
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
            ("system", system_message),
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
        
        return status, response_content
        
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

def process_speaker_separation(audio_file, progress=gr.Progress()):
    """
    Perform speaker separation on audio file.
    
    Args:
        audio_file: Audio file from Gradio interface
    
    Returns:
        tuple: (speaker_0_file, speaker_1_file, status_message)
    """
    if audio_file is None:
        return None, None, "❌ No audio file uploaded"
    
    try:
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="speaker_separated_")
        
        status = f"🔄 Processing audio file: {os.path.basename(audio_file)}\n"
        status += f"📁 Output directory: {temp_out_dir}\n\n"
        
        progress(0.1, desc="Starting diarization...")
        
        # Step 1: Perform diarization
        status += "=" * 60 + "\n"
        status += "STEP 1: Speaker Diarization with NVIDIA NeMo\n"
        status += "=" * 60 + "\n"
        
        start_time = time.time()
        
        try:
            rttm_file = perform_diarization(audio_file, temp_out_dir)
            status += f"✅ Diarization complete!\n"
            status += f"   RTTM file: {rttm_file}\n\n"
        except Exception as e:
            error_msg = f"❌ Error during diarization: {str(e)}\n{traceback.format_exc()}"
            return None, None, status + error_msg
        
        progress(0.5, desc="Separating speakers...")
        
        # Step 2: Separate speakers
        status += "=" * 60 + "\n"
        status += "STEP 2: Speaker Separation\n"
        status += "=" * 60 + "\n"
        
        try:
            speakers = separate_speakers(audio_file, rttm_file, temp_out_dir)
            status += f"✅ Speaker separation complete!\n"
            status += f"   Found {len(speakers) if speakers else 2} speaker(s)\n\n"
        except Exception as e:
            error_msg = f"❌ Error during speaker separation: {str(e)}\n{traceback.format_exc()}"
            return None, None, status + error_msg
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        progress(1.0, desc="Complete!")
        
        # Locate the separated audio files
        speaker_files = []
        for i in range(len(speakers) if speakers else 2):
            speaker_file = os.path.join(temp_out_dir, f"speaker_{i}_only.wav")
            if os.path.exists(speaker_file):
                speaker_files.append(speaker_file)
        
        status += "=" * 60 + "\n"
        status += "✅ COMPLETE!\n"
        status += "=" * 60 + "\n"
        status += f"⏱️ Total processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)\n"
        status += f"📊 Separated files created: {len(speaker_files)}\n\n"
        
        status += "📁 Generated files:\n"
        for i, speaker_file in enumerate(speaker_files):
            file_size = os.path.getsize(speaker_file)
            status += f"  {i+1}. speaker_{i}_only.wav ({file_size:,} bytes)\n"
        
        # Return files (up to 2 speakers for now)
        speaker_0 = speaker_files[0] if len(speaker_files) > 0 else None
        speaker_1 = speaker_files[1] if len(speaker_files) > 1 else None
        
        # Add more speakers to status if they exist
        if len(speaker_files) > 2:
            status += f"\n⚠️ Note: {len(speaker_files)} speakers detected. Only first 2 available for download in GUI.\n"
            status += f"   All files available in: {temp_out_dir}\n"
        
        return speaker_0, speaker_1, status
        
    except Exception as e:
        error_msg = f"❌ Error during speaker separation: {str(e)}"
        error_msg += f"\n\n{traceback.format_exc()}"
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
                        
                        with gr.Tab("📁 Upload RTTM File"):
                            chop_rttm_input = gr.File(
                                label="RTTM File",
                                file_types=[".rttm"],
                                file_count="single"
                            )
                        
                        with gr.Tab("📝 Paste RTTM Text"):
                            chop_rttm_text = gr.Textbox(
                                label="RTTM Content",
                                placeholder="Paste RTTM content here...\nExample:\nSPEAKER test 1 0.000 2.500 <NA> <NA> speaker_0 <NA> <NA>\nSPEAKER test 1 2.500 3.200 <NA> <NA> speaker_1 <NA> <NA>",
                                lines=10,
                                max_lines=20
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
                
                chop_process_btn.click(
                    fn=process_audio_chopping,
                    inputs=[chop_audio_input, chop_rttm_input, chop_rttm_text],
                    outputs=[chop_download_output, chop_status_output, chop_segment_details]
                )
            
            # ================================================================
            # TAB 3: BATCH SPEECH-TO-TEXT
            # ================================================================
            with gr.Tab("3️⃣ Batch Speech-to-Text"):
                gr.Markdown("### Transcribe multiple audio segments to text")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Input (Choose one or both methods)")
                        
                        with gr.Tab("📦 Upload Zip File"):
                            stt_zip_file = gr.File(
                                label="Upload Zip File Containing Audio Files",
                                file_count="single",
                                file_types=[".zip"],
                                type="filepath"
                            )
                            gr.Markdown("*The zip file should contain audio files (.wav, .mp3, .flac, .m4a, .ogg, .opus)*")
                        
                        with gr.Tab("📁 Upload Audio Files"):
                            stt_audio_files = gr.File(
                                label="Upload Audio Files",
                                file_count="multiple",
                                file_types=[".wav", ".mp3", ".flac", ".m4a"],
                                type="filepath"
                            )
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
                    inputs=[stt_audio_files, stt_zip_file, stt_language_dropdown],
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
                        
                        llm_status_box = gr.Textbox(
                            label="狀態",
                            lines=6,
                            interactive=False,
                        )
                        
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
                    ],
                    outputs=[llm_status_box, llm_response_box],
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
                        sep_process_btn = gr.Button("🎵 Separate Speakers", variant="primary", size="lg")
                        
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
        
        gr.Markdown(
            """
            ---
            ### 💡 Workflow Tips:
            1. **Audio Enhancement**: Clean up poor quality recordings before processing
            2. **Diarization**: Upload a phone call recording to identify speakers
            3. **Audio Chopper**: Use the RTTM output to split the audio by speaker
            4. **Transcription**: Convert the chopped segments to text
            5. **LLM Analysis**: Analyze the conversation for insights
            6. **Speaker Separation**: Create separate audio files for each speaker (with others silenced)
            
            ### 🎯 Recommended Workflow for Poor Quality Recordings:
            1. Start with **Audio Enhancement** to improve quality
            2. Use enhanced audio in **Diarization** for better speaker detection
            3. Continue with other tools as needed
            """
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

