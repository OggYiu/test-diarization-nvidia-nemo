import os
import gradio as gr
import tempfile
import shutil
import time
import json
from pathlib import Path
import zipfile
from batch_stt import transcribe_audio, format_str_v3, load_audio
import torch
import re
from funasr import AutoModel

# Global model variable
model = None
current_model_name = None


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


def process_batch_transcription(audio_files, language, progress=gr.Progress()):
    """
    Process multiple audio files for transcription.
    
    Args:
        audio_files: List of audio files from Gradio interface
        language: Language code for transcription
    
    Returns:
        tuple: (json_file, txt_file, zip_file, status_message)
    """
    if not audio_files or len(audio_files) == 0:
        return None, None, None, "❌ No audio files uploaded"
    
    try:
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="batch_stt_")
        audio_dir = os.path.join(temp_out_dir, "audio_files")
        os.makedirs(audio_dir, exist_ok=True)
        
        status = f"🔄 Starting batch transcription...\n"
        status += f"📊 Total files: {len(audio_files)}\n"
        status += f"🌐 Language: {language}\n"
        status += f"🤖 Model: SenseVoiceSmall\n\n"
        
        # Initialize model
        progress(0, desc="Loading model...")
        model_status = initialize_model()
        status += model_status + "\n\n"
        
        if model is None:
            return None, None, None, status + "\n❌ Failed to initialize model"
        
        # Copy uploaded files to temp directory and sort them
        progress(0.1, desc="Preparing files...")
        audio_paths = []
        for audio_file in audio_files:
            if audio_file is not None:
                # Get original filename
                original_name = os.path.basename(audio_file)
                dest_path = os.path.join(audio_dir, original_name)
                shutil.copy2(audio_file, dest_path)
                audio_paths.append(dest_path)
        
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
        
        return json_path, txt_path, zip_path, status
        
    except Exception as e:
        error_msg = f"❌ Error during batch transcription: {str(e)}"
        import traceback
        error_msg += f"\n\n{traceback.format_exc()}"
        return None, None, None, error_msg


def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Batch Speech-to-Text", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎤 Batch Speech-to-Text Transcription
            
            Upload multiple audio files to transcribe them using the SenseVoice model.
            The tool will process each file and generate transcriptions with emotion and event detection.
            
            **Features:**
            - 🌐 Multi-language support (Chinese, English, Japanese, Korean, Cantonese)
            - 😊 Emotion detection (Happy, Sad, Angry, Surprised, etc.)
            - 🎵 Event detection (Music, Applause, Laughter, Cough, etc.)
            - 📝 Batch processing of multiple files
            - 💾 Export to JSON and plain text formats
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")
                
                audio_files = gr.File(
                    label="Upload Audio Files",
                    file_count="multiple",
                    file_types=[".wav", ".mp3", ".flac", ".m4a"],
                    type="filepath"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=["auto", "zh", "en", "yue", "ja", "ko"],
                    value="yue",
                    label="Language",
                    info="Select the language of the audio"
                )
                
                process_btn = gr.Button("🚀 Start Transcription", variant="primary", size="lg")
                
                gr.Markdown(
                    """
                    ### 📝 Notes:
                    - **Supported formats:** WAV, MP3, FLAC, M4A
                    - **Model:** SenseVoiceSmall
                    - **Language codes:**
                      - `auto`: Auto-detect
                      - `zh`: Mandarin Chinese
                      - `en`: English
                      - `yue`: Cantonese (default)
                      - `ja`: Japanese
                      - `ko`: Korean
                    - **File ordering:** Files are processed in alphabetical order
                    - **Processing time:** Varies based on audio length
                    """
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📤 Results")
                
                status_output = gr.Textbox(
                    label="Processing Status",
                    lines=20,
                    max_lines=30,
                    interactive=False
                )
                
                with gr.Row():
                    json_download = gr.File(
                        label="Download JSON",
                        interactive=False
                    )
                    
                    txt_download = gr.File(
                        label="Download Conversation Text",
                        interactive=False
                    )
                
                zip_download = gr.File(
                    label="Download All Results (ZIP)",
                    interactive=False
                )
        
        # Examples section
        gr.Markdown("### 📂 Example Files")
        gr.Markdown(
            """
            Upload your chopped audio segments (e.g., from the Audio Chopper tool) or any audio files you want to transcribe.
            The files will be processed in alphabetical order, so name them accordingly (e.g., segment_001.wav, segment_002.wav).
            """
        )
        
        # Footer with tips
        gr.Markdown(
            """
            ---
            ### 💡 Tips:
            - **Batch processing:** Upload all your audio segments at once for efficient processing
            - **File naming:** For conversation transcripts, name files like `segment_001.wav`, `segment_002.wav`
            - **Speaker identification:** Files with `speaker_N` in the name will be labeled accordingly in the output
            - **Emotions & Events:** The model automatically detects emotions (😊😔😡) and events (🎼👏😀)
            - **Output formats:**
              - `transcriptions.json`: Detailed results with raw and formatted text
              - `conversation.txt`: Simple conversation format for easy reading
            
            ### 🔗 Workflow:
            1. **Diarization** → Identify speakers and their speaking times
            2. **Audio Chopper** → Split audio into segments based on speaker turns
            3. **Batch STT** (this tool) → Transcribe each segment
            4. **Analysis** → Analyze the conversation using LLM
            """
        )
        
        # Connect the button to the processing function
        process_btn.click(
            fn=process_batch_transcription,
            inputs=[audio_files, language_dropdown],
            outputs=[json_download, txt_download, zip_download, status_output]
        )
    
    return demo


if __name__ == "__main__":
    print("🚀 Starting Batch Speech-to-Text GUI...")
    print("📝 This tool transcribes multiple audio files using SenseVoice.")
    print("=" * 60)
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True
    )

