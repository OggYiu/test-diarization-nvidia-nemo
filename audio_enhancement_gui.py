#!/usr/bin/env python
# coding=utf-8
"""
Audio Enhancement GUI
Gradio interface for audio enhancement optimized for phone recordings
"""

import os
import gradio as gr
import tempfile
import traceback
from pathlib import Path
from audio_enhancement import AudioEnhancer, transcribe_enhanced_audio, FUNASR_AVAILABLE


def process_enhancement(
    audio_file, 
    enable_transcription=False,
    language="yue",
    progress=gr.Progress()
):
    """
    Process audio enhancement through GUI.
    
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
            import torchaudio
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
            if not FUNASR_AVAILABLE:
                status += "⚠️  Cannot transcribe: funasr not available\n"
            else:
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


def create_enhancement_interface():
    """Create Gradio interface for audio enhancement."""
    
    with gr.Blocks(title="Audio Enhancement for Phone Recordings", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎵 Audio Enhancement for Phone Recordings
            ### Optimize poor quality recordings for transcription
            
            **Optimized for:**
            - Poor quality phone recordings
            - Elderly speakers
            - Cantonese speech
            - SenseVoice transcription model
            
            **Enhancement Pipeline:**
            1. High-pass filter (remove low rumble)
            2. Low-pass filter (remove high hiss)
            3. Spectral noise reduction
            4. Speech frequency enhancement (300-3400 Hz)
            5. Dynamic range compression (helps with volume variations)
            6. Normalization
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                
                gr.Markdown("### Options")
                
                transcribe_checkbox = gr.Checkbox(
                    label="Transcribe Enhanced Audio",
                    value=False,
                    info="Automatically transcribe after enhancement using SenseVoice"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=["auto", "zh", "en", "yue", "ja", "ko"],
                    value="yue",
                    label="Transcription Language",
                    info="Select language for transcription (yue = Cantonese)",
                    visible=True
                )
                
                enhance_btn = gr.Button(
                    "🚀 Enhance Audio",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown(
                    """
                    ### Tips for Best Results:
                    - Works with WAV, MP3, FLAC, M4A files
                    - Processing time: ~10-30 seconds
                    - Enhanced audio is optimized for 16kHz (SenseVoice)
                    - Transcription requires funasr package
                    """
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                
                status_output = gr.Textbox(
                    label="Processing Status",
                    lines=25,
                    max_lines=35,
                    interactive=False
                )
                
                gr.Markdown("### Enhanced Audio")
                audio_output = gr.Audio(
                    label="Download Enhanced Audio",
                    type="filepath",
                    interactive=False
                )
                
                transcription_output = gr.Textbox(
                    label="Transcription (if enabled)",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
        
        # Connect button
        enhance_btn.click(
            fn=process_enhancement,
            inputs=[audio_input, transcribe_checkbox, language_dropdown],
            outputs=[audio_output, transcription_output, status_output]
        )
        
        # Show/hide language dropdown based on transcribe checkbox
        transcribe_checkbox.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[transcribe_checkbox],
            outputs=[language_dropdown]
        )
        
        gr.Markdown(
            """
            ---
            ### 📚 Technical Details:
            
            **Enhancement Techniques:**
            - **High-pass filter (80 Hz)**: Removes low-frequency rumble and handling noise
            - **Low-pass filter (8 kHz)**: Removes high-frequency hiss while preserving speech
            - **Spectral noise reduction**: Uses advanced algorithms to reduce background noise
            - **Speech enhancement**: Boosts frequencies in the 300-3400 Hz range (typical phone bandwidth)
            - **Dynamic range compression**: Makes quiet parts louder, reducing volume variations
            - **Normalization**: Ensures consistent volume level (-20 dBFS target)
            
            **Why This Helps:**
            - Elderly speakers often have lower volume or varying speech patterns
            - Phone recordings have limited bandwidth and background noise
            - Cantonese tones are preserved while noise is reduced
            - SenseVoice performs better with clean, normalized audio
            """
        )
    
    return demo


if __name__ == "__main__":
    print("🚀 Starting Audio Enhancement GUI...")
    print("📝 Optimized for phone recordings with elderly Cantonese speakers")
    print("=" * 60)
    
    demo = create_enhancement_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )

