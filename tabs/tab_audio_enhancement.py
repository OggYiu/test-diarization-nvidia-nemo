"""
Tab 6: Audio Enhancement
Enhance audio quality for better transcription
"""

import os
import tempfile
import traceback
import gradio as gr
import torchaudio

from audio_enhancement import AudioEnhancer, transcribe_enhanced_audio


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


def create_audio_enhancement_tab():
    """Create and return the Audio Enhancement tab"""
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

