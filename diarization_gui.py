import os
import gradio as gr
import tempfile
import shutil
import time
from diarization import diarize_audio


def process_audio(audio_file):
    """
    Process audio file through diarization and return results.
    
    Args:
        audio_file: Audio file from Gradio interface
        num_speakers: Number of speakers in the audio
    
    Returns:
        tuple: (rttm_content, output_directory_path, status_message)
    """
    if audio_file is None:
        return "Please upload an audio file.", "", "❌ No file uploaded"
    
    try:
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="diarization_")
        
        # Run diarization with timing
        status = f"🔄 Processing audio file: {os.path.basename(audio_file)}\n"
        # status += f"📊 Number of speakers: {num_speakers}\n"
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


def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Speaker Diarization", theme=gr.themes.Soft()) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                
                # num_speakers = gr.Slider(
                #     minimum=1,
                #     maximum=10,
                #     value=2,
                #     step=1,
                #     label="Number of Speakers",
                #     info="Expected number of speakers in the audio"
                # )
                
                process_btn = gr.Button("🚀 Start Diarization", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=8,
                    max_lines=15,
                    interactive=False
                )
                
                rttm_output = gr.Textbox(
                    label="RTTM Output (Rich Time-Marked Text)",
                    lines=15,
                    max_lines=30,
                    interactive=False,
                    show_copy_button=True
                )
                
                output_dir = gr.Textbox(
                    label="Output Directory Path",
                    interactive=False,
                    visible=True
                )
        
        
        # Connect the button to the processing function
        process_btn.click(
            fn=process_audio,
            # inputs=[audio_input, num_speakers],
            inputs=[audio_input],
            outputs=[rttm_output, status_output, output_dir]
        )
    
    return demo


if __name__ == "__main__":
    print("🚀 Starting Speaker Diarization GUI...")
    print("📝 Make sure you have NeMo and all dependencies installed.")
    print("=" * 60)
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

