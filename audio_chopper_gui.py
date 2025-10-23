import os
import gradio as gr
import tempfile
import shutil
import time
from pathlib import Path
import zipfile
from audio_chopper import chop_audio_file, read_rttm_file


def process_audio_chopping(audio_file, rttm_file):
    padding_ms = 500

    """
    Process audio file and RTTM file to chop audio into segments.
    
    Args:
        audio_file: Audio file from Gradio interface
        rttm_file: RTTM file from Gradio interface
    
    Returns:
        tuple: (zip_file_path, status_message, file_list)
    """
    if audio_file is None:
        return None, "❌ No audio file uploaded", ""
    
    if rttm_file is None:
        return None, "❌ No RTTM file uploaded", ""
    
    try:
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="chopped_audio_")
        
        status = f"🔄 Processing audio file: {os.path.basename(audio_file)}\n"
        status += f"📄 RTTM file: {os.path.basename(rttm_file)}\n"
        status += f"⏱️ Padding: {padding_ms} ms\n"
        status += f"📁 Output directory: {temp_out_dir}\n\n"
        
        start_time = time.time()
        
        # Read RTTM file to get segments
        status += "📖 Reading RTTM file...\n"
        segments = read_rttm_file(rttm_file)
        
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
        import traceback
        error_msg += f"\n\n{traceback.format_exc()}"
        return None, error_msg, ""


def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Audio Chopper", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # ✂️ Audio Chopper Tool

            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input Files")
                
                audio_input = gr.Audio(
                    label="Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                
                rttm_input = gr.File(
                    label="RTTM File",
                    file_types=[".rttm"],
                    file_count="single"
                )
                
                process_btn = gr.Button("✂️ Chop Audio", variant="primary", size="lg")
                
            
            with gr.Column(scale=2):
                gr.Markdown("### 📤 Results")
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=12,
                    max_lines=20,
                    interactive=False
                )
                
                segment_details = gr.Textbox(
                    label="Segment Details",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
                
                download_output = gr.File(
                    label="Download Chopped Segments (ZIP)",
                    interactive=False
                )
        
        # Connect the button to the processing function
        process_btn.click(
            fn=process_audio_chopping,
            inputs=[audio_input, rttm_input],
            outputs=[download_output, status_output, segment_details]
        )
    
    return demo


if __name__ == "__main__":
    print("🚀 Starting Audio Chopper GUI...")
    print("📝 This tool chops audio files based on RTTM diarization results.")
    print("=" * 60)
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )

