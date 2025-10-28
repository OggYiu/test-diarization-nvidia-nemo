"""
Tab 2: Audio Chopper
Chop audio files into speaker segments based on RTTM
"""

import os
import time
import tempfile
import zipfile
import traceback
import gradio as gr

from audio_chopper import chop_audio_file, read_rttm_file


def process_audio_chopping(audio_file, rttm_file, rttm_text):
    """
    Process audio file and RTTM file to chop audio into segments.
    
    Args:
        audio_file: Audio file from Gradio interface
        rttm_file: RTTM file from Gradio interface
        rttm_text: RTTM text string pasted by user
    
    Returns:
        tuple: (zip_file_path, status_message, segment_details, output_path)
    """
    padding_ms = 100
    
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
            return None, "❌ No segments found in RTTM file!", "", ""
        
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
        # Filter out the temp_rttm.rttm file if it exists
        output_files = [f for f in output_files if f != "temp_rttm.rttm"]
        
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


def create_chopper_tab():
    """Create and return the Audio Chopper tab"""
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

