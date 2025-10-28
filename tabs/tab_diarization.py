"""
Tab 1: Speaker Diarization
Identify and separate speakers in audio files
"""

import os
import time
import tempfile
from datetime import datetime
import gradio as gr

from diarization import diarize_audio
from mongodb_utils import load_from_mongodb, save_to_mongodb


# MongoDB collection name for diarization results
COLLECTION_NAME = "diarization_results"


def load_diarization_cache():
    """
    Load cached diarization results from MongoDB.
    
    Returns:
        dict: Dictionary mapping filename to cached results
    """
    cache = {}
    
    # Load all documents from MongoDB
    documents = load_from_mongodb(COLLECTION_NAME)
    
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
    save_to_mongodb(COLLECTION_NAME, document, unique_key='filename')


def process_audio(audio_file, overwrite=False):
    """
    Process audio file through diarization and return results.
    Checks cache first to avoid reprocessing.
    
    Args:
        audio_file: Audio file from Gradio interface
        overwrite: If True, reprocess even if cached results exist
    
    Returns:
        tuple: (rttm_content, status_message, output_directory_path)
    """
    if audio_file is None:
        return "Please upload an audio file.", "❌ No file uploaded", ""
    
    try:
        filename = os.path.basename(audio_file)
        
        # Check if this file has been processed before (unless overwrite is True)
        cache = load_diarization_cache()
        if filename in cache and not overwrite:
            cached = cache[filename]
            
            # Create status message from cached data
            status = f"💾 Loading cached results from MongoDB for: {filename}\n"
            status += f"📅 Previously processed: {cached['timestamp']}\n\n"
            status += "✅ Results loaded from MongoDB cache!"
            
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
        if overwrite and filename in cache:
            status += f"♻️ Overwrite mode: Reprocessing despite existing cache\n"
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
        summary += f"  • 💾 Results saved to MongoDB for future use\n\n"
        
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


def create_diarization_tab():
    """Create and return the Speaker Diarization tab"""
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
                diar_overwrite_checkbox = gr.Checkbox(
                    label="🔄 Overwrite existing cached RTTM data",
                    value=False,
                    info="If checked, will reprocess the file even if MongoDB cached results exist"
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
            inputs=[diar_audio_input, diar_overwrite_checkbox],
            outputs=[diar_rttm_output, diar_status_output, diar_output_dir]
        )

