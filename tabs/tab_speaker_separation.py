"""
Tab 5: Speaker Separation
Separate speakers into individual audio tracks
"""

import os
import time
import tempfile
import traceback
import torch
import gradio as gr

from speaker_separation import perform_diarization


def create_speaker_audio_separation(audio_tensor, sample_rate, segments, target_speaker):
    """
    Create audio with only the target speaker, silencing others.
    
    Args:
        audio_tensor: torch.Tensor - audio waveform [channels, samples]
        sample_rate: int - sample rate of audio
        segments: list - speaker segments from RTTM
        target_speaker: str - speaker to keep (e.g., 'speaker_0')
    
    Returns:
        torch.Tensor - modified audio with only target speaker
    """
    # Clone the audio tensor
    output_audio = audio_tensor.clone()
    
    # Get total duration
    total_samples = audio_tensor.shape[-1]
    
    # Create a mask for silence (1 = keep, 0 = silence)
    mask = torch.zeros(total_samples)
    
    # Mark regions where target speaker is talking
    for segment in segments:
        if segment['speaker'] == target_speaker:
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            # Ensure we don't exceed boundaries
            start_sample = max(0, start_sample)
            end_sample = min(total_samples, end_sample)
            mask[start_sample:end_sample] = 1.0
    
    # Apply mask to audio
    output_audio = output_audio * mask.unsqueeze(0)
    
    return output_audio


def parse_rttm_for_separation(rttm_path):
    """Parse RTTM file to extract speaker segments."""
    segments = []
    
    with open(rttm_path, 'r') as f:
        for line in f:
            if line.startswith('SPEAKER'):
                parts = line.strip().split()
                speaker = parts[7]
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                
                segments.append({
                    'speaker': speaker,
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })
    
    return segments


def process_speaker_separation(audio_file, progress=gr.Progress()):
    """
    Perform speaker separation on audio file.
    Reuses logic from speaker_separation.py for reliability.
    
    Args:
        audio_file: Audio file from Gradio interface
    
    Returns:
        tuple: (speaker_0_file, speaker_1_file, status_message)
    """
    if audio_file is None:
        return None, None, "[ERROR] No audio file uploaded"
    
    try:
        import soundfile as sf
        
        # Create a temporary output directory
        temp_out_dir = tempfile.mkdtemp(prefix="speaker_separated_")
        
        status = "=" * 60 + "\n"
        status += "SPEAKER SEPARATION TOOL\n"
        status += "=" * 60 + "\n"
        status += f"Input audio: {os.path.basename(audio_file)}\n"
        status += f"Output directory: {temp_out_dir}\n\n"
        
        overall_start_time = time.time()
        
        # ================================================================
        # STEP 1: Perform diarization
        # ================================================================
        progress(0.1, desc="Starting diarization...")
        
        status += "=" * 60 + "\n"
        status += "STEP 1: Speaker Diarization with NVIDIA NeMo\n"
        status += "=" * 60 + "\n\n"
        
        try:
            # Use the perform_diarization function from speaker_separation module
            rttm_file = perform_diarization(audio_file, temp_out_dir)
            status += f"[OK] Diarization complete!\n"
            status += f"[OK] RTTM file generated: {os.path.basename(rttm_file)}\n\n"
        except Exception as e:
            error_msg = f"[ERROR] Error during diarization: {str(e)}\n\n{traceback.format_exc()}"
            return None, None, status + error_msg
        
        # ================================================================
        # STEP 2: Separate speakers
        # ================================================================
        progress(0.4, desc="Loading audio...")
        
        status += "=" * 60 + "\n"
        status += "STEP 2: Speaker Separation\n"
        status += "=" * 60 + "\n\n"
        
        # Load audio using soundfile
        status += f"[*] Loading audio: {os.path.basename(audio_file)}\n"
        try:
            audio_data, sample_rate = sf.read(str(audio_file), dtype='float32')
            
            # Convert to torch tensor and ensure correct shape [channels, samples]
            if audio_data.ndim == 1:
                # Mono audio
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                # Stereo or multi-channel: transpose to [channels, samples]
                waveform = torch.from_numpy(audio_data.T)
            
            status += f"[OK] Audio loaded: shape={waveform.shape}, sample_rate={sample_rate} Hz\n"
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                status += "[*] Converting stereo to mono...\n"
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                status += "[OK] Converted to mono\n"
            
            status += "\n"
            
        except Exception as e:
            error_msg = f"[ERROR] Error loading audio: {str(e)}\n{traceback.format_exc()}"
            return None, None, status + error_msg
        
        progress(0.5, desc="Parsing diarization results...")
        
        # Parse RTTM file
        status += f"[*] Parsing diarization results from: {os.path.basename(rttm_file)}\n"
        try:
            segments = parse_rttm_for_separation(rttm_file)
            
            if not segments:
                error_msg = "[ERROR] No speaker segments found in RTTM file!"
                return None, None, status + error_msg
            
            # Get unique speakers
            speakers = sorted(set(seg['speaker'] for seg in segments))
            status += f"[OK] Found {len(speakers)} speaker(s): {', '.join(speakers)}\n\n"
            
            # Print segment summary
            status += "[*] Segment Summary:\n"
            for speaker in speakers:
                speaker_segments = [s for s in segments if s['speaker'] == speaker]
                total_duration = sum(s['duration'] for s in speaker_segments)
                status += f"  {speaker}: {len(speaker_segments)} segments, {total_duration:.2f}s total\n"
            status += "\n"
            
        except Exception as e:
            error_msg = f"[ERROR] Error parsing RTTM: {str(e)}\n{traceback.format_exc()}"
            return None, None, status + error_msg
        
        progress(0.6, desc="Creating separated audio files...")
        
        # Separate each speaker
        status += "[*] Creating separated audio files...\n\n"
        speaker_files = []
        
        for i, speaker in enumerate(speakers):
            try:
                progress(0.6 + (0.3 * i / len(speakers)), desc=f"Processing speaker {i}...")
                
                status += f"  Processing speaker_{i} ({speaker})...\n"
                
                # Create audio with only this speaker
                speaker_audio = create_speaker_audio_separation(waveform, sample_rate, segments, speaker)
                
                # Save to file using soundfile
                output_file = os.path.join(temp_out_dir, f"speaker_{i}_only.wav")
                
                # Convert tensor to numpy
                audio_numpy = speaker_audio.numpy()
                
                # Ensure audio_numpy is in the correct format for soundfile
                # soundfile expects [samples] for mono or [samples, channels] for multi-channel
                if audio_numpy.ndim == 2:
                    # If shape is [1, samples], transpose to [samples, 1]
                    # If shape is [channels, samples], transpose to [samples, channels]
                    if audio_numpy.shape[0] <= 2:  # Likely [channels, samples] format
                        audio_numpy = audio_numpy.T
                
                # If we have [samples, 1] format, convert to [samples] for mono
                if audio_numpy.ndim == 2 and audio_numpy.shape[1] == 1:
                    audio_numpy = audio_numpy.squeeze()
                
                status += f"    - Audio shape for saving: {audio_numpy.shape}\n"
                sf.write(str(output_file), audio_numpy, sample_rate)
                
                speaker_files.append(output_file)
                
                # Calculate non-silent duration
                speaker_segments = [s for s in segments if s['speaker'] == speaker]
                active_duration = sum(s['duration'] for s in speaker_segments)
                total_duration = waveform.shape[-1] / sample_rate
                
                status += f"  [OK] Saved: speaker_{i}_only.wav\n"
                status += f"    - Total duration: {total_duration:.2f}s\n"
                status += f"    - Active speech: {active_duration:.2f}s ({active_duration/total_duration*100:.1f}%)\n"
                status += f"    - Silenced: {total_duration - active_duration:.2f}s\n\n"
                
            except Exception as e:
                status += f"  [ERROR] Error processing speaker_{i}: {str(e)}\n"
                status += f"  {traceback.format_exc()}\n\n"
                # Continue to next speaker instead of stopping completely
                continue
        
        # ================================================================
        # Final summary
        # ================================================================
        progress(1.0, desc="Complete!")
        
        overall_end_time = time.time()
        processing_time = overall_end_time - overall_start_time
        
        status += "=" * 60 + "\n"
        status += "[SUCCESS] COMPLETE!\n"
        status += "=" * 60 + "\n"
        status += f"[*] Total processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)\n"
        status += f"[*] Separated files created: {len(speaker_files)}\n\n"
        
        status += "[*] Files created:\n"
        for speaker_file in speaker_files:
            file_size = os.path.getsize(speaker_file)
            status += f"  - {os.path.basename(speaker_file)} ({file_size:,} bytes)\n"
        
        # Return files (up to 2 speakers for now)
        speaker_0 = speaker_files[0] if len(speaker_files) > 0 else None
        speaker_1 = speaker_files[1] if len(speaker_files) > 1 else None
        
        # Add note if more than 2 speakers
        if len(speaker_files) > 2:
            status += f"\n[WARNING] Note: {len(speaker_files)} speakers detected. Only first 2 available for download in GUI.\n"
            status += f"   All files saved in: {temp_out_dir}\n"
        
        status += "\n"
        
        return speaker_0, speaker_1, status
        
    except Exception as e:
        error_msg = f"[ERROR] Error during speaker separation: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, error_msg


def create_speaker_separation_tab():
    """Create and return the Speaker Separation tab"""
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
                sep_process_btn = gr.Button("Separate Speakers", variant="primary", size="lg")
                
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

