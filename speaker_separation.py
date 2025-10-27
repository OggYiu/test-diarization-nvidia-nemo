#!/usr/bin/env python
# coding=utf-8
"""
Speaker Separation Script
Uses NVIDIA NeMo for speaker diarization and creates separate audio files for each speaker.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf
import json
import tempfile
import shutil


def create_manifest(audio_file, manifest_path):
    """Create manifest file required by NeMo diarizer."""
    manifest_data = {
        "audio_filepath": str(audio_file),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": 2,
        "rttm_filepath": None,
        "uem_filepath": None
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f)
    
    print(f"[OK] Created manifest: {manifest_path}")


def create_diarization_config(manifest_path, output_dir):
    """Create configuration for NeMo diarizer (based on working diarization.py config)."""
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {
        'batch_size': 32,
        'sample_rate': 16000,
        'verbose': True,
        'device': device,
        'num_workers': 0,
        'diarizer': {
            'collar': 0.25,
            'ignore_overlap': True,
            'manifest_filepath': str(manifest_path),
            'out_dir': str(output_dir),
            'oracle_vad': False,
            'vad': {
                'model_path': 'vad_multilingual_marblenet',
                'batch_size': 32,
                'parameters': {
                    'window_length_in_sec': 0.63,
                    'shift_length_in_sec': 0.08,
                    'smoothing': False,
                    'overlap': 0.5,
                    'scale': 'absolute',
                    'onset': 0.7,
                    'offset': 0.4,
                    'pad_onset': 0.1,
                    'pad_offset': -0.05,
                    'min_duration_on': 0.1,
                    'min_duration_off': 0.3,
                    'filter_speech_first': True,
                    'normalize': False
                }
            },
            'speaker_embeddings': {
                'model_path': 'titanet_large',
                'batch_size': 32,
                'parameters': {
                    'window_length_in_sec': [1.5, 1.25, 1.0, 0.75, 0.5],
                    'shift_length_in_sec': [0.75, 0.625, 0.5, 0.375, 0.25],
                    'multiscale_weights': [1, 1, 1, 1, 1],
                    'save_embeddings': False
                }
            },
            'clustering': {
                'parameters': {
                    'oracle_num_speakers': False,
                    'max_num_speakers': 8,
                    'max_rp_threshold': 0.15,
                    'sparse_search_volume': 30
                }
            },
            'msdd_model': {
                'model_path': 'diar_msdd_telephonic',
                'parameters': {
                    'sigmoid_threshold': [0.7, 1.0],
                    'use_speaker_embed': True,
                    'use_clus_as_spk_embed': False,
                    'infer_batch_size': 25,
                    'seq_eval_mode': False,
                    'diar_window_length': 50,
                    'overlap_infer_spk_limit': 5,
                    'max_overlap_spk_num': None
                }
            }
        }
    }
    
    return OmegaConf.create(config)


def parse_rttm_file(rttm_path):
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


def create_speaker_audio(audio_tensor, sample_rate, segments, target_speaker):
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
    total_duration = total_samples / sample_rate
    
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


def perform_diarization(audio_file, output_dir):
    """
    Perform speaker diarization using NeMo.
    
    Args:
        audio_file: Path to audio file
        output_dir: Directory for output files
    
    Returns:
        Path to RTTM file with diarization results
    """
    print("\n" + "="*60)
    print("STEP 1: Speaker Diarization with NVIDIA NeMo")
    print("="*60)
    
    # Create temporary directory for diarization files
    temp_dir = Path(output_dir) / "temp_diarization"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifest file
    manifest_path = temp_dir / "input_manifest.json"
    create_manifest(audio_file, manifest_path)
    
    # Create diarization config
    print("\n[*] Configuring NeMo diarizer...")
    config = create_diarization_config(manifest_path, temp_dir)
    print(f"   Using device: {config.device}")
    
    # Initialize and run diarizer
    print("[*] Initializing NeMo diarizer (this may take a moment)...")
    try:
        diarizer = ClusteringDiarizer(cfg=config)
        print("[OK] Diarizer initialized successfully!")
        
        print("\n[*] Performing speaker diarization...")
        diarizer.diarize()
        print("[OK] Diarization complete!")
        
    except Exception as e:
        print(f"[ERROR] Error during diarization: {e}")
        raise
    
    # Find RTTM file
    rttm_files = list(temp_dir.glob("pred_rttms/*.rttm"))
    
    if not rttm_files:
        raise FileNotFoundError("No RTTM file generated by diarizer")
    
    rttm_file = rttm_files[0]
    print(f"[OK] RTTM file generated: {rttm_file}")
    
    return rttm_file


def separate_speakers(audio_file, rttm_file, output_dir):
    """
    Separate speakers into individual audio files.
    
    Args:
        audio_file: Path to original audio file
        rttm_file: Path to RTTM file with diarization results
        output_dir: Directory for output files
    """
    print("\n" + "="*60)
    print("STEP 2: Speaker Separation")
    print("="*60)
    
    # Load audio using soundfile (to avoid torchcodec dependency)
    print(f"\n[*] Loading audio: {audio_file}")
    audio_data, sample_rate = sf.read(str(audio_file), dtype='float32')
    
    # Convert to torch tensor and ensure correct shape [channels, samples]
    if audio_data.ndim == 1:
        # Mono audio
        waveform = torch.from_numpy(audio_data).unsqueeze(0)
    else:
        # Stereo or multi-channel: transpose to [channels, samples]
        waveform = torch.from_numpy(audio_data.T)
    
    print(f"[OK] Audio loaded: {waveform.shape}, sample rate: {sample_rate} Hz")
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        print("[*] Converting stereo to mono...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Parse RTTM file
    print(f"\n[*] Parsing diarization results from: {rttm_file}")
    segments = parse_rttm_file(rttm_file)
    
    if not segments:
        print("[ERROR] No speaker segments found in RTTM file!")
        return
    
    # Get unique speakers
    speakers = sorted(set(seg['speaker'] for seg in segments))
    print(f"[OK] Found {len(speakers)} speaker(s): {', '.join(speakers)}")
    
    # Print segment summary
    print("\n[*] Segment Summary:")
    for speaker in speakers:
        speaker_segments = [s for s in segments if s['speaker'] == speaker]
        total_duration = sum(s['duration'] for s in speaker_segments)
        print(f"  {speaker}: {len(speaker_segments)} segments, {total_duration:.2f}s total")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Separate each speaker
    print("\n[*] Creating separated audio files...")
    for i, speaker in enumerate(speakers):
        try:
            print(f"\n  Processing speaker_{i} ({speaker})...")
            
            # Create audio with only this speaker
            speaker_audio = create_speaker_audio(waveform, sample_rate, segments, speaker)
            
            # Save to file using soundfile (to avoid torchcodec dependency)
            output_file = output_path / f"speaker_{i}_only.wav"
            
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
            
            print(f"    - Audio shape for saving: {audio_numpy.shape}")
            sf.write(str(output_file), audio_numpy, sample_rate)
            
            # Calculate non-silent duration
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            active_duration = sum(s['duration'] for s in speaker_segments)
            total_duration = waveform.shape[-1] / sample_rate
            
            print(f"  [OK] Saved: {output_file}")
            print(f"    - Total duration: {total_duration:.2f}s")
            print(f"    - Active speech: {active_duration:.2f}s ({active_duration/total_duration*100:.1f}%)")
            print(f"    - Silenced: {total_duration - active_duration:.2f}s")
            
        except Exception as e:
            print(f"  [ERROR] Failed to create speaker_{i}_only.wav: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next speaker instead of stopping completely
            continue
    
    return speakers


def main():
    parser = argparse.ArgumentParser(
        description="Separate speakers in audio file using NVIDIA NeMo diarization"
    )
    parser.add_argument(
        "audio_file",
        nargs='?',
        type=str,
        default="demo/phone_recordings/test.wav",
        help="Path to audio file (default: demo/phone_recordings/test.wav)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="demo/speaker_separated",
        help="Output directory for separated audio files (default: demo/speaker_separated)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        sys.exit(1)
    
    print("="*60)
    print("SPEAKER SEPARATION TOOL")
    print("="*60)
    print(f"Input audio: {audio_path}")
    print(f"Output directory: {args.output}")
    
    try:
        # Step 1: Perform diarization
        rttm_file = perform_diarization(audio_path, args.output)
        
        # Step 2: Separate speakers
        speakers = separate_speakers(audio_path, rttm_file, args.output)
        
        # Success message
        print("\n" + "="*60)
        print("[SUCCESS] COMPLETE!")
        print("="*60)
        print(f"Separated audio files saved to: {args.output}")
        print("\nFiles created:")
        for i in range(len(speakers) if speakers else 2):
            output_file = Path(args.output) / f"speaker_{i}_only.wav"
            if output_file.exists():
                print(f"  - {output_file}")
        print()
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

