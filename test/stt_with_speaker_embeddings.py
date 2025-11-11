"""
Simple example: Using speaker embeddings to enhance STT transcription

This script demonstrates how to load speaker embeddings from diarization
and use them as context for speech-to-text transcription.
"""

import os
import pickle
import librosa
import torch
import soundfile as sf
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path


def load_speaker_embeddings(embedding_path):
    """
    Load speaker embeddings from diarization output
    
    Args:
        embedding_path: Path to the pickle file containing embeddings
        
    Returns:
        dict: Speaker embeddings dictionary
    """
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def parse_rttm_file(rttm_path):
    """
    Parse RTTM file to extract speaker segments
    
    RTTM format: SPEAKER filename channel start_time duration <NA> <NA> speaker_label <NA> <NA>
    
    Args:
        rttm_path: Path to RTTM file
        
    Returns:
        list: List of speaker segments [{start, duration, speaker, end}, ...]
    """
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_label = parts[7]
                segments.append({
                    'start': start_time,
                    'duration': duration,
                    'end': start_time + duration,
                    'speaker': speaker_label
                })
    return segments


def extract_audio_segment(audio_data, sample_rate, start_time, duration):
    """
    Extract a segment from audio data
    
    Args:
        audio_data: Audio numpy array
        sample_rate: Sample rate of the audio
        start_time: Start time in seconds
        duration: Duration in seconds
        
    Returns:
        numpy array: Audio segment
    """
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)
    
    # Ensure we don't go out of bounds
    start_sample = max(0, start_sample)
    end_sample = min(len(audio_data), end_sample)
    
    return audio_data[start_sample:end_sample]


def get_device_info():
    """Get available compute device (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 Using GPU: {device_name}")
    else:
        device = torch.device("cpu")
        print(f"💻 Using CPU")
    return device


def load_whisper_model(device):
    """
    Load Whisper-v3-Cantonese model
    
    Args:
        device: torch device (cuda or cpu)
        
    Returns:
        tuple: (model, processor)
    """
    print("🔄 Loading Whisper-v3-Cantonese model...")
    
    model = WhisperForConditionalGeneration.from_pretrained(
        "khleeloo/whisper-large-v3-cantonese"
    )
    processor = WhisperProcessor.from_pretrained(
        "khleeloo/whisper-large-v3-cantonese"
    )
    
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, processor


def analyze_speaker_embedding(speaker_id, embeddings):
    """
    Analyze speaker embedding characteristics
    
    This can help identify:
    - Voice characteristics (pitch, tone)
    - Gender estimation
    - Speaking style
    
    Args:
        speaker_id: Speaker identifier (e.g., "speaker_0")
        embeddings: Speaker embeddings dictionary
        
    Returns:
        dict: Analysis results
    """
    # Extract embedding for this speaker
    # Note: The actual structure depends on NeMo's output format
    print(f"\n📊 Analyzing {speaker_id} characteristics...")
    
    # Placeholder analysis (actual implementation would depend on embedding structure)
    analysis = {
        "speaker_id": speaker_id,
        "note": "Speaker embedding loaded - can be used for adaptation"
    }
    
    return analysis


def transcribe_with_speaker_context(audio_segment, speaker_id, segment_info, embeddings, model, processor, device):
    """
    Transcribe audio segment with speaker embedding context
    
    Note: Standard Whisper doesn't directly use speaker embeddings,
    but this demonstrates the concept. The embeddings could be used to:
    1. Select speaker-specific language models
    2. Adjust decoding parameters based on speaker characteristics
    3. Post-process based on speaker profile
    
    Args:
        audio_segment: Audio numpy array (already extracted)
        speaker_id: Speaker identifier
        segment_info: Dictionary with segment info (start, duration, etc.)
        embeddings: Speaker embeddings dictionary
        model: Whisper model
        processor: Whisper processor
        device: torch device
        
    Returns:
        str: Transcription text
    """
    start_time = segment_info['start']
    duration = segment_info['duration']
    
    # Skip very short segments
    if duration < 0.3:
        return ""
    
    print(f"   🎙️ [{start_time:.2f}s - {start_time+duration:.2f}s] {speaker_id}")
    
    # Analyze speaker characteristics (optional)
    # speaker_analysis = analyze_speaker_embedding(speaker_id, embeddings)
    
    # Process the audio (already at 16kHz from librosa.load)
    input_features = processor(audio_segment, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    # Generate transcription
    # Note: In a full implementation, you could use speaker_analysis to:
    # - Adjust generation parameters (temperature, beam_size, etc.)
    # - Select language variant based on accent
    # - Apply speaker-specific post-processing
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features, 
            language="yue",
            # Future: Could add speaker-specific parameters here
            # max_length=448,
            # num_beams=5,
        )
    
    # Decode the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    transcription_text = transcription[0].strip()
    
    if transcription_text:
        print(f"      📝 {transcription_text[:80]}{'...' if len(transcription_text) > 80 else ''}")
    
    return transcription_text


def main():
    """
    Example usage: Transcribe audio segments with speaker embeddings
    """
    print("="*60)
    print("🎯 STT with Speaker Embeddings - Example")
    print("="*60)
    
    # Setup paths
    base_dir = Path(__file__).parent
    embedding_path = base_dir / "diarization_output" / "speaker_outputs" / "embeddings" / "subsegments_scale1_embeddings.pkl"
    rttm_path = base_dir / "diarization_output" / "pred_rttms" / "test01.rttm"
    audio_path = base_dir / "test01.wav"
    
    # Check if files exist
    if not embedding_path.exists():
        print(f"❌ Embedding file not found: {embedding_path}")
        print("   Please run diarization first to generate embeddings.")
        return
    
    if not rttm_path.exists():
        print(f"❌ RTTM file not found: {rttm_path}")
        print("   Please run diarization first to generate RTTM file.")
        return
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    # Load speaker embeddings
    print(f"\n📂 Loading speaker embeddings from: {embedding_path.name}")
    embeddings = load_speaker_embeddings(embedding_path)
    print(f"✅ Embeddings loaded!")
    
    # Parse RTTM file to get speaker segments
    print(f"\n📂 Parsing RTTM file: {rttm_path.name}")
    segments = parse_rttm_file(rttm_path)
    print(f"✅ Found {len(segments)} speaker segments")
    
    # Load the full audio file
    print(f"\n📂 Loading audio file: {audio_path.name}")
    audio_data, sample_rate = librosa.load(str(audio_path), sr=16000)
    print(f"✅ Audio loaded: {len(audio_data)/sample_rate:.2f} seconds")
    
    # Get device
    device = get_device_info()
    
    # Load Whisper model
    model, processor = load_whisper_model(device)
    
    # Process each segment
    print("\n" + "="*60)
    print("📝 Starting transcription with speaker context...")
    print("="*60)
    
    results = []
    for idx, segment in enumerate(segments, 1):
        print(f"\n[{idx}/{len(segments)}]", end=" ")
        
        # Extract audio segment
        audio_segment = extract_audio_segment(
            audio_data, 
            sample_rate, 
            segment['start'], 
            segment['duration']
        )
        
        # Skip empty segments
        if len(audio_segment) == 0:
            print(f"   ⚠️ Empty segment, skipping...")
            continue
        
        # Transcribe with speaker context
        transcription = transcribe_with_speaker_context(
            audio_segment,
            segment['speaker'],
            segment,
            embeddings,
            model,
            processor,
            device
        )
        
        if transcription:  # Only add non-empty transcriptions
            results.append({
                "start": segment['start'],
                "end": segment['end'],
                "duration": segment['duration'],
                "speaker": segment['speaker'],
                "transcription": transcription
            })
    
    # Display results in chronological order with speaker labels
    print("\n" + "="*60)
    print("📊 Full Transcription with Speaker Labels")
    print("="*60)
    
    for result in results:
        speaker_icon = "👤" if result['speaker'] == "speaker_0" else "👥"
        print(f"\n{speaker_icon} [{result['start']:.2f}s - {result['end']:.2f}s] {result['speaker']}")
        print(f"   {result['transcription']}")
    
    # Display speaker-separated transcriptions
    print("\n" + "="*60)
    print("📋 Transcription by Speaker")
    print("="*60)
    
    for speaker in ['speaker_0', 'speaker_1']:
        speaker_results = [r for r in results if r['speaker'] == speaker]
        if speaker_results:
            speaker_icon = "👤" if speaker == "speaker_0" else "👥"
            print(f"\n{speaker_icon} {speaker.upper()} ({len(speaker_results)} segments):")
            print("-" * 60)
            full_text = " ".join([r['transcription'] for r in speaker_results])
            print(full_text)
    
    print("\n" + "="*60)
    print("✅ All done!")
    print("="*60)


if __name__ == "__main__":
    main()

