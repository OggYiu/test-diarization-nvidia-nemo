import pickle
import numpy as np
from pydub import AudioSegment

def enhance_audio_with_vad(audio_path, vad_frame_path, threshold=0.5):
    """
    Use VAD scores to preprocess audio before STT
    - Remove/attenuate low-confidence frames
    - Focus STT on high-quality speech regions
    """
    # Load VAD scores
    with open(vad_frame_path, 'r') as f:
        vad_scores = [float(line.strip()) for line in f]
    
    # Load audio
    audio = AudioSegment.from_wav(audio_path)
    
    # Each frame is 10ms
    frame_duration_ms = 10
    
    # Create mask for high-confidence speech
    enhanced_segments = []
    for i, score in enumerate(vad_scores):
        if score > threshold:
            start_ms = i * frame_duration_ms
            end_ms = (i + 1) * frame_duration_ms
            enhanced_segments.append((start_ms, end_ms))
    
    # Merge consecutive segments
    merged = []
    if enhanced_segments:
        current_start, current_end = enhanced_segments[0]
        for start, end in enhanced_segments[1:]:
            if start <= current_end:  # Consecutive or overlapping
                current_end = end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
    
    return merged, vad_scores


def main():
    """Simple test program for VAD-guided audio preprocessing"""
    
    # Paths relative to this script location
    audio_path = "../test01.wav"
    vad_frame_path = "../diarization_output/vad_outputs/test01.frame"
    output_path = "../test01_enhanced.wav"
    
    print(f"Testing VAD-guided audio preprocessing")
    print(f"Audio file: {audio_path}")
    print(f"VAD frame file: {vad_frame_path}")
    print("-" * 60)
    
    # Run the VAD enhancement
    merged_segments, vad_scores = enhance_audio_with_vad(
        audio_path, 
        vad_frame_path, 
        threshold=0.5
    )
    
    # Print statistics
    print(f"\nVAD Analysis Results:")
    print(f"Total VAD frames: {len(vad_scores)}")
    print(f"Total audio duration: {len(vad_scores) * 10 / 1000:.2f} seconds")
    print(f"Number of speech segments detected: {len(merged_segments)}")
    
    # Calculate total speech time
    total_speech_ms = sum(end - start for start, end in merged_segments)
    print(f"Total speech duration: {total_speech_ms / 1000:.2f} seconds")
    print(f"Speech percentage: {total_speech_ms / (len(vad_scores) * 10) * 100:.2f}%")
    
    # Show VAD score statistics
    vad_array = np.array(vad_scores)
    print(f"\nVAD Score Statistics:")
    print(f"Min: {vad_array.min():.4f}")
    print(f"Max: {vad_array.max():.4f}")
    print(f"Mean: {vad_array.mean():.4f}")
    print(f"Median: {np.median(vad_array):.4f}")
    
    # Show all segments
    print(f"\nAll speech segments ({len(merged_segments)} total):")
    for i, (start, end) in enumerate(merged_segments):
        print(f"  Segment {i+1}: {start/1000:.2f}s - {end/1000:.2f}s (duration: {(end-start)/1000:.2f}s)")
    
    # Optionally export the enhanced audio (only speech segments)
    try:
        audio = AudioSegment.from_wav(audio_path)
        enhanced_audio = AudioSegment.empty()
        
        for start_ms, end_ms in merged_segments:
            segment = audio[start_ms:end_ms]
            enhanced_audio += segment
        
        enhanced_audio.export(output_path, format="wav")
        print(f"\n✓ Enhanced audio exported to: {output_path}")
        print(f"  Original duration: {len(audio)/1000:.2f}s")
        print(f"  Enhanced duration: {len(enhanced_audio)/1000:.2f}s")
    except Exception as e:
        print(f"\nNote: Could not export enhanced audio: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")


if __name__ == "__main__":
    main()