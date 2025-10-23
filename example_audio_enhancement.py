#!/usr/bin/env python
# coding=utf-8
"""
Example script demonstrating audio enhancement usage
"""

from audio_enhancement import AudioEnhancer, transcribe_enhanced_audio
from pathlib import Path

def example_basic_enhancement():
    """Example 1: Basic audio enhancement"""
    print("="*60)
    print("Example 1: Basic Audio Enhancement")
    print("="*60)
    
    # Create enhancer
    enhancer = AudioEnhancer(target_sr=16000)
    
    # Enhance audio
    input_file = "demo/phone_recordings/test.wav"
    if not Path(input_file).exists():
        print(f"⚠️  Input file not found: {input_file}")
        print("Please provide a valid audio file path")
        return
    
    enhanced_path = enhancer.enhance(
        audio_path=input_file,
        output_path="demo/phone_recordings/test_enhanced.wav",
        verbose=True
    )
    
    print(f"\n✅ Enhanced audio saved to: {enhanced_path}")


def example_enhance_and_transcribe():
    """Example 2: Enhance and transcribe"""
    print("\n" + "="*60)
    print("Example 2: Enhance and Transcribe")
    print("="*60)
    
    # Create enhancer
    enhancer = AudioEnhancer(target_sr=16000)
    
    # Enhance audio
    input_file = "demo/phone_recordings/test.wav"
    if not Path(input_file).exists():
        print(f"⚠️  Input file not found: {input_file}")
        print("Please provide a valid audio file path")
        return
    
    enhanced_path = enhancer.enhance(
        audio_path=input_file,
        output_path="demo/phone_recordings/test_enhanced.wav",
        verbose=False  # Less verbose for this example
    )
    
    print(f"✅ Enhanced audio: {enhanced_path}")
    
    # Transcribe the enhanced audio
    print("\n📝 Transcribing enhanced audio...")
    transcription = transcribe_enhanced_audio(
        audio_path=enhanced_path,
        language="yue"  # Cantonese
    )
    
    # Save transcription
    transcript_path = Path(enhanced_path).parent / "test_enhanced_transcript.txt"
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcription)
    
    print(f"✅ Transcription saved to: {transcript_path}")


def example_custom_pipeline():
    """Example 3: Custom enhancement pipeline"""
    print("\n" + "="*60)
    print("Example 3: Custom Enhancement Pipeline")
    print("="*60)
    
    import torchaudio
    
    # Create enhancer
    enhancer = AudioEnhancer(target_sr=16000)
    
    input_file = "demo/phone_recordings/test.wav"
    if not Path(input_file).exists():
        print(f"⚠️  Input file not found: {input_file}")
        return
    
    # Load audio
    print("Loading audio...")
    waveform, sr = enhancer.load_audio(input_file)
    
    # Apply only specific enhancements
    print("Applying custom enhancements...")
    
    # 1. Remove low-frequency noise
    waveform = enhancer.apply_highpass_filter(waveform, sr, cutoff=100)
    print("  ✓ High-pass filter (100 Hz)")
    
    # 2. Remove high-frequency hiss
    waveform = enhancer.apply_lowpass_filter(waveform, sr, cutoff=7000)
    print("  ✓ Low-pass filter (7000 Hz)")
    
    # 3. Enhance speech frequencies
    waveform = enhancer.apply_speech_enhancement(waveform, sr)
    print("  ✓ Speech enhancement")
    
    # 4. Normalize
    waveform = enhancer.normalize_audio(waveform, target_level=-18.0)
    print("  ✓ Normalized to -18 dBFS")
    
    # Save
    output_path = "demo/phone_recordings/test_custom_enhanced.wav"
    torchaudio.save(output_path, waveform, sr)
    print(f"\n✅ Custom enhanced audio saved to: {output_path}")


def example_batch_enhancement():
    """Example 4: Batch process multiple files"""
    print("\n" + "="*60)
    print("Example 4: Batch Enhancement")
    print("="*60)
    
    import os
    
    # Create enhancer
    enhancer = AudioEnhancer(target_sr=16000)
    
    # Input directory
    input_dir = Path("demo/phone_recordings")
    output_dir = Path("demo/phone_recordings/enhanced")
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"⚠️  Input directory not found: {input_dir}")
        return
    
    # Find all audio files
    audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3"))
    
    if not audio_files:
        print(f"⚠️  No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        output_path = output_dir / f"{audio_file.stem}_enhanced.wav"
        
        try:
            enhancer.enhance(
                audio_path=str(audio_file),
                output_path=str(output_path),
                verbose=False
            )
            print(f"  ✅ Saved: {output_path.name}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Batch processing complete!")
    print(f"Enhanced files saved to: {output_dir}")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         Audio Enhancement Examples                        ║
    ║  Optimized for phone recordings with elderly speakers     ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run examples
    # Uncomment the examples you want to run:
    
    # Example 1: Basic enhancement
    example_basic_enhancement()
    
    # Example 2: Enhance and transcribe
    # example_enhance_and_transcribe()
    
    # Example 3: Custom pipeline
    # example_custom_pipeline()
    
    # Example 4: Batch processing
    # example_batch_enhancement()
    
    print("\n" + "="*60)
    print("✅ Examples completed!")
    print("="*60)
    print("\nTips:")
    print("  • Uncomment other examples in the script to try them")
    print("  • Adjust parameters in audio_enhancement.py for your needs")
    print("  • Use the GUI for visual feedback: python audio_enhancement_gui.py")
    print("  • See AUDIO_ENHANCEMENT_README.md for full documentation")
    print()

