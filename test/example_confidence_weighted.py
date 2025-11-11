"""
Quick Example: Confidence-Weighted Transcription
=================================================

This script demonstrates how to use the ConfidenceWeightedTranscriber class
programmatically (as a library) instead of via command line.

Usage:
    python example_confidence_weighted.py
"""

import os
from pathlib import Path
from confidence_weighted_transcription import ConfidenceWeightedTranscriber, save_results


def main():
    """
    Example workflow showing how to use confidence-weighted transcription.
    """
    
    # Configuration
    audio_file = "test01.wav"  # Your audio file
    diarization_dir = "./diarization_output"
    output_dir = "./example_output"
    
    # Check if files exist
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        print("💡 Make sure test01.wav is in the agent/ directory")
        return 1
    
    audio_basename = Path(audio_file).stem
    rttm_path = os.path.join(diarization_dir, f"pred_rttms/{audio_basename}.rttm")
    vad_path = os.path.join(diarization_dir, f"vad_outputs/{audio_basename}.frame")
    
    if not os.path.exists(rttm_path):
        print(f"❌ RTTM file not found: {rttm_path}")
        print(f"💡 Run diarization first:")
        print(f"   python diarize.py {audio_file} --output {diarization_dir}")
        return 1
    
    if not os.path.exists(vad_path):
        print(f"❌ VAD file not found: {vad_path}")
        return 1
    
    print("✅ All required files found!\n")
    
    # Example 1: Basic usage with default settings
    print("="*80)
    print("Example 1: Default Settings")
    print("="*80 + "\n")
    
    transcriber = ConfidenceWeightedTranscriber(
        vad_threshold=0.5,
        min_combined_confidence=0.6
    )
    
    results = transcriber.process_audio_with_confidence(
        audio_path=audio_file,
        rttm_path=rttm_path,
        vad_path=vad_path,
        language="auto"
    )
    
    save_results(results, output_dir)
    
    # Example 2: Analyze results programmatically
    print("\n" + "="*80)
    print("Example 2: Analyzing Results")
    print("="*80 + "\n")
    
    high_quality = [r for r in results['results'] if r['quality'] == 'high']
    medium_quality = [r for r in results['results'] if r['quality'] == 'medium']
    low_quality = [r for r in results['results'] if r['quality'] == 'low']
    
    print("🎯 Quality Distribution:")
    print(f"   High:   {len(high_quality)} segments ({len(high_quality)/len(results['results'])*100:.1f}%)")
    print(f"   Medium: {len(medium_quality)} segments ({len(medium_quality)/len(results['results'])*100:.1f}%)")
    print(f"   Low:    {len(low_quality)} segments ({len(low_quality)/len(results['results'])*100:.1f}%)\n")
    
    # Example 3: Show high-quality conversation
    if high_quality:
        print("="*80)
        print("Example 3: High-Quality Conversation")
        print("="*80 + "\n")
        
        for segment in high_quality:
            print(f"[{segment['start']:.2f}s] {segment['speaker']}: {segment['transcription']}")
            print(f"  └─ Confidence: {segment['combined_confidence']:.2f}\n")
    
    # Example 4: Identify segments needing review
    needs_review = [r for r in results['results'] if r['quality'] in ['low', 'medium']]
    
    if needs_review:
        print("="*80)
        print("Example 4: Segments Needing Review")
        print("="*80 + "\n")
        
        for segment in needs_review[:3]:  # Show first 3
            print(f"⚠️  Time: {segment['start']:.2f}s - {segment['end']:.2f}s")
            print(f"   Speaker: {segment['speaker']}")
            print(f"   Quality: {segment['quality'].upper()}")
            print(f"   Combined Confidence: {segment['combined_confidence']:.2f}")
            print(f"   VAD Mean: {segment['vad_stats']['mean']:.2f}")
            print(f"   Speech Ratio: {segment['vad_stats']['speech_ratio']:.1%}")
            print(f"   Text: {segment['transcription']}\n")
    
    # Example 5: Custom filtering based on your criteria
    print("="*80)
    print("Example 5: Custom Filtering")
    print("="*80 + "\n")
    
    # Get only segments with very high confidence
    very_high_confidence = [
        r for r in results['results'] 
        if r['combined_confidence'] >= 0.75
    ]
    
    print(f"📊 Segments with combined confidence ≥ 0.75: {len(very_high_confidence)}")
    
    # Get segments by speaker
    speakers = {}
    for result in results['results']:
        speaker = result['speaker']
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(result)
    
    print(f"👥 Speakers found: {len(speakers)}")
    for speaker, segments in speakers.items():
        avg_conf = sum(s['combined_confidence'] for s in segments) / len(segments)
        print(f"   {speaker}: {len(segments)} segments, avg confidence: {avg_conf:.2f}")
    
    print("\n" + "="*80)
    print("✅ Examples Complete!")
    print("="*80)
    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"   - confidence_weighted_results.json    (full data)")
    print(f"   - high_quality_transcription.txt      (clean transcript)")
    print(f"   - segments_for_review.txt             (needs review)\n")
    
    return 0


if __name__ == "__main__":
    exit(main())

