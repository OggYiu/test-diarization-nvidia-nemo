"""
Confidence-Weighted Transcription
==================================

This script demonstrates how to use diarization output (RTTM + VAD scores) 
to improve STT quality by:
1. Combining VAD confidence with STT confidence
2. Filtering low-quality segments
3. Providing quality metrics for each transcription
4. Flagging segments that need manual review

Usage:
    python confidence_weighted_transcription.py audio.wav
    python confidence_weighted_transcription.py audio.wav --diarization-dir ./diarization_output
    python confidence_weighted_transcription.py audio.wav --vad-threshold 0.5 --min-confidence 0.6
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

# Audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    print("⚠️  pydub not available. Install with: pip install pydub")
    PYDUB_AVAILABLE = False

# STT Models
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    print("⚠️  funasr not available. Install with: pip install funasr")
    FUNASR_AVAILABLE = False


class ConfidenceWeightedTranscriber:
    """
    Transcriber that combines VAD confidence scores with STT for better quality control.
    """
    
    def __init__(self, 
                 vad_threshold: float = 0.5,
                 min_combined_confidence: float = 0.6,
                 stt_model_name: str = "iic/SenseVoiceSmall"):
        """
        Initialize the transcriber.
        
        Args:
            vad_threshold: Minimum VAD score to consider as speech (0-1)
            min_combined_confidence: Minimum combined confidence for valid transcription
            stt_model_name: Name of the STT model to use
        """
        self.vad_threshold = vad_threshold
        self.min_combined_confidence = min_combined_confidence
        self.stt_model_name = stt_model_name
        self.stt_model = None
        
    def load_stt_model(self):
        """Load the STT model (lazy loading)."""
        if not FUNASR_AVAILABLE:
            raise RuntimeError("funasr is not available. Please install it.")
            
        if self.stt_model is None:
            print(f"📥 Loading STT model: {self.stt_model_name}...")
            self.stt_model = AutoModel(
                model=self.stt_model_name,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                trust_remote_code=False,
                disable_update=True,
            )
            print("✅ STT model loaded\n")
    
    def load_rttm_segments(self, rttm_path: str) -> List[Dict]:
        """
        Load speaker segments from RTTM file.
        
        RTTM format:
        SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
        
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        with open(rttm_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 8 and parts[0] == 'SPEAKER':
                    segments.append({
                        'file': parts[1],
                        'start': float(parts[3]),
                        'duration': float(parts[4]),
                        'speaker': parts[7],
                        'end': float(parts[3]) + float(parts[4])
                    })
        
        return segments
    
    def load_vad_scores(self, vad_path: str) -> np.ndarray:
        """
        Load VAD scores from frame file.
        
        Each line contains a confidence score (0-1).
        Each frame represents 10ms of audio.
        
        Returns:
            Numpy array of VAD scores
        """
        scores = []
        with open(vad_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    scores.append(float(line))
        
        return np.array(scores)
    
    def calculate_segment_vad_confidence(self, 
                                         segment: Dict, 
                                         vad_scores: np.ndarray,
                                         frame_duration_ms: float = 10.0) -> Dict:
        """
        Calculate VAD confidence statistics for a segment.
        
        Args:
            segment: Segment dictionary with 'start' and 'duration'
            vad_scores: Array of VAD confidence scores
            frame_duration_ms: Duration of each VAD frame in milliseconds
            
        Returns:
            Dictionary with VAD statistics
        """
        # Convert segment time to frame indices
        start_frame = int(segment['start'] * 1000 / frame_duration_ms)
        end_frame = int((segment['start'] + segment['duration']) * 1000 / frame_duration_ms)
        
        # Ensure we don't go out of bounds
        start_frame = max(0, start_frame)
        end_frame = min(len(vad_scores), end_frame)
        
        if start_frame >= end_frame or start_frame >= len(vad_scores):
            return {
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0,
                'speech_ratio': 0.0,
                'frame_count': 0
            }
        
        segment_scores = vad_scores[start_frame:end_frame]
        
        # Calculate speech ratio (frames above threshold)
        speech_frames = np.sum(segment_scores > self.vad_threshold)
        speech_ratio = speech_frames / len(segment_scores) if len(segment_scores) > 0 else 0.0
        
        return {
            'mean': float(np.mean(segment_scores)),
            'median': float(np.median(segment_scores)),
            'min': float(np.min(segment_scores)),
            'max': float(np.max(segment_scores)),
            'std': float(np.std(segment_scores)),
            'speech_ratio': float(speech_ratio),
            'frame_count': len(segment_scores)
        }
    
    def extract_audio_segment(self, audio_path: str, start_sec: float, duration_sec: float) -> Optional[str]:
        """
        Extract audio segment and save to temporary file.
        
        Returns:
            Path to extracted segment file, or None if failed
        """
        if not PYDUB_AVAILABLE:
            return None
            
        try:
            audio = AudioSegment.from_wav(audio_path)
            start_ms = int(start_sec * 1000)
            end_ms = int((start_sec + duration_sec) * 1000)
            
            segment = audio[start_ms:end_ms]
            
            # Save to temp file
            temp_dir = Path(audio_path).parent / "temp_segments"
            temp_dir.mkdir(exist_ok=True)
            
            segment_path = temp_dir / f"segment_{start_sec:.3f}_{duration_sec:.3f}.wav"
            segment.export(str(segment_path), format="wav")
            
            return str(segment_path)
        except Exception as e:
            print(f"❌ Error extracting segment: {e}")
            return None
    
    def transcribe_segment(self, audio_path: str, language: str = "auto") -> Dict:
        """
        Transcribe an audio segment and return confidence scores.
        
        Returns:
            Dictionary with transcription and confidence
        """
        self.load_stt_model()
        
        try:
            result = self.stt_model.generate(
                input=audio_path,
                cache={},
                language=language,
                use_itn=True,
                batch_size_s=60,
            )
            
            if result and len(result) > 0:
                text = result[0].get("text", "")
                
                # Try to extract confidence if available
                # Note: Not all models provide confidence scores
                confidence = result[0].get("confidence", 0.8)  # Default if not available
                
                return {
                    'text': text,
                    'stt_confidence': confidence,
                    'success': True
                }
            else:
                return {
                    'text': "",
                    'stt_confidence': 0.0,
                    'success': False
                }
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return {
                'text': "",
                'stt_confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def calculate_combined_confidence(self, vad_confidence: float, stt_confidence: float) -> float:
        """
        Combine VAD and STT confidence scores.
        
        Uses weighted geometric mean to ensure both scores are reasonably high.
        
        Args:
            vad_confidence: VAD confidence (0-1)
            stt_confidence: STT model confidence (0-1)
            
        Returns:
            Combined confidence score (0-1)
        """
        # Weighted geometric mean (gives more weight to VAD)
        # This ensures both scores contribute, and low scores pull down the result
        vad_weight = 0.4
        stt_weight = 0.6
        
        combined = (vad_confidence ** vad_weight) * (stt_confidence ** stt_weight)
        
        return combined
    
    def process_audio_with_confidence(self, 
                                     audio_path: str,
                                     rttm_path: str,
                                     vad_path: str,
                                     language: str = "auto") -> Dict:
        """
        Main processing function that combines diarization with confidence-weighted transcription.
        
        Args:
            audio_path: Path to audio file
            rttm_path: Path to RTTM file with speaker segments
            vad_path: Path to VAD scores file
            language: Language for transcription
            
        Returns:
            Dictionary with results and statistics
        """
        print(f"\n{'='*80}")
        print("🎯 Confidence-Weighted Transcription")
        print(f"{'='*80}\n")
        
        # Load diarization data
        print("📥 Loading diarization data...")
        segments = self.load_rttm_segments(rttm_path)
        vad_scores = self.load_vad_scores(vad_path)
        
        print(f"  ✓ Loaded {len(segments)} segments")
        print(f"  ✓ Loaded {len(vad_scores)} VAD frames ({len(vad_scores) * 0.01:.2f}s)\n")
        
        # Process each segment
        results = []
        high_quality_count = 0
        low_quality_count = 0
        skipped_count = 0
        
        start_time = time.time()
        
        for i, segment in enumerate(segments):
            print(f"[{i+1}/{len(segments)}] Processing segment at {segment['start']:.2f}s...")
            
            # Calculate VAD confidence for this segment
            vad_stats = self.calculate_segment_vad_confidence(segment, vad_scores)
            
            # Skip segments with very low speech ratio
            if vad_stats['speech_ratio'] < 0.3:
                print(f"  ⏭️  Skipped (low speech ratio: {vad_stats['speech_ratio']:.2%})\n")
                skipped_count += 1
                results.append({
                    **segment,
                    'vad_stats': vad_stats,
                    'transcription': "",
                    'stt_confidence': 0.0,
                    'combined_confidence': 0.0,
                    'quality': 'skipped',
                    'reason': 'low_speech_ratio'
                })
                continue
            
            # Extract and transcribe audio segment
            segment_audio = self.extract_audio_segment(
                audio_path, 
                segment['start'], 
                segment['duration']
            )
            
            if segment_audio is None:
                print(f"  ❌ Failed to extract audio\n")
                skipped_count += 1
                results.append({
                    **segment,
                    'vad_stats': vad_stats,
                    'transcription': "",
                    'stt_confidence': 0.0,
                    'combined_confidence': 0.0,
                    'quality': 'skipped',
                    'reason': 'extraction_failed'
                })
                continue
            
            # Transcribe
            stt_result = self.transcribe_segment(segment_audio, language)
            
            # Calculate combined confidence
            combined_conf = self.calculate_combined_confidence(
                vad_stats['mean'],
                stt_result['stt_confidence']
            )
            
            # Determine quality
            if combined_conf >= self.min_combined_confidence:
                quality = 'high'
                high_quality_count += 1
                icon = "✅"
            elif combined_conf >= 0.4:
                quality = 'medium'
                icon = "⚠️ "
            else:
                quality = 'low'
                low_quality_count += 1
                icon = "❌"
            
            print(f"  {icon} Speaker: {segment['speaker']}")
            print(f"     VAD: {vad_stats['mean']:.2f} | STT: {stt_result['stt_confidence']:.2f} | Combined: {combined_conf:.2f}")
            print(f"     Quality: {quality.upper()}")
            print(f"     Text: {stt_result['text'][:60]}{'...' if len(stt_result['text']) > 60 else ''}\n")
            
            # Store result
            results.append({
                **segment,
                'vad_stats': vad_stats,
                'transcription': stt_result['text'],
                'stt_confidence': stt_result['stt_confidence'],
                'combined_confidence': combined_conf,
                'quality': quality,
                'success': stt_result['success']
            })
            
            # Cleanup temp file
            try:
                if segment_audio and os.path.exists(segment_audio):
                    os.remove(segment_audio)
            except:
                pass
        
        processing_time = time.time() - start_time
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("📊 Processing Summary")
        print(f"{'='*80}\n")
        print(f"Total segments:      {len(segments)}")
        print(f"High quality:        {high_quality_count} ({high_quality_count/len(segments)*100:.1f}%)")
        print(f"Medium quality:      {len(segments) - high_quality_count - low_quality_count - skipped_count}")
        print(f"Low quality:         {low_quality_count} ({low_quality_count/len(segments)*100:.1f}%)")
        print(f"Skipped:             {skipped_count} ({skipped_count/len(segments)*100:.1f}%)")
        print(f"Processing time:     {processing_time:.2f}s ({processing_time/len(segments):.2f}s per segment)\n")
        
        return {
            'results': results,
            'statistics': {
                'total': len(segments),
                'high_quality': high_quality_count,
                'low_quality': low_quality_count,
                'skipped': skipped_count,
                'processing_time': processing_time
            },
            'audio_path': audio_path,
            'vad_threshold': self.vad_threshold,
            'min_combined_confidence': self.min_combined_confidence
        }


def save_results(output_data: Dict, output_dir: str):
    """Save results to JSON and formatted text files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete JSON
    json_path = os.path.join(output_dir, "confidence_weighted_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved full results: {json_path}")
    
    # Save high-quality transcriptions only
    high_quality_path = os.path.join(output_dir, "high_quality_transcription.txt")
    with open(high_quality_path, 'w', encoding='utf-8') as f:
        for result in output_data['results']:
            if result['quality'] == 'high':
                f.write(f"{result['speaker']}: {result['transcription']}\n")
    print(f"✅ Saved high-quality transcription: {high_quality_path}")
    
    # Save segments needing review
    review_path = os.path.join(output_dir, "segments_for_review.txt")
    with open(review_path, 'w', encoding='utf-8') as f:
        f.write("Segments Requiring Manual Review\n")
        f.write("="*80 + "\n\n")
        for result in output_data['results']:
            if result['quality'] in ['low', 'medium']:
                f.write(f"Time: {result['start']:.2f}s - {result['end']:.2f}s\n")
                f.write(f"Speaker: {result['speaker']}\n")
                f.write(f"Combined Confidence: {result['combined_confidence']:.2f}\n")
                f.write(f"Transcription: {result['transcription']}\n")
                f.write("-"*80 + "\n\n")
    print(f"⚠️  Saved review list: {review_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Confidence-weighted transcription using diarization output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python confidence_weighted_transcription.py test01.wav
  python confidence_weighted_transcription.py test01.wav --vad-threshold 0.6
  python confidence_weighted_transcription.py test01.wav --output results/
        """
    )
    
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--diarization-dir', default='./diarization_output',
                       help='Directory containing diarization output (default: ./diarization_output)')
    parser.add_argument('--output', default='./confidence_output',
                       help='Output directory for results (default: ./confidence_output)')
    parser.add_argument('--vad-threshold', type=float, default=0.5,
                       help='VAD threshold for speech detection (default: 0.5)')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                       help='Minimum combined confidence for high quality (default: 0.6)')
    parser.add_argument('--language', default='auto',
                       help='Language for transcription (default: auto)')
    
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        return 1
    
    # Find RTTM and VAD files
    audio_basename = Path(args.audio_file).stem
    rttm_path = os.path.join(args.diarization_dir, f"pred_rttms/{audio_basename}.rttm")
    vad_path = os.path.join(args.diarization_dir, f"vad_outputs/{audio_basename}.frame")
    
    if not os.path.exists(rttm_path):
        print(f"❌ RTTM file not found: {rttm_path}")
        print(f"💡 Run diarization first using diarize.py")
        return 1
    
    if not os.path.exists(vad_path):
        print(f"❌ VAD file not found: {vad_path}")
        print(f"💡 Make sure diarization generated VAD outputs")
        return 1
    
    print(f"✅ Found diarization files:")
    print(f"   RTTM: {rttm_path}")
    print(f"   VAD:  {vad_path}")
    
    # Initialize transcriber
    transcriber = ConfidenceWeightedTranscriber(
        vad_threshold=args.vad_threshold,
        min_combined_confidence=args.min_confidence
    )
    
    # Process audio
    results = transcriber.process_audio_with_confidence(
        audio_path=args.audio_file,
        rttm_path=rttm_path,
        vad_path=vad_path,
        language=args.language
    )
    
    # Save results
    save_results(results, args.output)
    
    print(f"{'='*80}")
    print("✅ Processing complete!")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())

