#!/usr/bin/env python
# coding=utf-8
"""
Audio Enhancement Script for Phone Recordings
Optimized for poor quality recordings, elderly speakers, and Cantonese speech
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import tempfile
import subprocess
import warnings
from scipy import signal

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if sys.stderr:
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

warnings.filterwarnings('ignore')

# Check if Highpass/Lowpass are available (torchaudio >= 0.12.0)
try:
    _test_highpass = T.Highpass(sample_rate=16000, cutoff_freq=80)
    TORCHAUDIO_FILTERS_AVAILABLE = True
except AttributeError:
    TORCHAUDIO_FILTERS_AVAILABLE = False
    print("⚠️  Warning: Using older torchaudio version. Falling back to scipy filters.")

# Try importing optional enhancement libraries
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("⚠️  Warning: noisereduce not installed. Install with: pip install noisereduce")

try:
    from funasr import AutoModel
    from batch_stt import format_str_v3
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False
    print("⚠️  Warning: funasr not available. Transcription will not work.")


class AudioEnhancer:
    """Audio enhancement specifically for phone recordings."""
    
    def __init__(self, target_sr=16000):
        """
        Initialize audio enhancer.
        
        Args:
            target_sr: Target sample rate (16000 is optimal for SenseVoice)
        """
        self.target_sr = target_sr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_audio(self, audio_path):
        """
        Load audio file and convert to mono at target sample rate.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            tuple: (waveform, sample_rate)
        """
        # Load audio - try soundfile directly to avoid torchcodec dependency
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path)
            # Convert to torch tensor and add channel dimension
            waveform = torch.from_numpy(data).float()
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension
            else:
                waveform = waveform.T  # soundfile returns (samples, channels), we need (channels, samples)
        except Exception as e:
            # Fallback to torchaudio.load
            try:
                waveform, sr = torchaudio.load(audio_path, backend="soundfile")
            except:
                waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            sr = self.target_sr
        
        return waveform, sr
    
    def normalize_audio(self, waveform, target_level=-20.0):
        """
        Normalize audio to target dBFS level.
        
        Args:
            waveform: Audio tensor
            target_level: Target dBFS level (default -20.0 for speech)
            
        Returns:
            Normalized waveform
        """
        # Calculate current RMS
        rms = torch.sqrt(torch.mean(waveform ** 2))
        
        if rms == 0:
            return waveform
        
        # Calculate current dBFS
        current_db = 20 * torch.log10(rms)
        
        # Calculate gain needed
        gain_db = target_level - current_db
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = waveform * gain_linear
        
        # Prevent clipping
        max_val = torch.max(torch.abs(normalized))
        if max_val > 0.99:
            normalized = normalized * (0.99 / max_val)
        
        return normalized
    
    def apply_highpass_filter(self, waveform, sr, cutoff=80):
        """
        Apply high-pass filter to remove low-frequency rumble.
        Good for removing handling noise and room rumble.
        
        Args:
            waveform: Audio tensor
            sr: Sample rate
            cutoff: Cutoff frequency in Hz (default 80Hz)
            
        Returns:
            Filtered waveform
        """
        if TORCHAUDIO_FILTERS_AVAILABLE:
            highpass = T.Highpass(sample_rate=sr, cutoff_freq=cutoff)
            filtered = highpass(waveform)
        else:
            # Fallback using scipy
            nyquist = sr / 2
            normal_cutoff = cutoff / nyquist
            sos = signal.butter(5, normal_cutoff, btype='high', output='sos')
            
            # Apply filter to numpy array
            audio_np = waveform.squeeze().cpu().numpy()
            filtered_np = signal.sosfilt(sos, audio_np)
            filtered = torch.from_numpy(filtered_np).unsqueeze(0).float()
        
        return filtered
    
    def apply_lowpass_filter(self, waveform, sr, cutoff=8000):
        """
        Apply low-pass filter to remove high-frequency hiss.
        Phone bandwidth is typically 300-3400 Hz, but we use 8kHz to preserve quality.
        
        Args:
            waveform: Audio tensor
            sr: Sample rate
            cutoff: Cutoff frequency in Hz (default 8000Hz)
            
        Returns:
            Filtered waveform
        """
        # Ensure cutoff is less than Nyquist frequency
        nyquist = sr / 2
        if cutoff >= nyquist:
            cutoff = nyquist * 0.95  # Use 95% of Nyquist as safe maximum
        
        if TORCHAUDIO_FILTERS_AVAILABLE:
            lowpass = T.Lowpass(sample_rate=sr, cutoff_freq=cutoff)
            filtered = lowpass(waveform)
        else:
            # Fallback using scipy
            normal_cutoff = cutoff / nyquist
            sos = signal.butter(5, normal_cutoff, btype='low', output='sos')
            
            # Apply filter to numpy array
            audio_np = waveform.squeeze().cpu().numpy()
            filtered_np = signal.sosfilt(sos, audio_np)
            filtered = torch.from_numpy(filtered_np).unsqueeze(0).float()
        
        return filtered
    
    def reduce_noise(self, waveform, sr):
        """
        Apply advanced noise reduction using spectral gating.
        
        Args:
            waveform: Audio tensor
            sr: Sample rate
            
        Returns:
            Denoised waveform
        """
        if not NOISEREDUCE_AVAILABLE:
            print("  ⚠️  Skipping noise reduction (noisereduce not installed)")
            return waveform
        
        # Convert to numpy for noisereduce
        audio_np = waveform.squeeze().cpu().numpy()
        
        # Apply stationary noise reduction
        # This works well for constant background noise like phone line hiss
        reduced = nr.reduce_noise(
            y=audio_np,
            sr=sr,
            stationary=True,
            prop_decrease=0.8,  # Reduce noise by 80%
        )
        
        # Convert back to torch
        reduced_tensor = torch.from_numpy(reduced).unsqueeze(0)
        
        return reduced_tensor
    
    def apply_dynamic_range_compression(self, waveform, threshold=-30, ratio=4, attack=5, release=50):
        """
        Apply dynamic range compression to make quiet parts louder and loud parts quieter.
        This helps with elderly speakers who may have varying volume.
        
        Args:
            waveform: Audio tensor
            threshold: Compression threshold in dB
            ratio: Compression ratio (4:1 is moderate)
            attack: Attack time in ms
            release: Release time in ms
            
        Returns:
            Compressed waveform
        """
        # Simple soft-knee compression
        # Convert to dB
        db = 20 * torch.log10(torch.abs(waveform) + 1e-8)
        
        # Apply compression above threshold
        mask = db > threshold
        excess_db = db - threshold
        compressed_excess = excess_db / ratio
        
        # Calculate new values
        new_db = torch.where(mask, threshold + compressed_excess, db)
        
        # Convert back to linear
        compressed = torch.sign(waveform) * (10 ** (new_db / 20))
        
        return compressed
    
    def apply_speech_enhancement(self, waveform, sr):
        """
        Apply spectral enhancement for speech clarity.
        Emphasizes speech frequencies (300-3400 Hz for telephony).
        
        Args:
            waveform: Audio tensor
            sr: Sample rate
            
        Returns:
            Enhanced waveform
        """
        # Create equalizer to boost speech frequencies
        # Boost 500-2000 Hz (fundamental speech frequencies)
        
        # Simple approach: bandpass emphasis
        # Create bandpass for speech range
        speech_low = 300
        speech_high = 3400
        
        if TORCHAUDIO_FILTERS_AVAILABLE:
            # Apply gentle boost to speech frequencies
            lowpass_speech = T.Lowpass(sample_rate=sr, cutoff_freq=speech_high)
            highpass_speech = T.Highpass(sample_rate=sr, cutoff_freq=speech_low)
            
            speech_band = highpass_speech(lowpass_speech(waveform))
        else:
            # Fallback using scipy bandpass
            nyquist = sr / 2
            low = speech_low / nyquist
            high = speech_high / nyquist
            sos = signal.butter(5, [low, high], btype='band', output='sos')
            
            audio_np = waveform.squeeze().cpu().numpy()
            speech_band_np = signal.sosfilt(sos, audio_np)
            speech_band = torch.from_numpy(speech_band_np).unsqueeze(0).float()
        
        # Mix back with original (boost speech band by 20%)
        enhanced = waveform + (speech_band * 0.2)
        
        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(enhanced))
        if max_val > 0.99:
            enhanced = enhanced * (0.99 / max_val)
        
        return enhanced
    
    def enhance(self, audio_path, output_path=None, verbose=True):
        """
        Apply full enhancement pipeline.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save enhanced audio (optional)
            verbose: Print progress messages
            
        Returns:
            Path to enhanced audio file
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🎵 Audio Enhancement Pipeline")
            print(f"{'='*60}")
            print(f"Input: {audio_path}")
            print(f"Target sample rate: {self.target_sr} Hz")
            print(f"Device: {self.device}")
        
        # Create output path if not provided
        if output_path is None:
            input_path = Path(audio_path)
            output_path = input_path.parent / f"{input_path.stem}_enhanced.wav"
        
        # Load audio
        if verbose:
            print("\n📂 Loading audio...")
        waveform, sr = self.load_audio(audio_path)
        if verbose:
            print(f"✓ Loaded: {waveform.shape}, {sr} Hz")
        
        # Step 1: High-pass filter (remove low rumble)
        if verbose:
            print("\n🔧 Step 1: High-pass filter (remove rumble)...")
        waveform = self.apply_highpass_filter(waveform, sr, cutoff=80)
        if verbose:
            print("✓ Applied high-pass filter at 80 Hz")
        
        # Step 2: Low-pass filter (remove high hiss)
        if verbose:
            print("\n🔧 Step 2: Low-pass filter (remove hiss)...")
        waveform = self.apply_lowpass_filter(waveform, sr, cutoff=8000)
        if verbose:
            print("✓ Applied low-pass filter at 8000 Hz")
        
        # Step 3: Noise reduction
        if verbose:
            print("\n🔧 Step 3: Noise reduction...")
        waveform = self.reduce_noise(waveform, sr)
        if verbose:
            print("✓ Applied spectral noise reduction")
        
        # Step 4: Speech enhancement
        if verbose:
            print("\n🔧 Step 4: Speech frequency enhancement...")
        waveform = self.apply_speech_enhancement(waveform, sr)
        if verbose:
            print("✓ Enhanced speech frequencies (300-3400 Hz)")
        
        # Step 5: Dynamic range compression
        if verbose:
            print("\n🔧 Step 5: Dynamic range compression...")
        waveform = self.apply_dynamic_range_compression(waveform)
        if verbose:
            print("✓ Applied compression (helps with volume variations)")
        
        # Step 6: Final normalization
        if verbose:
            print("\n🔧 Step 6: Final normalization...")
        waveform = self.normalize_audio(waveform, target_level=-20.0)
        if verbose:
            print("✓ Normalized to -20 dBFS")
        
        # Save enhanced audio
        if verbose:
            print(f"\n💾 Saving enhanced audio...")
        # Use soundfile to save (avoid torchcodec dependency)
        import soundfile as sf
        audio_np = waveform.squeeze().cpu().numpy()
        sf.write(output_path, audio_np, sr)
        
        if verbose:
            print(f"✓ Saved: {output_path}")
            
            # Calculate file sizes
            input_size = os.path.getsize(audio_path) / 1024
            output_size = os.path.getsize(output_path) / 1024
            print(f"\n📊 Summary:")
            print(f"  Input size:  {input_size:.2f} KB")
            print(f"  Output size: {output_size:.2f} KB")
            print(f"{'='*60}\n")
        
        return str(output_path)


def transcribe_enhanced_audio(audio_path, language="yue", model_name="iic/SenseVoiceSmall"):
    """
    Transcribe enhanced audio using SenseVoice.
    
    Args:
        audio_path: Path to audio file
        language: Language code (default "yue" for Cantonese)
        model_name: Model name (default SenseVoiceSmall)
        
    Returns:
        Transcription text
    """
    if not FUNASR_AVAILABLE:
        print("❌ Error: funasr not available. Cannot transcribe.")
        return None
    
    print(f"\n{'='*60}")
    print("📝 Transcription with SenseVoice")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Language: {language}")
    
    # Load model
    print("\n⚙️  Loading SenseVoice model...")
    model = AutoModel(
        model=model_name,
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_kwargs={"max_single_segment_time": 30000},
        trust_remote_code=False,
        disable_update=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    print("✓ Model loaded")
    
    # Transcribe
    print("\n🎤 Transcribing...")
    result = model.generate(
        input=audio_path,
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    
    # Extract and format text
    raw_text = result[0]["text"]
    formatted_text = format_str_v3(raw_text)
    
    print("✓ Transcription complete")
    print(f"\n{'='*60}")
    print("📄 Transcription:")
    print(f"{'='*60}")
    print(formatted_text)
    print(f"{'='*60}\n")
    
    return formatted_text


def main():
    parser = argparse.ArgumentParser(
        description="Enhance audio quality for phone recordings (optimized for elderly Cantonese speakers)"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to input audio file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for enhanced audio (default: <input>_enhanced.wav)"
    )
    parser.add_argument(
        "-t", "--transcribe",
        action="store_true",
        help="Transcribe the enhanced audio using SenseVoice"
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default="yue",
        choices=["auto", "zh", "en", "yue", "ja", "ko"],
        help="Language for transcription (default: yue for Cantonese)"
    )
    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000 Hz)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Create enhancer
    enhancer = AudioEnhancer(target_sr=args.sample_rate)
    
    # Enhance audio
    try:
        enhanced_path = enhancer.enhance(
            str(audio_path),
            output_path=args.output,
            verbose=not args.quiet
        )
        
        # Transcribe if requested
        if args.transcribe:
            if not FUNASR_AVAILABLE:
                print("❌ Cannot transcribe: funasr not available")
                sys.exit(1)
            
            transcription = transcribe_enhanced_audio(
                enhanced_path,
                language=args.language
            )
            
            # Save transcription
            transcript_path = Path(enhanced_path).parent / f"{Path(enhanced_path).stem}_transcript.txt"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
            
            print(f"💾 Transcription saved to: {transcript_path}")
        
        print("\n✅ Success!")
        print(f"Enhanced audio: {enhanced_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

