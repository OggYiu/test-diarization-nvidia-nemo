# Audio Enhancement for Phone Recordings

This tool enhances poor quality phone recordings to improve transcription accuracy, specifically optimized for:
- 📞 Phone recordings with background noise
- 👴👵 Elderly speakers with varying volume levels
- 🗣️ Cantonese speech (though works for any language)
- 🤖 SenseVoice transcription model

## Features

### Enhancement Pipeline

1. **High-pass Filter (80 Hz)** - Removes low-frequency rumble and handling noise
2. **Low-pass Filter (8 kHz)** - Removes high-frequency hiss while preserving speech
3. **Spectral Noise Reduction** - Advanced noise reduction using spectral gating
4. **Speech Enhancement** - Boosts frequencies in the 300-3400 Hz range (phone bandwidth)
5. **Dynamic Range Compression** - Makes quiet parts louder, reducing volume variations
6. **Normalization** - Ensures consistent volume level (-20 dBFS target)

### Why This Helps

- **Elderly speakers** often have lower volume or varying speech patterns
- **Phone recordings** have limited bandwidth (300-3400 Hz) and background noise
- **Cantonese tones** are preserved while noise is reduced
- **SenseVoice** performs significantly better with clean, normalized audio

## Installation

### Basic Installation (Required)

```bash
pip install torch torchaudio numpy scipy
```

### Optional (for better noise reduction)

```bash
pip install noisereduce
```

> **Note**: The tool will work without `noisereduce`, but noise reduction will be skipped. Installing it provides significantly better results.

## Usage

### 1. Command Line Interface

#### Basic Enhancement

```bash
python audio_enhancement.py input_audio.wav
```

This will create `input_audio_enhanced.wav` in the same directory.

#### Custom Output Path

```bash
python audio_enhancement.py input_audio.mp3 -o output/enhanced.wav
```

#### Enhance and Transcribe

```bash
python audio_enhancement.py input_audio.wav -t -l yue
```

Options:
- `-t, --transcribe`: Transcribe the enhanced audio using SenseVoice
- `-l, --language`: Language for transcription (default: `yue` for Cantonese)
  - Options: `auto`, `zh`, `en`, `yue`, `ja`, `ko`
- `-o, --output`: Output path for enhanced audio
- `-s, --sample-rate`: Target sample rate (default: 16000 Hz)
- `-q, --quiet`: Suppress progress messages

### 2. Gradio GUI (Standalone)

```bash
python audio_enhancement_gui.py
```

Then open your browser to `http://localhost:7861`

Features:
- Upload audio files (WAV, MP3, FLAC, M4A)
- Real-time progress tracking
- Optional transcription after enhancement
- Audio playback and download

### 3. Unified GUI (All Tools)

The audio enhancement tool is also integrated into the unified GUI as **Tab 6**:

```bash
python unified_gui.py
```

Open your browser to `http://localhost:7860` and navigate to the **"6️⃣ Audio Enhancement"** tab.

### 4. Python API

```python
from audio_enhancement import AudioEnhancer, transcribe_enhanced_audio

# Create enhancer
enhancer = AudioEnhancer(target_sr=16000)

# Enhance audio
enhanced_path = enhancer.enhance(
    audio_path="input_audio.wav",
    output_path="enhanced.wav",
    verbose=True
)

# Optional: Transcribe enhanced audio
transcription = transcribe_enhanced_audio(
    audio_path=enhanced_path,
    language="yue"  # Cantonese
)

print(transcription)
```

#### Custom Enhancement Pipeline

```python
from audio_enhancement import AudioEnhancer
import torchaudio

# Create enhancer
enhancer = AudioEnhancer(target_sr=16000)

# Load audio
waveform, sr = enhancer.load_audio("input.wav")

# Apply individual enhancement steps
waveform = enhancer.apply_highpass_filter(waveform, sr, cutoff=80)
waveform = enhancer.apply_lowpass_filter(waveform, sr, cutoff=8000)
waveform = enhancer.reduce_noise(waveform, sr)
waveform = enhancer.apply_speech_enhancement(waveform, sr)
waveform = enhancer.apply_dynamic_range_compression(waveform)
waveform = enhancer.normalize_audio(waveform, target_level=-20.0)

# Save
torchaudio.save("custom_enhanced.wav", waveform, sr)
```

## Comparison with Grok's Version

### What's Better:

1. **Better Noise Reduction**
   - Uses `noisereduce` library with spectral gating (much better than Wiener filter)
   - Stationary noise reduction optimized for phone line hiss

2. **Speech-Specific Enhancement**
   - Dedicated speech frequency enhancement (300-3400 Hz)
   - Better preservation of speech intelligibility

3. **Dynamic Range Compression**
   - Helps with elderly speakers who may have varying volume levels
   - Soft-knee compression for natural sound

4. **Better Integration**
   - Works seamlessly with existing codebase
   - Proper error handling and progress reporting
   - Both CLI and GUI interfaces

5. **More Robust**
   - Uses PyTorch/torchaudio (already in your stack)
   - No dependency on ffmpeg for audio loading
   - Better handling of different audio formats

6. **Better Normalization**
   - Targets -20 dBFS (optimal for speech recognition)
   - Prevents clipping with proper limiting

## Technical Details

### Audio Processing

- **Sample Rate**: Converts to 16kHz (optimal for SenseVoice)
- **Channels**: Converts stereo to mono
- **Bit Depth**: Outputs 16-bit PCM WAV
- **Normalization Target**: -20 dBFS RMS

### Filter Parameters

- **High-pass**: 80 Hz cutoff (removes rumble, handling noise)
- **Low-pass**: 8 kHz cutoff (removes hiss, preserves speech)
- **Speech band**: 300-3400 Hz (telephone bandwidth)
- **Compression**: 4:1 ratio, -30 dB threshold

### Performance

- **Processing Time**: ~10-30 seconds for typical phone recordings
- **GPU Acceleration**: Automatic when CUDA is available
- **Memory Usage**: Low (processes entire file in memory)

## Recommended Workflow

### For Poor Quality Recordings:

1. **Audio Enhancement** → Clean up the recording first
2. **Diarization** → Better speaker detection with clean audio
3. **Audio Chopper** → Split by speaker
4. **Transcription** → Better transcription from enhanced segments
5. **LLM Analysis** → Analyze the conversation

### Quick Transcription:

1. **Audio Enhancement** with transcription enabled
   - Single-step process
   - Enhanced audio + transcription
   - Saves time for simple workflows

## Troubleshooting

### "noisereduce not installed" warning

Install the optional dependency:
```bash
pip install noisereduce
```

The tool will still work but will skip the noise reduction step.

### "funasr not available" error (transcription)

Make sure funasr and dependencies are installed:
```bash
pip install funasr
pip install modelscope
```

### Audio quality issues

Try adjusting the enhancement parameters in code:
- Lower cutoff frequency for high-pass filter if voice sounds thin
- Adjust compression ratio if dynamics sound unnatural
- Modify target normalization level if output is too loud/quiet

## Examples

### Example 1: Simple Enhancement

```bash
python audio_enhancement.py demo/phone_recordings/test.wav
```

Output: `demo/phone_recordings/test_enhanced.wav`

### Example 2: Enhancement + Transcription

```bash
python audio_enhancement.py demo/phone_recordings/test.wav -t -l yue
```

Output:
- `demo/phone_recordings/test_enhanced.wav`
- `demo/phone_recordings/test_enhanced_transcript.txt`

### Example 3: Batch Processing

```bash
for file in demo/phone_recordings/*.wav; do
    python audio_enhancement.py "$file" -o "enhanced/$(basename "$file")"
done
```

## Support

For issues or questions:
1. Check this README
2. Review the code comments in `audio_enhancement.py`
3. Try the GUI for visual feedback
4. Check the status output for detailed error messages

## Credits

- Uses PyTorch and torchaudio for audio processing
- Optional noise reduction via noisereduce library
- Optimized for SenseVoice transcription model
- Designed for Cantonese phone call analysis

