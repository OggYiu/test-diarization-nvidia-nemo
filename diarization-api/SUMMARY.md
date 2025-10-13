# Audio Processing Pipeline - Summary

## What Was Created

A unified audio processing pipeline that combines functionality from:
1. `diarization.py` - Speaker diarization using NVIDIA NeMo
2. `audio_chopper.py` - Audio segmentation based on diarization
3. `batch_stt.py` - Speech-to-text using FunASR SenseVoice
4. `llm_analysis.py` - LLM-based conversation analysis using Ollama

## File Structure

```
diarization-api/
├── audio_pipeline.py          # Main unified pipeline (run this!)
├── requirements.txt           # All dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md             # Quick start guide
├── check_installation.py      # Verify installation
├── example.sh                # Example commands
├── .gitignore                # Git ignore rules
└── output/                   # All results go here (created on first run)
    ├── diarization/          # Speaker diarization results
    ├── audio_chunks/         # Segmented audio files
    ├── transcriptions/       # Speech-to-text results
    └── analysis/             # LLM analysis results
```

## Key Features

### 1. Self-Contained
- All outputs and temporary files stay within `./output/` (or custom `--work-dir`)
- No files scattered across different directories
- Easy to clean up or backup results

### 2. Complete Pipeline
```
Input Audio → Diarization → Audio Chopping → STT → LLM Analysis → Results
```

### 3. Flexible Configuration
- Number of speakers: `--num-speakers`
- Language selection: `--language` (auto/zh/en/yue/ja/ko)
- Custom output directory: `--work-dir`
- Optional LLM analysis: `--skip-llm`
- Custom LLM prompt: `--prompt`

### 4. Production Ready
- Error handling and validation
- Progress reporting
- Comprehensive logging
- Results saved in multiple formats (JSON, TXT)

## Quick Start

### 1. Check Installation
```bash
python check_installation.py
```

### 2. Run Pipeline
```bash
# Basic usage (no LLM)
python audio_pipeline.py path/to/audio.wav --skip-llm

# Full pipeline with LLM
python audio_pipeline.py path/to/audio.wav \
  --llm-model qwen2.5:7b-instruct \
  --ollama-url http://localhost:11434
```

### 3. Check Results
```bash
# View conversation transcript
cat output/transcriptions/conversation.txt

# View LLM analysis
cat output/analysis/analysis.txt

# View full JSON results
cat output/transcriptions/transcriptions.json
```

## Pipeline Steps Explained

### Step 1: Diarization
- Uses NVIDIA NeMo ClusteringDiarizer
- Identifies speaker segments in audio
- Outputs RTTM file with speaker timestamps
- Location: `output/diarization/pred_rttms/`

### Step 2: Audio Chopping
- Reads RTTM file from Step 1
- Segments audio based on speaker changes
- Adds optional padding (default: 100ms)
- Outputs: `output/audio_chunks/segment_*.wav`

### Step 3: Speech-to-Text
- Processes each audio segment
- Uses FunASR SenseVoice model
- Supports multiple languages
- Outputs: `output/transcriptions/transcriptions.json` and `conversation.txt`

### Step 4: LLM Analysis (Optional)
- Analyzes full conversation
- Uses Ollama for local LLM inference
- Customizable prompt
- Output: `output/analysis/analysis.txt`

## Use Cases

### 1. Phone Call Analysis
```bash
python audio_pipeline.py call_recording.wav \
  --num-speakers 2 \
  --language yue \
  --prompt "Identify who is the broker and who is the customer. Extract trading orders."
```

### 2. Meeting Transcription
```bash
python audio_pipeline.py meeting.wav \
  --num-speakers 5 \
  --language en \
  --skip-llm
```

### 3. Podcast Processing
```bash
python audio_pipeline.py podcast.wav \
  --num-speakers 3 \
  --language auto \
  --work-dir ./podcast_results
```

## Output Files

### transcriptions/conversation.txt
```
speaker_0: Hello, how can I help you today?
speaker_1: I'd like to buy 1000 shares of stock 0005.
speaker_0: Confirmed. 1000 shares at market price.
```

### transcriptions/transcriptions.json
```json
[
  {
    "file": "segment_001.wav",
    "segment_num": 1,
    "speaker": "speaker_0",
    "start": 0.5,
    "end": 3.2,
    "duration": 2.7,
    "transcription": "Hello, how can I help you today?"
  }
]
```

### analysis/analysis.txt
```
=== Conversation ===
[conversation transcript]

=== Analysis ===
[LLM analysis of the conversation]
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--work-dir` | `./output` | Output directory |
| `--num-speakers` | `2` | Number of speakers |
| `--language` | `auto` | Language (auto/zh/en/yue/ja/ko) |
| `--padding` | `100` | Padding in ms for audio chunks |
| `--stt-model` | `iic/SenseVoiceSmall` | STT model |
| `--skip-llm` | `False` | Skip LLM analysis |
| `--llm-model` | `gpt-oss:20b` | Ollama model |
| `--ollama-url` | `http://192.168.61.2:11434` | Ollama server |
| `--prompt` | (default) | Custom LLM prompt |

## Performance Notes

- **First run**: Downloads models (~several GB)
- **CPU processing**: Slower but works
- **GPU processing**: Much faster (recommended for production)
- **Memory**: ~4-8GB RAM recommended
- **Storage**: ~10GB for models + output files

## Dependencies

### Required
- Python 3.8+
- PyTorch + TorchAudio
- NeMo Toolkit
- FunASR
- Pydub + FFmpeg

### Optional
- LangChain + LangChain Ollama (for LLM analysis)
- Ollama server (for LLM analysis)

## Troubleshooting

### Module not found errors
```bash
pip install -r requirements.txt
```

### FFmpeg not found
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### NeMo installation issues
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install nemo_toolkit[all]
```

### Ollama connection error
- Make sure Ollama is running: `ollama serve`
- Or skip LLM: `--skip-llm`

## Integration with Existing Code

The pipeline can be used as a Python module:

```python
from audio_pipeline import AudioPipeline

# Initialize
pipeline = AudioPipeline(work_dir='./output', num_speakers=2)

# Process audio
results = pipeline.process_audio(
    audio_filepath='audio.wav',
    language='auto',
    skip_llm=True
)

# Access results
print(results['transcriptions'])
print(results['analysis'])
```

## Future Enhancements

Potential improvements:
- [ ] Web interface (FastAPI + React)
- [ ] Batch processing multiple files
- [ ] Real-time streaming support
- [ ] Custom model fine-tuning
- [ ] Export to various formats (SRT, VTT, etc.)
- [ ] Speaker identification (not just diarization)
- [ ] Emotion detection
- [ ] Multi-language mixed conversations

## Credits

This pipeline integrates:
- **NVIDIA NeMo** - Speaker diarization
- **FunASR/SenseVoice** - Speech-to-text
- **LangChain** - LLM integration
- **Ollama** - Local LLM inference

## License

Refer to individual component licenses:
- NeMo: Apache 2.0
- FunASR: MIT
- LangChain: MIT

## Support

For issues:
1. Check `check_installation.py`
2. Read `QUICKSTART.md`
3. Review `README.md`
4. Check error logs in console output

---

Created by combining functionality from:
- `diarization.py`
- `audio_chopper.py`
- `batch_stt.py`
- `llm_analysis.py`

All in one unified pipeline! 🎉

