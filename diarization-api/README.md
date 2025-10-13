# Audio Processing Pipeline

A unified audio processing pipeline that combines speaker diarization, audio segmentation, speech-to-text transcription, and LLM-based analysis.

## Features

- **Speaker Diarization**: Uses NVIDIA NeMo to identify different speakers in audio
- **Audio Segmentation**: Chops audio into segments based on speaker changes
- **Speech-to-Text**: Transcribes audio segments using FunASR SenseVoice model
- **LLM Analysis**: Analyzes transcripts using Ollama (optional)
- **Self-contained**: All outputs and temporary files stay within the work directory

## Pipeline Steps

1. **Diarization** → Identifies speaker segments and creates RTTM files
2. **Audio Chopping** → Splits audio into segments based on speaker changes
3. **Speech-to-Text** → Transcribes each audio segment
4. **LLM Analysis** → Analyzes the conversation (optional)

## Installation

### 1. Install PyTorch

```bash
# CPU version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or GPU version (CUDA 11.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install NeMo

```bash
pip install nemo_toolkit[all]
```

### 3. Install other dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg (required by pydub)

- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

### Basic Usage

```bash
python audio_pipeline.py path/to/audio.wav
```

This will:
- Process the audio with default settings (2 speakers)
- Save all outputs to `./output/` directory
- Skip LLM analysis (no Ollama server needed)

### Advanced Usage

```bash
python audio_pipeline.py path/to/audio.wav \
  --work-dir ./my_results \
  --num-speakers 3 \
  --language zh \
  --padding 200 \
  --llm-model qwen2.5:7b-instruct \
  --ollama-url http://localhost:11434
```

### Command-Line Arguments

**Input/Output:**
- `audio_file`: Path to the audio file (required)
- `--work-dir`: Working directory for outputs (default: `./output`)

**Diarization:**
- `--num-speakers`: Number of speakers (default: 2)

**Audio Chopping:**
- `--padding`: Padding in milliseconds for audio chunks (default: 100)

**Speech-to-Text:**
- `--language`: Language code - auto/zh/en/yue/ja/ko (default: auto)
- `--stt-model`: STT model name (default: iic/SenseVoiceSmall)

**LLM Analysis:**
- `--skip-llm`: Skip LLM analysis step
- `--llm-model`: Ollama model name (default: gpt-oss:20b)
- `--ollama-url`: Ollama server URL (default: http://192.168.61.2:11434)
- `--prompt`: Custom system prompt for LLM

### Examples

**Process English audio with 2 speakers:**
```bash
python audio_pipeline.py recording.wav --language en
```

**Process Cantonese audio with 3 speakers:**
```bash
python audio_pipeline.py recording.wav --language yue --num-speakers 3
```

**Skip LLM analysis:**
```bash
python audio_pipeline.py recording.wav --skip-llm
```

**Use local Ollama server:**
```bash
python audio_pipeline.py recording.wav \
  --ollama-url http://localhost:11434 \
  --llm-model llama2
```

**Custom prompt for LLM:**
```bash
python audio_pipeline.py recording.wav \
  --prompt "Analyze this conversation and summarize the key points."
```

## Output Structure

All outputs are saved in the work directory (default: `./output/`):

```
output/
├── diarization/           # Diarization results
│   ├── pred_rttms/        # RTTM files
│   ├── speaker_outputs/   # Speaker embeddings
│   └── vad_outputs/       # Voice activity detection
├── audio_chunks/          # Segmented audio files
│   ├── segment_001.wav
│   ├── segment_002.wav
│   └── ...
├── transcriptions/        # Transcription results
│   ├── transcriptions.json
│   └── conversation.txt
├── analysis/              # LLM analysis (if not skipped)
│   └── analysis.txt
└── pipeline_summary.json  # Overall summary
```

## File Formats

### transcriptions.json
```json
[
  {
    "file": "segment_001.wav",
    "segment_num": 1,
    "speaker": "speaker_0",
    "start": 0.5,
    "end": 3.2,
    "duration": 2.7,
    "transcription": "Hello, how are you?",
    "raw_transcription": "<|en|>Hello, how are you?"
  },
  ...
]
```

### conversation.txt
```
speaker_0: Hello, how are you?
speaker_1: I'm doing well, thank you!
speaker_0: That's great to hear.
...
```

## Requirements

- Python 3.8+
- PyTorch (CPU or GPU)
- NeMo Toolkit
- FunASR
- FFmpeg
- (Optional) Ollama server for LLM analysis

## Troubleshooting

### NeMo Installation Issues

If you encounter issues installing NeMo:

```bash
# Try installing with specific index
pip install nemo_toolkit[all] --extra-index-url https://pypi.ngc.nvidia.com
```

### FFmpeg Not Found

If pydub complains about FFmpeg:
- Make sure FFmpeg is installed and in your PATH
- Test: `ffmpeg -version`

### Ollama Connection Error

If LLM analysis fails:
- Make sure Ollama is running: `ollama serve`
- Verify the URL is correct
- Or skip LLM analysis with `--skip-llm`

### CUDA/GPU Issues

If you want to use CPU only:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Notes

- First run will download required models (NeMo, SenseVoice)
- Diarization can be slow on CPU (recommend GPU for production)
- SenseVoice supports multiple languages (Chinese, English, Cantonese, Japanese, Korean)
- LLM analysis requires a running Ollama server

## License

This pipeline combines multiple open-source tools. Please refer to their respective licenses:
- NeMo: Apache 2.0
- FunASR: MIT
- LangChain: MIT

