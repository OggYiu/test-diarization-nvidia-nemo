# Quick Start Guide

Get started with the audio processing pipeline in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system

## Installation

### Step 1: Install PyTorch

```bash
# CPU version (recommended for testing)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# OR GPU version (for production)
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

This will install:
- NeMo Toolkit (for diarization)
- FunASR (for speech-to-text)
- LangChain Ollama (for LLM analysis)
- Other required packages

**Note**: First run will download several GB of models (NeMo, SenseVoice). Be patient!

## Quick Test

### Test without LLM (no Ollama required)

```bash
python audio_pipeline.py path/to/your/audio.wav --skip-llm
```

This will:
1. ✅ Diarize the audio (identify speakers)
2. ✅ Chop audio into speaker segments
3. ✅ Transcribe each segment
4. ✅ Save all results to `./output/`

### Test with LLM (requires Ollama)

First, install and start Ollama:

```bash
# Install Ollama from https://ollama.ai
# Then pull a model
ollama pull qwen2.5:7b-instruct

# Start Ollama server
ollama serve
```

Then run the pipeline:

```bash
python audio_pipeline.py path/to/your/audio.wav \
  --llm-model qwen2.5:7b-instruct \
  --ollama-url http://localhost:11434
```

## Output

Check the `./output/` directory:

```
output/
├── transcriptions/
│   ├── conversation.txt      ← Read this! (formatted conversation)
│   └── transcriptions.json   ← Full details
├── analysis/
│   └── analysis.txt          ← LLM analysis (if not skipped)
├── audio_chunks/             ← Segmented audio files
└── diarization/              ← Diarization results
```

## Common Options

```bash
# Process English audio
python audio_pipeline.py audio.wav --language en

# Specify number of speakers
python audio_pipeline.py audio.wav --num-speakers 3

# Change output directory
python audio_pipeline.py audio.wav --work-dir ./my_results

# Use different STT model (larger, more accurate)
python audio_pipeline.py audio.wav --stt-model iic/SenseVoiceLarge
```

## Troubleshooting

### "Module 'nemo' not found"
```bash
pip install nemo_toolkit[all]
```

### "FFmpeg not found"
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### "Ollama connection error"
- Make sure Ollama is running: `ollama serve`
- Or skip LLM: `--skip-llm`

### "CUDA out of memory"
- Use CPU version of PyTorch
- Or reduce batch size in the config

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize the LLM prompt with `--prompt`
- Process multiple files with a bash script
- Integrate into your own application

## Example for Hong Kong Stock Trading Calls

```bash
python audio_pipeline.py cantonese_call.wav \
  --language yue \
  --num-speakers 2 \
  --llm-model qwen2.5:7b-instruct \
  --prompt "你是一位精通粵語以及香港股市的分析師。請從對話中找出買賣的股票代號、價格、數量等資訊。"
```

Happy processing! 🎉

