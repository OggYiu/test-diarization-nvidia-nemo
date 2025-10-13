# Files Overview

Quick reference for all files in the diarization-api folder.

## Core Files

### `audio_pipeline.py` ⭐
**Main program - run this!**

The unified audio processing pipeline that combines:
- Speaker diarization (from `diarization.py`)
- Audio chopping (from `audio_chopper.py`)
- Speech-to-text (from `batch_stt.py`)
- LLM analysis (from `llm_analysis.py`)

**Usage:**
```bash
python audio_pipeline.py audio.wav --skip-llm
```

**Key features:**
- Complete end-to-end pipeline
- All outputs in one directory
- Modular design (can use as library)
- Extensive error handling

---

### `requirements.txt`
**Dependencies list**

All required Python packages:
- PyTorch + TorchAudio
- NeMo Toolkit (diarization)
- FunASR (speech-to-text)
- LangChain + Ollama (LLM)
- Audio libraries (pydub, librosa)

**Usage:**
```bash
pip install -r requirements.txt
```

---

## Utility Scripts

### `check_installation.py`
**Installation verification tool**

Checks if all dependencies are installed correctly.

**Usage:**
```bash
python check_installation.py
```

**Output:**
```
✓ numpy
✓ torch
✓ nemo_toolkit
✗ langchain - NOT INSTALLED
```

---

### `test_pipeline.py`
**Simple test script**

Quick way to test the pipeline with your audio file.
Uses default settings and skips LLM analysis.

**Usage:**
```bash
python test_pipeline.py path/to/audio.wav
```

**What it does:**
- Runs pipeline with defaults (2 speakers, auto language)
- Saves output to `./test_output/`
- Shows conversation preview
- Reports success/failure

---

### `batch_process.py`
**Batch processing script**

Process multiple audio files at once.

**Usage:**
```bash
python batch_process.py /path/to/audio/folder --output-dir ./results
```

**Features:**
- Process entire directories
- Individual output folders per file
- Summary report (JSON)
- Error handling per file

---

### `example.sh`
**Shell script with example commands**

Various usage examples demonstrating different options.

**Usage:**
```bash
bash example.sh
```

Or copy commands from the file for your use case.

---

## Documentation

### `README.md` 📖
**Complete documentation**

Comprehensive guide covering:
- Installation instructions
- Full pipeline explanation
- All command-line options
- Troubleshooting
- Output formats
- Examples

**Start here for detailed information!**

---

### `QUICKSTART.md` 🚀
**5-minute quick start**

Get up and running quickly:
- Fast installation steps
- Simple examples
- Common problems
- Basic usage

**Start here if you want to try it quickly!**

---

### `SUMMARY.md` 📋
**Project summary**

High-level overview:
- What was created
- Key features
- Use cases
- Configuration options
- Integration examples

**Good for understanding the project at a glance.**

---

### `FILES_OVERVIEW.md` 📑
**This file!**

Quick reference for all files and their purposes.

---

## Configuration Files

### `.gitignore`
**Git ignore rules**

Excludes from version control:
- Python cache files
- Output directories
- Downloaded models
- IDE settings
- Log files

---

## Typical Workflow

### First Time Setup
1. **Check installation**: `python check_installation.py`
2. **Read documentation**: Start with `QUICKSTART.md`
3. **Run test**: `python test_pipeline.py audio.wav`

### Regular Use
```bash
# Single file
python audio_pipeline.py call.wav --skip-llm

# With LLM analysis
python audio_pipeline.py call.wav \
  --llm-model qwen2.5:7b-instruct \
  --ollama-url http://localhost:11434

# Batch processing
python batch_process.py ./recordings/ --output-dir ./results
```

### Integration in Your Code
```python
from audio_pipeline import AudioPipeline

pipeline = AudioPipeline(work_dir='./output')
results = pipeline.process_audio('audio.wav', skip_llm=True)
```

---

## File Hierarchy

```
diarization-api/
│
├── 🟢 Main Program
│   └── audio_pipeline.py       ← Run this!
│
├── 🔧 Utilities
│   ├── check_installation.py   ← Verify setup
│   ├── test_pipeline.py        ← Quick test
│   ├── batch_process.py        ← Process multiple files
│   └── example.sh              ← Example commands
│
├── 📚 Documentation
│   ├── README.md               ← Full docs
│   ├── QUICKSTART.md           ← Quick start
│   ├── SUMMARY.md              ← Overview
│   └── FILES_OVERVIEW.md       ← This file
│
├── ⚙️ Configuration
│   ├── requirements.txt        ← Dependencies
│   └── .gitignore              ← Git rules
│
└── 📁 Output (created on run)
    └── output/                 ← Results go here
        ├── diarization/
        ├── audio_chunks/
        ├── transcriptions/
        └── analysis/
```

---

## Quick Command Reference

| Task | Command |
|------|---------|
| **Check setup** | `python check_installation.py` |
| **Quick test** | `python test_pipeline.py audio.wav` |
| **Basic run** | `python audio_pipeline.py audio.wav --skip-llm` |
| **With LLM** | `python audio_pipeline.py audio.wav --llm-model qwen2.5:7b-instruct` |
| **Custom speakers** | `python audio_pipeline.py audio.wav --num-speakers 3` |
| **Set language** | `python audio_pipeline.py audio.wav --language zh` |
| **Batch process** | `python batch_process.py ./folder/ --output-dir ./results` |
| **Get help** | `python audio_pipeline.py --help` |

---

## Need Help?

1. **Installation issues**: Run `python check_installation.py`
2. **Quick start**: Read `QUICKSTART.md`
3. **Detailed docs**: Read `README.md`
4. **Examples**: Check `example.sh`
5. **Test first**: Use `python test_pipeline.py`

---

## What's Next?

After successful installation and testing:

1. **Process your audio files**: Use `audio_pipeline.py`
2. **Customize settings**: Adjust speakers, language, prompts
3. **Batch processing**: Use `batch_process.py` for multiple files
4. **Integration**: Import `AudioPipeline` class in your code
5. **Production deployment**: Consider GPU, Docker, web interface

---

**Happy processing! 🎉**

