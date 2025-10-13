# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```powershell
# Navigate to the diarization folder
cd diarization

# Install PyTorch (CPU version for Windows)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install NeMo
pip install nemo_toolkit[all] --extra-index-url https://pypi.ngc.nvidia.com

# Install API requirements
pip install -r requirements.txt
```

### Step 2: Start the Server

```powershell
# Run the FastAPI server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

### Step 3: Test the API

Open your browser and go to:
- **API Documentation:** http://localhost:8000/docs
- **Try the interactive interface to upload an audio file!**

Or use the test client:
```powershell
# In a new terminal
python test_client.py path/to/your/audio.wav 2
```

## 📝 Example Usage

### Using the Web Interface
1. Go to http://localhost:8000/docs
2. Click on "POST /diarize"
3. Click "Try it out"
4. Upload your audio file
5. Set `num_speakers` (optional, default is 2)
6. Click "Execute"
7. See the RTTM results in the response!

### Using cURL
```bash
curl -X POST "http://localhost:8000/diarize" \
  -F "audio_file=@audio.wav" \
  -F "num_speakers=2"
```

### Using PowerShell
```powershell
$form = @{
    audio_file = Get-Item "audio.wav"
    num_speakers = 2
}
Invoke-RestMethod -Uri "http://localhost:8000/diarize" -Method Post -Form $form
```

## 🐳 Docker Alternative

If you prefer Docker:

```powershell
# Build and run with Docker Compose
docker-compose up --build

# The API will be available at http://localhost:8000
```

## ❓ Common Issues

### NeMo Installation Issues
If you get errors installing NeMo, make sure PyTorch is installed first:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install nemo_toolkit[all] --extra-index-url https://pypi.ngc.nvidia.com
```

### Audio Format Issues
If your audio file format isn't supported, convert it to WAV:
```python
from pydub import AudioSegment
audio = AudioSegment.from_file("input.mp3")
audio.export("output.wav", format="wav")
```

### First Run is Slow
The first time you run diarization, NeMo will download required models. This is normal and only happens once.

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the API docs at http://localhost:8000/docs
- Test with your own audio files!

## 🎯 What is RTTM?

RTTM (Rich Transcription Time Marked) format shows who spoke when:

```
SPEAKER audio 1 0.0 1.5 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER audio 1 1.5 2.3 <NA> <NA> speaker_1 <NA> <NA>
```

- `speaker_0` spoke from 0.0s to 1.5s (1.5 seconds duration)
- `speaker_1` spoke from 1.5s to 3.8s (2.3 seconds duration)

Enjoy your diarization API! 🎉

