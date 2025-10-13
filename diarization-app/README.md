# Speaker Diarization API

A FastAPI-based REST API for performing speaker diarization on audio files using NVIDIA NeMo.

## Features

- Upload audio files for speaker diarization
- Automatic speaker segmentation
- Returns RTTM (Rich Transcription Time Marked) format results
- RESTful API with JSON responses
- Built with FastAPI for high performance

## Installation

### 1. Install PyTorch

First, install PyTorch. Choose the appropriate version for your system:

**CPU-only (Windows):**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**CUDA (if you have NVIDIA GPU):**
```powershell
# For CUDA 11.8 (adjust version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install NVIDIA NeMo

```powershell
pip install nemo_toolkit[all] --extra-index-url https://pypi.ngc.nvidia.com
```

### 3. Install API Requirements

```powershell
pip install -r requirements.txt
```

## Running the API

### Development Mode

```powershell
# Navigate to the diarization folder
cd diarization

# Run the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or simply:

```powershell
python app/main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- **Interactive API docs (Swagger UI):** http://localhost:8000/docs
- **Alternative docs (ReDoc):** http://localhost:8000/redoc

## API Endpoints

### Health Check
```
GET /
```
Returns API status and available endpoints.

### Diarize Audio
```
POST /diarize
```

Upload an audio file for speaker diarization.

**Request:**
- `audio_file` (file): Audio file to process (WAV, MP3, etc.)
- `num_speakers` (form data, optional): Expected number of speakers (default: 2)

**Response:**
```json
{
  "status": "success",
  "audio_filename": "conversation.wav",
  "num_speakers": 2,
  "rttm_content": "SPEAKER conversation 1 0.0 1.5 <NA> <NA> speaker_0 <NA> <NA>\nSPEAKER conversation 1 1.5 2.3 <NA> <NA> speaker_1 <NA> <NA>\n...",
  "message": "Diarization completed successfully"
}
```

## Usage Examples

### Using cURL

```bash
curl -X POST "http://localhost:8000/diarize" \
  -F "audio_file=@path/to/your/audio.wav" \
  -F "num_speakers=2"
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/diarize"

with open("audio.wav", "rb") as f:
    files = {"audio_file": f}
    data = {"num_speakers": 2}
    response = requests.post(url, files=files, data=data)

print(response.json())
```

### Using PowerShell

```powershell
$uri = "http://localhost:8000/diarize"
$audioFile = "C:\path\to\audio.wav"

$form = @{
    audio_file = Get-Item -Path $audioFile
    num_speakers = 2
}

$response = Invoke-RestMethod -Uri $uri -Method Post -Form $form
$response | ConvertTo-Json
```

## RTTM Format

The RTTM (Rich Transcription Time Marked) format contains speaker segmentation information:

```
SPEAKER <audio_filename> 1 <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
```

Example:
```
SPEAKER audio 1 0.0 1.5 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER audio 1 1.5 2.3 <NA> <NA> speaker_1 <NA> <NA>
```

This indicates:
- Speaker 0 spoke from 0.0s to 1.5s
- Speaker 1 spoke from 1.5s to 3.8s

## Project Structure

```
diarization/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   └── diarization_service.py   # Diarization logic
├── requirements.txt
└── README.md
```

## Notes

- The first run may take longer as NeMo downloads required models
- Supported audio formats: WAV (recommended), MP3, FLAC, etc.
- Processing time depends on audio length and system resources
- Default configuration uses CPU; GPU can be enabled by modifying the config in `diarization_service.py`

## Troubleshooting

### ModuleNotFoundError: nemo
Make sure you've installed NeMo after PyTorch:
```powershell
pip install nemo_toolkit[all] --extra-index-url https://pypi.ngc.nvidia.com
```

### Audio file format errors
Convert your audio to WAV format if you encounter issues:
```python
from pydub import AudioSegment
audio = AudioSegment.from_file("input.mp3")
audio.export("output.wav", format="wav")
```

### Memory issues
For large audio files, consider:
- Using a system with more RAM
- Splitting the audio into smaller chunks
- Enabling GPU processing if available

## License

This project uses NVIDIA NeMo, which has its own licensing terms. Please refer to the NeMo documentation for details.

