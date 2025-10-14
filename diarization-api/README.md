# Diarization API

A FastAPI-based service for audio processing that performs speaker diarization, audio segmentation, speech-to-text transcription, and optional LLM analysis.

## Features

- **Speaker Diarization**: Identify and separate different speakers using NeMo's ClusteringDiarizer
- **Audio Segmentation**: Automatically chop audio into speaker-specific segments
- **Speech-to-Text**: Transcribe audio using SenseVoice (FunASR)
- **LLM Analysis**: Optional conversation analysis using Ollama (supports Cantonese and stock trading context)
- **Asynchronous Processing**: Non-blocking job processing with status tracking
- **REST API**: Simple HTTP endpoints for easy integration

## Requirements

- Python 3.8+
- PyTorch (CPU or GPU)
- Docker (optional, for containerized deployment)
- Ollama server (optional, for LLM analysis)

## Installation

### Option 1: Local Installation

1. **Install PyTorch** (choose CPU or CUDA based on your system):

```bash
# CPU version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or CUDA version (if you have NVIDIA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. **Install dependencies**:

```bash
cd diarization-api
pip install -r requirements.txt
```

3. **Install NeMo toolkit**:

```bash
pip install nemo_toolkit[all]
```

### Option 2: Docker

```bash
cd diarization-api
docker build -t diarization-api .
docker run -p 8000:8000 diarization-api
```

## Quick Start

### Start the API Server

```bash
cd diarization-api
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Basic Usage

1. **Upload an audio file for processing**:

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@your_audio.wav" \
  -F "num_speakers=2" \
  -F "skip_llm=true"
```

2. **Check the job status**:

```bash
curl "http://localhost:8000/status/{job_id}"
```

## API Endpoints

### `GET /`

Health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "message": "Diarization API up"
}
```

---

### `POST /process`

Submit an audio file for processing.

**Parameters**:
- `file` (file, required): Audio file to process (WAV format recommended)
- `num_speakers` (int, optional): Number of speakers in the audio (default: 2)
- `skip_llm` (bool, optional): Skip LLM analysis step (default: false)
- `llm_model` (str, optional): Ollama model name (default: "deepsek-r1:32b")
- `ollama_url` (str, optional): Ollama server URL (default: "http://192.168.61.2:11434")

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status_endpoint": "/status/550e8400-e29b-41d4-a716-446655440000"
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@phone_call.wav" \
  -F "num_speakers=2" \
  -F "skip_llm=false" \
  -F "llm_model=llama2" \
  -F "ollama_url=http://localhost:11434"
```

---

### `GET /status/{job_id}`

Check the status of a processing job.

**Path Parameters**:
- `job_id` (str, required): The job ID returned from `/process`

**Response** (job running):
```json
{
  "status": "running"
}
```

**Response** (job complete):
```json
{
  "status": "done",
  "result": {
    "rttm_file": "/path/to/output.rttm",
    "chopped_files": [
      "/path/to/segment_001.wav",
      "/path/to/segment_002.wav"
    ],
    "transcriptions": [
      {
        "file": "segment_001.wav",
        "segment_num": 1,
        "speaker": "speaker_0",
        "start": 0.0,
        "end": 5.2,
        "duration": 5.2,
        "transcription": "Hello, this is the transcription"
      }
    ],
    "analysis": "LLM analysis result here...",
    "duration_seconds": 45.3
  }
}
```

**Response** (job failed):
```json
{
  "status": "error",
  "error": "Error message here"
}
```

## Python Client Example

```python
import requests
import time

# Upload audio file
with open("phone_call.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/process",
        files={"file": f},
        data={
            "num_speakers": 2,
            "skip_llm": False,
            "llm_model": "llama2",
            "ollama_url": "http://localhost:11434"
        }
    )

job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")

# Poll for status
while True:
    status_response = requests.get(f"http://localhost:8000/status/{job_id}")
    status_data = status_response.json()
    
    if status_data["status"] == "done":
        print("Processing complete!")
        print(f"Transcriptions: {status_data['result']['transcriptions']}")
        break
    elif status_data["status"] == "error":
        print(f"Error: {status_data['error']}")
        break
    else:
        print("Processing...")
        time.sleep(5)
```

## JavaScript/Node.js Client Example

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function processAudio() {
  // Upload audio file
  const form = new FormData();
  form.append('file', fs.createReadStream('phone_call.wav'));
  form.append('num_speakers', '2');
  form.append('skip_llm', 'false');
  
  const uploadResponse = await axios.post('http://localhost:8000/process', form, {
    headers: form.getHeaders()
  });
  
  const jobId = uploadResponse.data.job_id;
  console.log(`Job ID: ${jobId}`);
  
  // Poll for status
  while (true) {
    const statusResponse = await axios.get(`http://localhost:8000/status/${jobId}`);
    const status = statusResponse.data;
    
    if (status.status === 'done') {
      console.log('Processing complete!');
      console.log('Transcriptions:', status.result.transcriptions);
      break;
    } else if (status.status === 'error') {
      console.error('Error:', status.error);
      break;
    } else {
      console.log('Processing...');
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
}

processAudio();
```

## Command-Line Tool

You can also use the pipeline directly from the command line:

```bash
# Single file processing
python audio_pipeline.py your_audio.wav --num-speakers 2 --skip-llm

# Batch processing
python batch_process.py /path/to/audio/files --output-dir ./results --num-speakers 2
```

### Command-Line Options

```
audio_pipeline.py:
  audio_file              Path to the audio file to process
  --work-dir DIR         Working directory for outputs (default: ./output)
  --num-speakers N       Number of speakers (default: 2)
  --padding MS           Padding in milliseconds (default: 100)
  --language LANG        Language for STT (auto/zh/en/yue/ja/ko)
  --skip-llm             Skip LLM analysis step
  --llm-model MODEL      Ollama model name (default: deepsek-r1:32b)
  --ollama-url URL       Ollama server URL
  --prompt TEXT          Custom system prompt for LLM
```

## Output Structure

After processing, the working directory will contain:

```
work_directory/
├── diarization/
│   ├── pred_rttms/
│   │   └── audio.rttm              # Speaker diarization results
│   ├── speaker_outputs/            # Speaker embeddings
│   └── vad_outputs/                # Voice Activity Detection outputs
├── audio_chunks/
│   ├── segment_001.wav             # Chopped audio segments
│   ├── segment_002.wav
│   └── ...
├── transcriptions/
│   ├── transcriptions.json         # Detailed transcriptions
│   └── conversation.txt            # Simple conversation format
├── analysis/
│   └── analysis.txt                # LLM analysis (if enabled)
└── pipeline_summary.json           # Overall summary
```

## Configuration

### Number of Speakers

The API uses speaker diarization to identify different speakers. Set `num_speakers` based on your audio:

- Phone calls: typically 2
- Meetings: 3-8 speakers
- If unsure, the system can auto-detect (set `oracle_num_speakers: false` in config)

### LLM Analysis

The LLM analysis step is designed for Cantonese stock trading conversations but can be customized:

1. **Skip LLM**: Set `skip_llm=true` to disable analysis
2. **Custom Model**: Use any Ollama model via `llm_model` parameter
3. **Custom Prompt**: Provide a custom system prompt for different use cases

### Ollama Setup

If using LLM analysis, you need an Ollama server:

1. Install Ollama: https://ollama.ai/
2. Start the server: `ollama serve`
3. Pull a model: `ollama pull llama2`
4. Point the API to your Ollama URL

## Performance Considerations

- **Processing Time**: Expect ~1-2 minutes per minute of audio (varies by hardware)
- **Memory**: Minimum 4GB RAM recommended, 8GB+ for larger files
- **GPU**: Optional but significantly speeds up processing
- **Audio Format**: WAV files work best; other formats may require conversion

## Troubleshooting

### "Module 'nemo' is not installed"

```bash
pip install nemo_toolkit[all]
```

### "Module 'funasr' is not installed"

```bash
pip install funasr
```

### "Ollama connection error"

- Ensure Ollama server is running
- Check the `ollama_url` parameter matches your server
- Test with: `curl http://localhost:11434/api/tags`

### "No .rttm file found"

- Check audio file format (WAV recommended)
- Ensure audio has speech content
- Try increasing `num_speakers`

### API Returns 404 for Job Status

- The API uses in-memory job storage
- Jobs are lost if server restarts
- For production, implement persistent storage

## Development

### Running Tests

```bash
# Check installation
python check_installation.py

# Test with sample audio
python audio_pipeline.py demo/phone_recordings/sample.wav
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

This project uses several open-source components:
- NeMo Toolkit (Apache 2.0)
- FunASR (MIT)
- FastAPI (MIT)

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review output logs in the work directory
3. Examine `error.log` if present

## Acknowledgments

- **NeMo Toolkit**: NVIDIA's framework for conversational AI
- **SenseVoice**: Alibaba's multilingual speech recognition model
- **FastAPI**: Modern Python web framework

