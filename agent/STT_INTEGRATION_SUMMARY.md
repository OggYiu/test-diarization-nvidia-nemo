# STT Integration Summary

## Overview
Added SenseVoiceSmall speech-to-text functionality to the agent workflow. After audio is chopped into speaker segments, the agent can now transcribe them to text.

## What Was Added

### 1. New STT Tool (`agent/tools/stt_tool.py`)
Copied the SenseVoiceSmall implementation from `tabs/tab_stt.py` (used in unified_gui.py) to create a standalone tool for the agent.

**Key Features:**
- Transcribes audio segments using SenseVoiceSmall model
- Supports GPU/CPU automatic detection
- Converts Simplified Chinese to Traditional Chinese (Hong Kong variant)
- Formats transcription output with `format_str_v3`
- Saves results to JSON file
- Supports hotwords for improved recognition
- Configurable VAD model and segment time

**Main Functions:**
- `initialize_sensevoice_model()` - Loads the SenseVoiceSmall model
- `transcribe_single_audio_sensevoice()` - Transcribes a single audio file
- `transcribe_audio_segments()` - LangChain tool that transcribes all segments in a directory

### 2. Updated Agent App (`agent/app.py`)
Added the STT tool to the agent's workflow:
- Imported `transcribe_audio_segments` from `tools.stt_tool`
- Added tool to the tools list
- Updated system message to include STT workflow step
- Updated example usage to demonstrate full pipeline

## Complete Workflow

The agent now supports a 3-step audio processing pipeline:

1. **Diarization** (`diarize_audio`)
   - Identifies speakers and when they spoke
   - Generates RTTM file with timestamps

2. **Audio Chopping** (`chop_audio_by_rttm`)
   - Splits audio into speaker segments
   - Uses RTTM data from diarization
   - Adds optional padding

3. **Transcription** (`transcribe_audio_segments`) ✨ NEW
   - Transcribes all chopped segments
   - Uses SenseVoiceSmall model
   - Outputs transcriptions with speaker labels
   - Saves to JSON file

## Usage Example

```python
from langchain.messages import HumanMessage

# The agent can now handle the full pipeline in one command
messages = [
    HumanMessage(
        content="Diarize the audio file 'path/to/audio.wav' with 2 speakers, "
                "then chop it into speaker segments with 50ms padding, "
                "and finally transcribe all the segments to text using SenseVoiceSmall."
    )
]

result = agent.invoke({"messages": messages})
```

## Tool Parameters

### `transcribe_audio_segments`
- `segments_directory` (required): Directory containing chopped audio segments
- `language` (optional): Language code, default "yue" (Cantonese)
- `hotwords` (optional): Space-separated hotwords to boost recognition
- `vad_model` (optional): VAD model to use, default "fsmn-vad"
- `max_single_segment_time` (optional): Max segment time in ms, default 30000

## Output Format

The tool creates:
1. **Console output**: Summary with speaker labels and transcriptions
2. **JSON file**: `transcriptions.json` in the segments directory

Example JSON structure:
```json
{
  "total_segments": 10,
  "total_processing_time": 5.23,
  "language": "yue",
  "transcriptions": [
    {
      "file": "speaker_0_segment_001.wav",
      "path": "/full/path/to/segment.wav",
      "transcription": "你好，我想開戶口",
      "raw_transcription": "你好，我想开户口",
      "processing_time": 0.52
    }
  ]
}
```

## Dependencies

The STT tool requires:
- `funasr` - For SenseVoiceSmall model
- `torch` - PyTorch for GPU/CPU handling
- `opencc-python-reimplemented` - For Traditional Chinese conversion
- `batch_stt.py` - For helper functions (`format_str_v3`, `load_audio`)

## Code Duplication Note

As requested, the code was copied from `tabs/tab_stt.py` to avoid modifying the existing unified_gui.py implementation. This ensures the original functionality remains untouched while providing the same STT capabilities to the agent workflow.

## Testing

Run the agent with:
```bash
cd agent
python app.py
```

The example at the bottom of `app.py` will:
1. Diarize a test audio file
2. Chop it into speaker segments
3. Transcribe all segments using SenseVoiceSmall

## Model Information

**SenseVoiceSmall**
- Model ID: `iic/SenseVoiceSmall`
- Provider: Alibaba DAMO Academy
- Language Support: Multilingual (optimized for Chinese/Cantonese)
- Features: VAD, emotion detection, speaker diarization
- Device: Automatic GPU/CPU detection

## Next Steps

You can now:
- Process audio files through the complete pipeline
- Get speaker-labeled transcriptions
- Chain results with downstream analysis tasks
- Customize language, hotwords, and VAD settings

