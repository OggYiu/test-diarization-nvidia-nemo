# Enhanced Format Feature for STT Tab

## Overview
This feature adds metadata headers and RTTM timestamps to transcription results in the Auto-Diarize & Transcribe tab.

## Changes Made

### 1. New Checkbox in UI
- **Location**: STT tab input section
- **Label**: "📋 Enhanced format (metadata + timestamps)"
- **Default**: Off (unchecked)
- **Purpose**: Enables/disables the enhanced format feature

### 2. New Helper Functions

#### `parse_rttm_timestamps(rttm_content: str) -> dict`
- Parses RTTM content to extract start times for each speaker segment
- Returns a dictionary mapping segment index to (speaker_id, start_time)

#### `add_timestamps_to_conversation(conversation, rttm_timestamps, broker_name, client_name, broker_speaker_id) -> str`
- Adds timestamps from RTTM to each line of conversation
- Formats lines as: `- 經紀 Name (timestamp): text`
- Only applies to conversation lines (經紀/客戶)

### 3. Modified Function

#### `process_chop_and_transcribe()`
- Added `use_enhanced_format` parameter
- When enabled and metadata is available:
  - Adds metadata header with:
    - 對話時間 (conversation time in HKT)
    - 經紀 (broker name)
    - broker_id
    - 客戶 (client name)
    - client_id
  - Adds timestamps to each conversation line from RTTM data

## Output Format

### When Enhanced Format is DISABLED (Default)
```
經紀 Dickson Lau: 请到时点啊。
客戶 CHENG SUK HING: 刘生啊，我想买腾讯个轮啊买个声得唔得啊嗯。
```

### When Enhanced Format is ENABLED
```
對話時間: 2025-10-20T10:01:20
經紀: Dickson Lau
broker_id: 0489
客戶: CHENG SUK HING
client_id: P77197

- 經紀 Dickson Lau (0.6): 请到时点啊。
- 客戶 CHENG SUK HING (1.75): 刘生啊，我想买腾讯个轮啊买个声得唔得啊嗯。
```

## Technical Details

### RTTM Format Used
The RTTM file format:
```
SPEAKER filename channel start_time duration <NA> <NA> speaker_id <NA> <NA>
```
- Column 3: Start time (in seconds)
- Column 7: Speaker ID (e.g., speaker_0, speaker_1)

### Timestamp Extraction
- Timestamps are extracted from the RTTM content
- Each segment's start time is mapped to the corresponding transcription line
- The order of segments in RTTM matches the order of chopped audio segments

### Metadata Extraction
- Metadata is extracted from the audio filename
- Format: `[Broker Name ID]_8330-97501167_YYYYMMDDHHMMSS(20981).wav`
- Client information is looked up from `client.csv` using phone number
- Time is converted from UTC to HKT (UTC+8)

## Usage

1. Upload an audio file in the "Auto-Diarize & Transcribe" tab
2. Check the "📋 Enhanced format (metadata + timestamps)" checkbox
3. Click "🎯 Auto-Diarize & Transcribe"
4. View results in the labeled transcription textboxes

## Files Modified

- `tabs/tab_stt.py`: Main implementation file
  - Added checkbox UI component
  - Added helper functions for timestamp parsing
  - Modified `process_chop_and_transcribe()` function
  - Updated button click handler to include new parameter

## Benefits

1. **Better Context**: Metadata at the top provides immediate context about the conversation
2. **Precise Timing**: Timestamps help identify when each speaker starts talking
3. **Flexible**: Can be toggled on/off based on user preference
4. **Backward Compatible**: Default is off, existing workflows are not affected
5. **Traditional Chinese**: SenseVoice results are automatically converted to Traditional Chinese using OpenCC

## Notes

- Enhanced format only applies when LLM speaker identification is successful
- Requires valid metadata in the filename
- Works with both SenseVoiceSmall and Whisper-v3-Cantonese models
- Timestamps are in seconds from the start of the audio file
- **SenseVoiceSmall results are automatically converted to Traditional Chinese** using OpenCC (s2t converter)
- Whisper-v3-Cantonese results are kept as-is (already in Traditional Chinese)

