# Transcription Caching Implementation

## Overview
This document describes the implementation of MongoDB caching for SenseVoiceSmall and Whisper-v3-Cantonese transcriptions.

## Changes Made

### 1. New MongoDB Collections
Two new MongoDB collections have been added to store transcriptions separately:

- **`transcriptions_sensevoice`**: Stores SenseVoiceSmall transcription results
- **`transcriptions_whisperv3_cantonese`**: Stores Whisper-v3-Cantonese transcription results

### 2. Cache Functions

#### SenseVoiceSmall Functions:
- **`load_transcription_cache_sensevoice()`**: Loads all cached SenseVoice transcriptions from MongoDB
- **`save_transcription_to_cache_sensevoice()`**: Saves a SenseVoice transcription result to MongoDB

#### Whisper-v3-Cantonese Functions:
- **`load_transcription_cache_whisperv3()`**: Loads all cached Whisper-v3 transcriptions from MongoDB
- **`save_transcription_to_cache_whisperv3()`**: Saves a Whisper-v3 transcription result to MongoDB

### 3. Updated Transcription Functions

Both `transcribe_single_audio_sensevoice()` and `transcribe_single_audio_whisperv3_cantonese()` have been updated to:

1. **Check cache first**: Before transcribing, the function checks if the filename already exists in the cache
2. **Return cached result**: If found, returns the cached transcription immediately (with `cache_hit: True` flag)
3. **Save to cache**: After successful transcription, saves the result to MongoDB for future use
4. **Track processing time**: Records how long each transcription took

### 4. Cache Key
- **Primary key**: `filename` (the base filename of the audio file)
- For SenseVoiceSmall, cache also validates that the `language` parameter matches

### 5. Cached Data Structure

#### SenseVoiceSmall Cache Document:
```json
{
  "filename": "segment_001.wav",
  "transcription": "formatted transcription text",
  "raw_transcription": "raw transcription text",
  "language": "yue",
  "processing_time": 2.45,
  "timestamp": "2024-10-31 14:30:00",
  "model": "SenseVoiceSmall"
}
```

#### Whisper-v3-Cantonese Cache Document:
```json
{
  "filename": "segment_001.wav",
  "transcription": "transcription text",
  "raw_transcription": "transcription text",
  "processing_time": 3.21,
  "timestamp": "2024-10-31 14:30:00",
  "model": "Whisper-v3-Cantonese"
}
```

### 6. Status Output Improvements

The status output now shows:
- **💾 emoji**: Indicates when a result came from cache
- **Cache hit count**: Summary shows how many files were loaded from cache vs. newly transcribed
- Example: `✅ SenseVoice completed: 10/10 files (7 from cache)`

## Benefits

1. **Faster Processing**: Subsequent transcriptions of the same audio files are instant (no model inference needed)
2. **Resource Savings**: Reduces GPU/CPU usage by avoiding redundant transcriptions
3. **Separate Storage**: Each model's transcriptions are stored independently, allowing different models to be compared
4. **Persistent Cache**: Cache persists across application restarts
5. **Automatic Upsert**: If the same file is re-transcribed, the cache is automatically updated

## Usage

The caching is **automatic** and **transparent** to the user:

1. First time transcribing a file: Normal processing, result saved to cache
2. Subsequent times: Result loaded from cache instantly
3. Status messages clearly indicate cache hits vs. new transcriptions

## Cache Management

To clear the cache (if needed), you can use MongoDB commands:

```javascript
// Clear all SenseVoice transcriptions
db.transcriptions_sensevoice.deleteMany({})

// Clear all Whisper-v3-Cantonese transcriptions
db.transcriptions_whisperv3_cantonese.deleteMany({})

// Clear a specific file
db.transcriptions_sensevoice.deleteOne({filename: "segment_001.wav"})
```

## Notes

- Cache is stored in the `audio_processing` MongoDB database
- The cache key is based on filename only, not file contents or path
- If you re-upload a file with the same name but different content, the cache will return the old result
- For SenseVoiceSmall, changing the language parameter will bypass the cache and re-transcribe

## Integration Points

The caching is integrated into:
- **Auto-Diarize & Transcribe** tab (chopped segments)
- **Batch Speech-to-Text** tab (multiple audio files)

Both workflows benefit from transcription caching.

