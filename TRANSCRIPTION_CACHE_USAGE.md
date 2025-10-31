# Transcription Cache Usage Guide

## Overview
The transcription caching feature automatically saves and reuses transcription results for both SenseVoiceSmall and Whisper-v3-Cantonese models. This significantly speeds up processing when working with the same audio files multiple times.

## How It Works

### Automatic Caching
When you transcribe an audio file:
1. **First time**: The system transcribes the file and saves the result to MongoDB
2. **Subsequent times**: The system loads the cached result instantly (no model inference needed)

### Cache Indicators
In the status output, you'll see:
- **✅ SenseVoice**: New transcription (processed with model)
- **💾 SenseVoice (cached)**: Loaded from cache (instant)
- **Summary**: Shows cache hit rate, e.g., `✅ SenseVoice completed: 10/10 files (7 from cache)`

## Usage Examples

### Example 1: Auto-Diarize & Transcribe Tab
```
1. Upload an audio file (e.g., conversation.wav)
2. Select both models: SenseVoiceSmall ✓ and Whisper-v3-Cantonese ✓
3. Click "🎯 Auto-Diarize & Transcribe"
4. System processes and caches results

First run output:
  [1/10] segment_001.wav
    ✅ SenseVoice: 你好，我係...
  [2/10] segment_002.wav
    ✅ SenseVoice: 今日天氣...
  ...
  ✅ SenseVoice completed: 10/10 files (0 from cache)

5. Process the SAME file again
6. Click "🎯 Auto-Diarize & Transcribe"

Second run output:
  [1/10] segment_001.wav
    💾 SenseVoice (cached): 你好，我係...
  [2/10] segment_002.wav
    💾 SenseVoice (cached): 今日天氣...
  ...
  ✅ SenseVoice completed: 10/10 files (10 from cache)
  
Processing time: ~50x faster!
```

### Example 2: Batch Transcription
```
1. Upload multiple audio files or a ZIP file
2. Select models and process
3. Results are cached for each individual file
4. Re-process same files → instant results from cache
```

## Cache Behavior

### What Gets Cached
- **Filename**: Used as the cache key
- **Transcription**: The formatted transcription text
- **Raw Transcription**: The original unformatted text
- **Language**: (SenseVoice only) The language parameter used
- **Processing Time**: How long the original transcription took
- **Timestamp**: When the transcription was cached
- **Model**: Which model produced the transcription

### Cache Validation
- **SenseVoiceSmall**: Cache is used only if the filename AND language match
- **Whisper-v3-Cantonese**: Cache is used if the filename matches

### When Cache is Bypassed
- Different filename (even if content is identical)
- Different language parameter (SenseVoice only)
- Cache parameter set to `use_cache=False` (in code)

## Managing the Cache

### View Cache Statistics
Run the test script to see current cache stats:
```bash
python test_transcription_cache.py
```

### Clear Cache (if needed)
You can clear the cache using MongoDB commands or Python:

#### Option 1: Using MongoDB Shell
```javascript
// Connect to MongoDB
mongo

// Use the audio_processing database
use audio_processing

// View all cached SenseVoice transcriptions
db.transcriptions_sensevoice.find()

// View all cached Whisper-v3 transcriptions
db.transcriptions_whisperv3_cantonese.find()

// Clear all SenseVoice cache
db.transcriptions_sensevoice.deleteMany({})

// Clear all Whisper-v3 cache
db.transcriptions_whisperv3_cantonese.deleteMany({})

// Clear a specific file from cache
db.transcriptions_sensevoice.deleteOne({filename: "segment_001.wav"})
```

#### Option 2: Using Python
```python
from mongodb_utils import delete_from_mongodb
from tabs.tab_stt import (
    TRANSCRIPTION_SENSEVOICE_COLLECTION,
    TRANSCRIPTION_WHISPERV3_COLLECTION
)

# Clear all SenseVoice cache
delete_from_mongodb(TRANSCRIPTION_SENSEVOICE_COLLECTION, {})

# Clear all Whisper-v3 cache
delete_from_mongodb(TRANSCRIPTION_WHISPERV3_COLLECTION, {})

# Clear a specific file
delete_from_mongodb(
    TRANSCRIPTION_SENSEVOICE_COLLECTION, 
    {'filename': 'segment_001.wav'}
)
```

### View Cached Transcriptions
```python
from tabs.tab_stt import (
    load_transcription_cache_sensevoice,
    load_transcription_cache_whisperv3
)

# Load and display SenseVoice cache
sensevoice_cache = load_transcription_cache_sensevoice()
for filename, data in sensevoice_cache.items():
    print(f"{filename}: {data['transcription'][:50]}...")

# Load and display Whisper-v3 cache
whisperv3_cache = load_transcription_cache_whisperv3()
for filename, data in whisperv3_cache.items():
    print(f"{filename}: {data['transcription'][:50]}...")
```

## Performance Benefits

### Without Cache
- Processing 10 audio segments: ~30-60 seconds
- Re-processing same 10 segments: ~30-60 seconds each time

### With Cache
- Processing 10 audio segments (first time): ~30-60 seconds
- Re-processing same 10 segments: **~1 second** (50-60x faster!)

### Best Use Cases
1. **Iterative Development**: Testing different pipeline stages with same audio
2. **Re-running Analysis**: Need to re-process for comparison or debugging
3. **Batch Processing**: Processing similar/duplicate files in large datasets
4. **A/B Testing**: Comparing different models on the same audio files

## Important Notes

⚠️ **Cache Key Based on Filename Only**
- If you re-upload a file with the same name but different content, the cached result will be returned
- To force re-transcription, rename the file or clear the cache for that filename

⚠️ **Storage Space**
- Each cached transcription takes minimal space (typically < 1KB per file)
- Even thousands of cached transcriptions use negligible disk space
- MongoDB handles this efficiently

✅ **Automatic Updates**
- Cache entries are automatically updated (upserted) if you re-transcribe
- No need to manually clear cache before re-processing

✅ **Separate Storage**
- SenseVoice and Whisper-v3 caches are completely independent
- You can clear one without affecting the other

## Testing the Cache

Run the included test script to verify the cache is working:

```bash
python test_transcription_cache.py
```

Expected output:
```
Transcription Cache Test Suite
============================================================
Testing SenseVoiceSmall Cache
============================================================

1. Saving test transcription to cache...
✓ Saved document to MongoDB collection 'transcriptions_sensevoice'

2. Loading from cache...
✅ Cache entry found!
   Transcription: 這是一個測試
   Language: yue
   Processing time: 2.5s
   Timestamp: 2024-10-31 14:30:00

3. Total SenseVoice cache entries: 1

4. Test data cleaned up
...
✅ All tests completed successfully!
```

## Troubleshooting

### Cache Not Working?
1. **Check MongoDB Connection**: Ensure MongoDB is running on `localhost:27017`
2. **Check Database**: Verify the `audio_processing` database exists
3. **Check Collections**: Collections are auto-created on first save
4. **Run Test Script**: `python test_transcription_cache.py` to verify functionality

### Cache Taking Too Much Space?
- Cache typically uses minimal space
- Clear old entries if needed using MongoDB commands
- Consider implementing TTL (time-to-live) if needed

### Want to Disable Cache?
- Modify the transcription function calls to use `use_cache=False`
- Not recommended, as cache significantly improves performance

