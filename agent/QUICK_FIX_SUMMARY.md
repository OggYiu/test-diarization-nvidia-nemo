# Quick Fix Summary - Folder Naming Issues

## Issues You Reported

1. ❌ **`agent/output/chopped/` folder wrongly created** - Should be `chopped_segments/[filename]/`
2. ❌ **Transcription folder named `[filename].wav_segments`** - Should be just `[filename]`
3. ❌ **Missing transcription for `[Dickson Lau]_8330-96674941_20251020060841(2868).wav`**

## Root Cause

These issues were caused by:
- Old code versions that used different path conventions
- LLM agent potentially passing incorrect path parameters
- Lack of validation to prevent wrong folder locations

## Fixes Applied

### Code Changes

✅ **Removed `output_dir` parameter** from `chop_audio_by_rttm` tool - prevents LLM from specifying wrong paths

✅ **Added path validation** to all tools - prevents creating outputs in source directories

✅ **Added suffix cleaning** to transcription tool - removes `.wav_segments` and similar suffixes

✅ **Updated system prompt** - explicitly tells LLM not to specify output directories

### Files Modified

- `agent/tools/audio_chopper_tool.py` - Path validation + removed parameter
- `agent/tools/diarize_tool.py` - Path validation
- `agent/tools/stt_tool.py` - Path validation + suffix cleaning
- `agent/app.py` - Updated system prompt

## How to Fix Existing Folders

### Option 1: Run Cleanup Script (Recommended)

```bash
cd agent
python cleanup_old_folders.py
```

The script will:
- Remove `agent/output/chopped/` folder
- Rename `[filename].wav_segments/` to `[filename]/`
- Remove generic folders like `transcriptions/chopped/`

### Option 2: Manual Cleanup

1. **Delete old chopped folder:**
   ```bash
   rm -rf agent/output/chopped/
   ```

2. **Rename transcription folder:**
   ```bash
   cd agent/output/transcriptions/
   mv "[Dickson Lau 0489]_8330-96674941_20251014015606(8356).wav_segments" \
      "[Dickson Lau 0489]_8330-96674941_20251014015606(8356)"
   ```

3. **Remove generic folders:**
   ```bash
   rm -rf agent/output/transcriptions/chopped/
   rm -rf agent/output/transcriptions/chopped_segments/
   ```

## Correct Folder Structure

Going forward, all outputs will follow this structure:

```
agent/
├── output/
│   ├── diarization/
│   │   └── [audio_filename]/
│   │       └── pred_rttms/diarization.rttm
│   ├── chopped_segments/
│   │   └── [audio_filename]/
│   │       ├── segment_001_*.wav
│   │       └── segment_002_*.wav
│   └── transcriptions/
│       └── [audio_filename]/
│           ├── transcriptions.json
│           └── transcriptions_text.txt
└── assets/
    └── phone-recordings/  (SOURCE - never modified)
```

## Testing

To verify everything works:

1. ✅ Run the cleanup script to fix old folders
2. ✅ Process a new audio file through the agent
3. ✅ Verify folder structure matches the correct format above
4. ✅ Check that no folders are created in `assets/phone-recordings/`

## Questions?

See `PATH_PROTECTION_FIX.md` for detailed technical documentation.

