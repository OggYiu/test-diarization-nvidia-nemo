# Path Protection Fix - Preventing Output in Source Folders

## Problems

### Problem 1: Output Created in Source Directory

The audio processing agent was sometimes creating output folders (like `chopped_segments`, `diarization`, `transcriptions`) in the source audio directory instead of in the designated `agent/output/` directory structure.

### Problem 2: Incorrect Folder Names - "chopped" instead of "chopped_segments/[filename]"

Old code was creating a flat `agent/output/chopped/` folder containing segments from multiple audio files mixed together, instead of organizing them properly in `agent/output/chopped_segments/[audio_filename]/`.

### Problem 3: Transcription Folders with .wav_segments Suffix

Transcription folders were being created with names like `[filename].wav_segments` instead of just `[filename]`, making it hard to match with the original audio file.

### Root Causes

The tools (`diarize_audio`, `chop_audio_by_rttm`, `transcribe_audio_segments`) were designed to automatically determine output paths, but:

1. **`chop_audio_by_rttm` had an optional `output_dir` parameter** that could be overridden by the LLM agent
2. The LLM could potentially pass incorrect paths, causing output to be created in the source folder
3. There were no validation checks to prevent creating folders in the source directory
4. **`transcribe_audio_segments` used the segments directory name directly** without cleaning up unwanted suffixes
5. Old versions of the code used different path conventions

## Solution

### 1. Removed Optional Output Parameter

**File: `agent/tools/audio_chopper_tool.py`**

- **Removed** the `output_dir` parameter from the `chop_audio_by_rttm` tool signature
- The tool now **always** determines the output directory internally based on the audio filename
- Output is guaranteed to be in: `agent/output/chopped_segments/[audio_filename]/`

### 2. Added Path Validation

**Files: `agent/tools/audio_chopper_tool.py`, `agent/tools/diarize_tool.py`, `agent/tools/stt_tool.py`**

Added validation logic to all tools to prevent creating output in source directories:

```python
# Validate that output_dir is NOT in the source audio directory
audio_dir = os.path.dirname(os.path.abspath(audio_filepath))
output_dir_abs = os.path.abspath(output_dir)
if output_dir_abs.startswith(audio_dir):
    # This would create output in the source folder - prevent it!
    output_dir = os.path.join(agent_dir, "output", "chopped_segments", audio_basename)
    print(f"⚠️  Prevented creating output in source directory")
```

### 3. Added Suffix Cleaning

**File: `agent/tools/stt_tool.py`**

Added logic to clean up unwanted suffixes from folder names:

```python
suffixes_to_remove = ['.wav_segments', '.mp3_segments', '.flac_segments', '.m4a_segments', '.ogg_segments']
for suffix in suffixes_to_remove:
    if segments_folder_name.endswith(suffix):
        segments_folder_name = segments_folder_name[:-len(suffix)]
        print(f"🔧 Cleaned folder name suffix: {suffix}")
        break
```

This ensures that even if the LLM passes a path with unwanted suffixes, the transcription folder will have the correct name.

### 4. Updated System Prompt

**File: `agent/app.py`**

Updated the LLM system prompt to be more explicit about:
- Which parameters to pass to each tool
- That output directories are automatically determined
- Not to manually specify output paths

New prompt includes:
```
IMPORTANT: Never specify output directories manually. All tools automatically organize outputs into agent/output/ subdirectories.
```

## Output Directory Structure

All audio processing outputs are now guaranteed to be organized as follows:

```
agent/
├── output/
│   ├── diarization/
│   │   └── [audio_filename]/
│   │       ├── pred_rttms/
│   │       │   └── diarization.rttm
│   │       ├── speaker_outputs/
│   │       └── vad_outputs/
│   ├── chopped_segments/
│   │   └── [audio_filename]/
│   │       ├── segment_001_*.wav
│   │       ├── segment_002_*.wav
│   │       └── ...
│   └── transcriptions/
│       └── [audio_filename]/
│           ├── transcriptions.json
│           └── transcriptions_text.txt
└── assets/
    └── phone-recordings/  (SOURCE - never modified)
        └── [audio files]
```

## Benefits

1. **Clean Separation**: Source audio files remain untouched in `assets/phone-recordings/`
2. **Organized Outputs**: All processing outputs are in `agent/output/` subdirectories
3. **Prevention**: Validation checks prevent accidental creation of folders in wrong locations
4. **Clarity**: LLM system prompt explicitly guides correct tool usage
5. **Consistency**: All tools follow the same output path pattern

## Cleanup Script

**File: `agent/cleanup_old_folders.py`**

A cleanup script is provided to fix folders created by old code versions:

```bash
cd agent
python cleanup_old_folders.py
```

The script will:
1. ✅ Remove `agent/output/chopped/` folder (wrong location from old code)
2. ✅ Rename transcription folders with `.wav_segments` suffix to remove the suffix
3. ✅ Remove generic transcription folders like `chopped`, `chopped_segments`

The script is interactive and will ask for confirmation before making changes.

## Testing

To verify the fix works:

1. Run the agent on audio files in `agent/assets/phone-recordings/`
2. Verify that no output folders are created in the `phone-recordings` directory
3. Verify that all outputs appear in `agent/output/` subdirectories
4. Check for warning messages if the validation logic is triggered
5. Run the cleanup script to fix any old folders

## Example Issues Fixed

### Before (Old Code):
```
agent/output/
├── chopped/  ❌ Wrong! Flat structure, mixed files
│   ├── [File1]_segment_001_*.wav
│   ├── [File2]_segment_001_*.wav
│   └── ...
└── transcriptions/
    ├── [File1].wav_segments/  ❌ Wrong! Has .wav_segments suffix
    │   └── transcriptions.json
    └── chopped/  ❌ Wrong! Generic name
        └── transcriptions.json
```

### After (Fixed Code):
```
agent/output/
├── chopped_segments/  ✅ Correct! Organized by filename
│   ├── [File1]/
│   │   ├── segment_001_*.wav
│   │   └── ...
│   └── [File2]/
│       ├── segment_001_*.wav
│       └── ...
└── transcriptions/  ✅ Correct! Matches audio filenames
    ├── [File1]/
    │   └── transcriptions.json
    └── [File2]/
        └── transcriptions.json
```

## Modified Files

- `agent/tools/audio_chopper_tool.py` - Removed `output_dir` parameter, added validation
- `agent/tools/diarize_tool.py` - Added validation to prevent source folder pollution
- `agent/tools/stt_tool.py` - Added validation and suffix cleaning
- `agent/app.py` - Updated system prompt for clarity
- `agent/cleanup_old_folders.py` - NEW: Interactive cleanup script for old folders

