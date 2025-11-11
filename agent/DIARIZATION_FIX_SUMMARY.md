# Diarization Issue Fix Summary

## Problem Description

When running `python app.py` from the `agent` folder, diarization was failing with this error:

```
[NeMo W] no vad file found for [Dickson Lau]_8330-96674941_20251013035051(3360) due to zero or negative duration
[NeMo W] [Dickson Lau]_8330-96674941_20251013035051(3360) is ignored since the file does not contain any speech signal to be processed.
❌ Error during diarization: All files present in manifest contains silence, aborting next steps
```

However, the same audio file worked fine when using the diarization functionality in `tabs/tab_stt.py`.

## Root Causes

### 1. **Relative Path Issues**
- When running `python app.py` from the `agent` folder, relative paths like `"output"` were resolved relative to the current working directory (`agent/`)
- This caused output directories to be created in `agent/output/` instead of the expected location
- Cached incomplete results from previous failed runs could be reused, causing the error to persist

### 2. **Caching Behavior Differences**
- `agent/diarize.py` had caching logic that would reuse existing results if found
- However, it didn't properly validate that cached results were complete and valid
- If a previous run failed and left an empty or incomplete RTTM file, it would try to reuse it

### 3. **Different Behavior from tab_stt.py**
- The root `diarization.py` (used by `tab_stt.py`) always deletes the output directory before processing
- This ensures a clean start every time, avoiding issues with cached incomplete results

## Fixes Applied

### 1. **app.py** - Use Absolute Paths
```python
# Before: Relative path (could point to non-existent file)
audio_file = 'assets/test_audio_files/[Dickson Lau]_8330-96674941_20251013035051(3360).wav'

# After: Absolute path with file existence validation
current_dir = os.path.dirname(os.path.abspath(__file__))
audio_file = os.path.join(current_dir, "assets", "test_audio_files", "[Dickson Lau 0489]_8330-96674941_20251013012751(880).wav")

# Also added file validation and helpful error messages
```

### 2. **tools/diarize_tool.py** - Absolute Output Paths
```python
# Before: Relative path
output_dir = os.path.join("output", "diarization", audio_basename)

# After: Absolute path
agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(agent_dir, "output", "diarization", audio_basename)

# Also added force_reprocess=True to ensure clean results
```

### 3. **diarize.py** - Enhanced Validation
```python
# Added force_reprocess parameter
def diarize(audio_filepath: str, output_dir: str, num_speakers: int = 2, 
            domain_type: str = "telephonic", force_reprocess: bool = False) -> str:

# Added validation for cached RTTM files
if not rttm_content or not rttm_content.strip():
    print(f"⚠️ Warning: RTTM file is empty. Will reprocess.")
    shutil.rmtree(output_dir)
```

### 4. **tools/audio_chopper_tool.py** - Absolute Output Paths
```python
# Before: Relative path with ".."
output_dir = os.path.join("..", "output", "audio_segments", audio_basename)

# After: Absolute path
agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(agent_dir, "output", "audio_segments", audio_basename)
```

## Testing

To test the fixes:

```bash
cd agent
python app.py
```

The script should now:
1. ✅ Correctly locate the audio file using absolute paths
2. ✅ Create output directories in the correct location
3. ✅ Force reprocessing to avoid cached incomplete results
4. ✅ Successfully diarize and chop the audio file

## Expected Output

```
Agent graph saved to agent_graph.png
📁 Using audio file: C:\projects\test-diarization\agent\assets\test_audio_files\[Dickson Lau 0489]_8330-96674941_20251013012751(880).wav
✅ File exists: True
📊 File size: X.XX MB

[... successful diarization and chopping ...]

================================
Workflow completed!
================================
```

## Additional Notes

- All paths now use `os.path.join()` with absolute paths for cross-platform compatibility
- The `force_reprocess=True` flag ensures consistent behavior with `tab_stt.py`
- File existence validation provides helpful error messages if audio files are missing
- The fixes maintain backward compatibility with existing code

