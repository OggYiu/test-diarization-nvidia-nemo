# Fix for Special Characters in Audio Filenames

## Problem

Audio files with special characters in their filenames (particularly square brackets `[` and `]`) were causing the diarization process to fail with the error:

```
[NeMo W] no vad file found for [Dickson Lau]_8330-96674941_20251013035051(3360) due to zero or negative duration
[NeMo W] [Dickson Lau]_8330-96674941_20251013035051(3360) is ignored since the file does not contain any speech signal to be processed.
❌ Error during diarization: All files present in manifest contains silence, aborting next steps
```

The issue occurs because NeMo (or one of its underlying libraries) treats square brackets as **glob pattern characters** rather than literal characters, causing the file path to be misinterpreted.

## Root Cause

- Square brackets `[]` are special characters used in glob patterns for character classes
- When NeMo's VAD (Voice Activity Detection) component tries to process the file, it interprets `[Dickson Lau]` as a glob pattern
- This causes the file lookup to fail, resulting in "zero or negative duration" errors
- The same audio file works fine with a simple filename like `1.wav`

## Solution

The fix implements a **two-part approach** to handle special characters:

### Implementation Details

1. **Audio File Sanitization**: 
   - **Detection**: Check if the filename contains problematic characters (currently `[` and `]`)
   - **Temporary Copy**: Create a copy with a timestamp-based sanitized filename
   - **Processing**: Use the temporary file for diarization
   - **Cleanup**: Remove the temporary file after processing completes (or on error)

2. **Output Directory Sanitization**:
   - **Directory Names**: Replace special characters (`[]<>:"|?*`) with underscores in output paths
   - This prevents issues with NeMo's internal file operations
   - Example: `[Dickson Lau]_8330-96674941_20251013035051(3360)` → `_Dickson Lau__8330-96674941_20251013035051(3360)`

### Modified Files

1. **`agent/diarize.py`** - Standalone diarization module
2. **`agent/tools/diarize_tool.py`** - LangChain tool wrapper

### Key Functions Added

```python
def has_special_chars(filename: str) -> bool:
    """Check if filename contains problematic characters."""
    special_chars = r'[\[\]]'
    return bool(re.search(special_chars, filename))

def create_temp_audio_copy(audio_filepath: str, temp_dir: str) -> tuple[str, str]:
    """Create a temporary copy with sanitized filename."""
    # Creates: temp_audio_{timestamp}.wav
    # Returns: (temp_filepath, original_basename)

def sanitize_directory_name(name: str) -> str:
    """Sanitize a directory name by removing problematic characters."""
    # Replaces: []<>:"|?* → underscores
    sanitized = re.sub(r'[\[\]<>:"|?*]', '_', name)
    return sanitized
```

### User Experience

When processing a file with special characters, users will see:

```
📂 Output directory: C:\...\output\diarization\_Dickson Lau__8330-96674941_20251013035051(3360)

⚠️  Detected special characters in filename: [Dickson Lau]_8330-96674941_20251013035051(3360).wav
🔧 Creating temporary copy with sanitized filename...
🔧 Created temporary copy with sanitized filename: temp_audio_1731314997000.wav
✅ Using temporary file: temp_audio_1731314997000.wav

[... diarization proceeds normally ...]

🧹 Cleaned up temporary audio file
```

Note: The output directory path now has brackets replaced with underscores to prevent path handling issues.

## Benefits

1. **Transparent**: Works automatically without user intervention
2. **Safe**: Original file is never modified
3. **Clean**: Temporary files are always cleaned up
4. **Extensible**: Easy to add more problematic characters to the detection regex
5. **Backward Compatible**: Files without special characters are processed normally

## Testing

To verify the fix works:

```bash
# This should now work without errors
python agent/app.py

# Or test directly:
python agent/diarize.py "path/to/[file with brackets].wav"
```

## Alternative Approaches Considered

1. **Escaping**: Tried to escape brackets in the path - didn't work because NeMo processes paths internally
2. **Symbolic Links**: Would work but requires admin rights on Windows
3. **Path Quoting**: NeMo's internal handling bypasses shell quoting
4. **Renaming Original**: Rejected to avoid modifying user files

## Future Enhancements

- Add more problematic characters if discovered: `()`, `{}`, `*`, `?`, etc.
- Add configuration option to disable automatic sanitization
- Cache sanitized copies for repeated processing of the same file

