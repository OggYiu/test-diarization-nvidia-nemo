# RTTM Parsing Fix - Handling Filenames with Spaces

## Problem

When the "Enhanced format (metadata + timestamps)" checkbox was enabled, the processing would fail with an error:

```
❌ Error during pipeline: could not convert string to float: '0489_8330-96674941_202510200201201108'

ValueError: could not convert string to float: '0489_8330-96674941_202510200201201108'
```

## Root Cause

### RTTM File Format
The RTTM (Rich Transcription Time Marked) format is:
```
SPEAKER <filename> <channel> <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
```

Example with simple filename:
```
SPEAKER test 1 0.380 0.875 <NA> <NA> speaker_1 <NA> <NA>
```

### The Issue
When the audio filename contains **spaces** (like `[Dickson Lau 0489]_8330-96674941...`), the RTTM line becomes:
```
SPEAKER [Dickson Lau 0489]_8330-96674941_202510200201201108 1 0.380 0.875 <NA> <NA> speaker_1 <NA> <NA>
```

**Old parsing method** used `.split()` which splits on ALL whitespace:
```python
parts = line.split()
# parts[0] = "SPEAKER"
# parts[1] = "[Dickson"      ❌ Wrong!
# parts[2] = "Lau"           ❌ Wrong!
# parts[3] = "0489]_8330-96674941_202510200201201108"  ❌ Should be start_time!
# parts[4] = "1"
# parts[5] = "0.380"         ← This is the actual start_time
```

When the code tried to do `float(parts[3])`, it got `'0489_8330-96674941_202510200201201108'` instead of a number, causing the error.

## Solution

**Parse from the right side** of the RTTM format instead of the left, since the last 5 fields are always in a fixed format:

```
... <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    parts[-7]    parts[-6]   ...    parts[-3]    ...
```

### New Implementation

```python
def parse_rttm_timestamps(rttm_content: str) -> dict:
    timestamps = {}
    lines = rttm_content.strip().split('\n')
    
    for idx, line in enumerate(lines):
        if line.strip():
            parts = line.split()
            if len(parts) >= 10:
                # Parse from the end (last 5 fields are always fixed format)
                try:
                    speaker_id = parts[-3]      # speaker_0, speaker_1, etc.
                    start_time = float(parts[-7])  # timestamp in seconds
                    timestamps[idx] = (speaker_id, start_time)
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Warning: Could not parse RTTM line {idx}: {e}")
                    continue
    
    return timestamps
```

### Why This Works

The last 7 fields in RTTM are always:
- `parts[-7]` = start_time (float)
- `parts[-6]` = duration (float)
- `parts[-5]` = `<NA>`
- `parts[-4]` = `<NA>`
- `parts[-3]` = speaker_id (string like "speaker_0")
- `parts[-2]` = `<NA>`
- `parts[-1]` = `<NA>`

**Regardless of how many spaces are in the filename**, these positions remain constant when counting from the end!

## Example

### RTTM Line
```
SPEAKER [Dickson Lau 0489]_8330-96674941_202510200201201108 1 0.380 0.875 <NA> <NA> speaker_1 <NA> <NA>
```

### After `.split()`
```python
parts = [
    'SPEAKER',                                    # parts[0]
    '[Dickson',                                   # parts[1]
    'Lau',                                        # parts[2]
    '0489]_8330-96674941_202510200201201108',    # parts[3]
    '1',                                          # parts[4]
    '0.380',                                      # parts[5] or parts[-7] ✅
    '0.875',                                      # parts[6] or parts[-6]
    '<NA>',                                       # parts[7] or parts[-5]
    '<NA>',                                       # parts[8] or parts[-4]
    'speaker_1',                                  # parts[9] or parts[-3] ✅
    '<NA>',                                       # parts[10] or parts[-2]
    '<NA>'                                        # parts[11] or parts[-1]
]
```

### New Parsing (from right)
```python
speaker_id = parts[-3]      # 'speaker_1' ✅ Correct!
start_time = float(parts[-7])  # 0.380 ✅ Correct!
```

## Benefits

✅ **Robust**: Works with filenames containing spaces  
✅ **Backward Compatible**: Still works with simple filenames  
✅ **Safe**: Error handling with try-except  
✅ **Standard**: Follows RTTM format specification  

## Testing

### Test Case 1: Filename with spaces
**Input RTTM:**
```
SPEAKER [Dickson Lau 0489]_8330-96674941_202510200201201108 1 0.380 0.875 <NA> <NA> speaker_1 <NA> <NA>
SPEAKER [Dickson Lau 0489]_8330-96674941_202510200201201108 1 1.255 4.705 <NA> <NA> speaker_0 <NA> <NA>
```

**Expected Output:**
```python
{
    0: ('speaker_1', 0.380),
    1: ('speaker_0', 1.255)
}
```

**Result:** ✅ Pass

### Test Case 2: Simple filename (backward compatibility)
**Input RTTM:**
```
SPEAKER test 1 0.380 0.875 <NA> <NA> speaker_1 <NA> <NA>
SPEAKER test 1 1.255 4.705 <NA> <NA> speaker_0 <NA> <NA>
```

**Expected Output:**
```python
{
    0: ('speaker_1', 0.380),
    1: ('speaker_0', 1.255)
}
```

**Result:** ✅ Pass

## Files Modified

- **`tabs/tab_stt.py`**: Fixed `parse_rttm_timestamps()` function to parse from right side

## Related Issue

This fix resolves the error that occurred when:
1. Enhanced format checkbox was enabled ✅
2. Audio filename contained spaces (e.g., broker names with multiple words) ✅
3. RTTM parsing attempted to extract timestamps ❌ → ✅

## RTTM Format Reference

Standard RTTM format specification:
```
Type File Channel Start Duration Ortho Stype Name Confidence
```

Where:
- **Type**: always "SPEAKER"
- **File**: audio filename (may contain spaces!)
- **Channel**: usually "1"
- **Start**: start time in seconds (float)
- **Duration**: duration in seconds (float)
- **Ortho**: orthography, usually `<NA>`
- **Stype**: speaker type, usually `<NA>`
- **Name**: speaker identifier (e.g., "speaker_0")
- **Confidence**: confidence score, usually `<NA>`

Source: NIST RT evaluation specifications

