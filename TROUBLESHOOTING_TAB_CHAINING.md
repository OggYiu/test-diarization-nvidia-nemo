# Troubleshooting: Tab Chaining

## Issue: ValueError - Function didn't return enough output values

### Error Message

```
ValueError: A function (process_audio_or_folder) didn't return enough output values 
(needed: 9, returned: 8).
```

### What Happened

When the STT tab was configured to output to a shared state for chaining:
- The original function returns **8 values**
- Gradio expected **9 values** (8 display outputs + 1 state output)
- Result: Mismatch causing the error

### The Fix

Created a wrapper function that duplicates the JSON output:

```python
if output_json_state is not None:
    # Wrapper that duplicates JSON output for both display and state
    def process_with_state(*args):
        result = process_audio_or_folder(*args)
        # result is a tuple of 8 values, last one is combined_json
        # Return all 8 + duplicate the last one for state
        return result + (result[-1],)  # Add JSON to state
    
    process_fn = process_with_state
else:
    # No state needed, use original function
    process_fn = process_audio_or_folder
```

### Why This Works

1. **Original function unchanged**: `process_audio_or_folder` still returns 8 values
2. **Wrapper adds 9th value**: The JSON is duplicated for the state
3. **Conditional logic**: Only uses wrapper when state is needed
4. **Backward compatible**: Works with or without state parameter

### Output Mapping

When state is enabled:

| Index | Output Component | Source |
|-------|-----------------|--------|
| 0 | metadata textbox | result[0] |
| 1 | json file (hidden) | result[1] |
| 2 | sensevoice txt (hidden) | result[2] |
| 3 | whisperv3 txt (hidden) | result[3] |
| 4 | zip download | result[4] |
| 5 | sensevoice labeled | result[5] |
| 6 | whisperv3 labeled | result[6] |
| 7 | json output textbox | result[7] |
| 8 | **shared state** | result[7] *(duplicated)* |

### Testing

Run the test script to verify the fix:

```bash
python test_chaining.py
```

Expected output:
```
✓ PASS: tab_stt.py wrapper function
Total: 5/5 tests passed
```

## Common Issues

### 1. "No data from STT tab"

**Symptoms**: Load button shows warning message

**Causes**:
- STT tab hasn't been run yet
- Processing failed or was interrupted
- State was cleared (page refresh)

**Solution**: Run the STT tab and wait for completion

### 2. Old Data Appears

**Symptoms**: Previous transcription shows up instead of new one

**Causes**:
- State wasn't updated with new data
- Looking at cached results

**Solution**: Re-run STT tab to update state

### 3. Data Doesn't Load

**Symptoms**: JSON input box stays empty after clicking load

**Causes**:
- State is `None` or empty
- Button handler not connected properly

**Debug Steps**:
1. Check browser console for errors
2. Verify STT tab completed successfully
3. Check that JSON output appeared in STT tab

## Prevention

The wrapper function approach prevents this class of errors by:
- ✅ Isolating output count logic in one place
- ✅ Making the state output optional
- ✅ Not modifying core processing functions
- ✅ Clear separation between display and state outputs

## Alternative Approaches (Not Used)

### Modify Core Function
```python
# ❌ Not used - would break existing code
def process_audio_or_folder(..., return_for_state=False):
    # ... processing ...
    if return_for_state:
        return tuple(8 values) + (combined_json,)
    else:
        return tuple(8 values)
```

**Problems**: 
- Complicates core logic
- Every caller needs to know about state
- Not backward compatible

### Gradio .then() Chaining
```python
# ❌ Not used - more complex
btn.click(...).then(
    fn=lambda x: x,
    inputs=[json_output],
    outputs=[state]
)
```

**Problems**:
- Harder to understand
- Two separate operations
- More points of failure

### Direct State Update
```python
# ❌ Not used - requires manual state management
btn.click(fn=process)
# Then manually update state in a separate step
```

**Problems**:
- Requires tracking state updates
- Race conditions possible
- Not atomic

## Summary

The **wrapper function approach** is:
- ✅ Simple and clean
- ✅ Backward compatible
- ✅ Easy to test
- ✅ Self-contained
- ✅ Follows Gradio patterns

---

**Last Updated**: November 7, 2025  
**Status**: ✅ Fixed and Tested

