# Tab Chaining Guide

## Overview

This guide explains how to use the tab chaining feature that allows you to pass data from one tab to another in the Phone Call Analysis Suite.

## What is Tab Chaining?

Tab chaining enables you to:
1. Process audio files in the **STT Tab** (Speech-to-Text)
2. Automatically pass the transcription results to the **JSON Batch Analysis Tab**
3. Extract stock information without manually copying/pasting data

## How to Use Tab Chaining

### Step 1: Process Audio in STT Tab

1. Open the **"3️⃣ Auto-Diarize & Transcribe"** tab
2. Select your audio file(s)
3. Configure your settings (model selection, corrections, etc.)
4. Click **"🚀 Transcribe Audio"**
5. Wait for the transcription to complete
6. The results will appear in the output areas, including a JSON output

### Step 2: Load Data in JSON Batch Analysis Tab

1. Switch to the **"🔟 JSON Batch Analysis"** tab
2. Look for the **"📥 Load from STT Tab"** button at the top
3. Click this button to automatically load the JSON data from Step 1
4. The JSON input box will be populated with your transcription data
5. Configure your analysis settings (LLM selection, temperature, etc.)
6. Click **"🚀 Analyze All Conversations"**

## What Data is Passed Between Tabs?

The STT tab generates JSON data in this format:

```json
[
  {
    "conversation_number": 1,
    "filename": "example.wav",
    "metadata": {
      "hkt_datetime": "2024-01-15 14:30:00",
      "broker_name": "John Doe",
      "broker_id": "B123",
      "client_name": "Jane Smith",
      "client_id": "C456"
    },
    "transcriptions": {
      "sensevoice": "經紀: 你好\n客戶: 我想買騰訊",
      "whisperv3_cantonese": "經紀: 你好\n客戶: 我想買騰訊"
    }
  }
]
```

This complete data structure includes:
- Conversation metadata (timestamp, broker, client info)
- Transcriptions from both models (if both were enabled)
- Filename and conversation number for tracking

## Technical Details

### Architecture

The implementation uses Gradio's state management:

```python
# In unified_gui.py
shared_json_data = gr.State(None)  # Shared state for passing data

create_stt_tab(output_json_state=shared_json_data)
create_json_batch_analysis_tab(input_json_state=shared_json_data)
```

### Modified Functions

1. **`create_stt_tab(output_json_state=None)`**
   - Added optional parameter to receive state component
   - Outputs JSON data to the state when processing completes

2. **`create_json_batch_analysis_tab(input_json_state=None)`**
   - Added optional parameter to receive state component
   - Creates a "Load from STT Tab" button when state is provided
   - Button loads data from the state into the JSON input box

## Benefits

✅ **No Manual Copy/Paste**: Data flows automatically between tabs  
✅ **Preserves Metadata**: All conversation information is maintained  
✅ **Flexible Workflow**: You can still manually paste JSON if needed  
✅ **Multi-File Support**: Process multiple audio files and analyze them together  

## Extending the Chain

To add more tabs to the chain:

1. **Add a new state variable in `unified_gui.py`**:
   ```python
   shared_analysis_results = gr.State(None)
   ```

2. **Pass states to tabs**:
   ```python
   create_json_batch_analysis_tab(
       input_json_state=shared_json_data,
       output_analysis_state=shared_analysis_results  # New output
   )
   create_next_tab(
       input_analysis_state=shared_analysis_results  # New input
   )
   ```

3. **Modify tab functions to accept/output states**:
   - Add optional parameters to function signatures
   - Create load buttons for input states
   - Add state to outputs list for output states

## Troubleshooting

### "No data from STT tab" Message

**Problem**: Clicking "Load from STT Tab" shows a warning message.

**Solution**: 
- Make sure you've run the STT tab first
- The STT processing must complete successfully
- The JSON output must be generated

### Data Not Updating

**Problem**: Old data appears when loading from STT tab.

**Solution**:
- Run the STT tab again to update the shared state
- The most recent STT result will always be available

### Manual JSON Entry Still Works

You can always manually paste JSON into the input box if you prefer, or if you want to process data from a different source.

## Example Workflow

```
1. Upload audio files: call1.wav, call2.wav
   ↓
2. STT Tab processes both files
   ↓
3. JSON output generated with 2 conversations
   ↓
4. Switch to JSON Batch Analysis Tab
   ↓
5. Click "📥 Load from STT Tab"
   ↓
6. JSON appears in input box automatically
   ↓
7. Configure LLM settings
   ↓
8. Click "Analyze All Conversations"
   ↓
9. Get stock extraction results for both conversations
```

## Future Enhancements

Potential future improvements:
- Add visual indicators showing which tabs have data ready
- Create a pipeline tab that runs multiple steps automatically
- Add ability to chain more than 2 tabs together
- Save and load complete workflows

---

**Created**: November 7, 2025  
**Version**: 1.0

