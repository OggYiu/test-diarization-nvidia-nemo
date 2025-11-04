# STT Tab - LLM Speaker Identification Enhancement

## Summary of Changes

The STT (Speech-to-Text) tab has been enhanced with automatic metadata extraction and LLM-powered speaker identification.

## Key Features Added

### 1. Automatic Metadata Extraction
- **Removed**: Manual metadata extraction button
- **Added**: Automatic metadata extraction during the "Auto-Diarize & Transcribe" process
- The metadata (broker name, client name, etc.) is now automatically extracted from the filename and saved to MongoDB when you click the main button
- Metadata is displayed in a dedicated text box at the top of the results

### 2. LLM Speaker Identification
- **Purpose**: Automatically identify which speaker (speaker_0 or speaker_1) is the broker and which is the client
- **Method**: Uses Ollama LLM (default: qwen3:32b) to analyze the conversation context
- **Output**: Replaces generic labels with meaningful labels:
  - `speaker_0:` → `經紀 {broker_name}:` or `客戶 {client_name}:`
  - `speaker_1:` → `經紀 {broker_name}:` or `客戶 {client_name}:`

### 3. Enhanced UI Layout

#### Input Section (Left Column)
- Audio file upload
- Overwrite diarization checkbox
- Model selection (SenseVoiceSmall, Whisper-v3-Cantonese)
- Language selection
- **Single button**: "🎯 Auto-Diarize & Transcribe"

#### Output Section (Right Column)
1. **File Metadata** - Auto-extracted metadata in JSON format
2. **Raw Transcriptions** - Original transcriptions with speaker_0/speaker_1 labels
   - SenseVoiceSmall results
   - Whisper-v3-Cantonese results
3. **LLM-Labeled Transcriptions** - Conversations with 經紀/客戶 labels
   - SenseVoiceSmall (Labeled)
   - Whisper-v3-Cantonese (Labeled)
4. **LLM Speaker Identification Log** - LLM's reasoning for speaker identification
5. **Download ZIP** - All results packaged for download
6. **Status Log** - Processing status and logs

## Technical Implementation

### New Function: `identify_speakers_with_llm()`
```python
def identify_speakers_with_llm(conversation_text: str, broker_name: str, client_name: str, 
                                model: str = DEFAULT_MODEL, ollama_url: str = DEFAULT_OLLAMA_URL)
    -> tuple[str, str, str]:
```

**Parameters:**
- `conversation_text`: The conversation with speaker_0 and speaker_1 labels
- `broker_name`: Name of the broker from metadata
- `client_name`: Name of the client from metadata
- `model`: LLM model to use (default: qwen3:32b)
- `ollama_url`: Ollama server URL (default: http://localhost:11434)

**Returns:**
- `labeled_conversation`: Conversation with '經紀 {name}:' and '客戶 {name}:' labels
- `identification_log`: LLM's reasoning for identification
- `broker_speaker_id`: "speaker_0" or "speaker_1" indicating which one is the broker

**LLM Prompt Strategy:**
The LLM is given specific criteria to identify speakers:
1. **Broker characteristics**:
   - Provides market information and advice
   - Confirms transaction instructions
   - Uses professional terminology
   - Proactively asks about client needs
   - Repeats order details for confirmation

2. **Client characteristics**:
   - Asks about stock information
   - Issues buy/sell orders
   - Responds to broker confirmations

### Modified Function: `process_chop_and_transcribe()`

**New Steps Added:**
- **Step 0**: Extract metadata from filename (automatic)
- **Step 6.5**: Use LLM to identify speakers and create labeled conversations

**Updated Return Values:**
```python
return (metadata_json, json_file, sensevoice_txt, whisperv3_txt, zip_file, 
        sensevoice_conversation, whisperv3_conversation, 
        sensevoice_labeled, whisperv3_labeled, 
        llm_identification_log, status)
```

## Workflow

1. **User uploads audio file** and clicks "🎯 Auto-Diarize & Transcribe"
2. **System extracts metadata** from filename automatically
   - Broker name, client name, datetime, etc.
   - Saves to MongoDB
3. **System performs diarization** (cached if available)
4. **System chops audio** into segments based on speakers
5. **System transcribes** segments using selected models
6. **System uses LLM** to identify speakers
   - Analyzes conversation context
   - Determines who is broker and who is client
7. **System displays results**:
   - Metadata
   - Raw transcriptions (speaker_0/speaker_1)
   - Labeled transcriptions (經紀/客戶)
   - LLM reasoning log
   - Status log

## Configuration

### LLM Settings
- **Default Model**: qwen3:32b (can be changed in `model_config.py`)
- **Default URL**: http://localhost:11434 (can be changed in `model_config.py`)
- **Temperature**: 0.3 (lower for more deterministic results)

### Available Models
As defined in `model_config.py`:
- qwen3:32b
- qwen3:14b
- qwen3:8b
- gpt-oss:20b
- gemma3:27b
- deepseek-r1:32b
- deepseek-r1:70b
- qwen2.5:72b

## Error Handling

1. **Metadata extraction fails**: 
   - Continues with "Unknown" for broker/client names
   - Skips LLM speaker identification

2. **LLM identification fails**:
   - Returns original conversation with speaker_0/speaker_1 labels
   - Logs error message in identification log

3. **LLM cannot determine speaker**:
   - Defaults to speaker_0 as broker
   - Adds warning message to log

## Dependencies

- `langchain_ollama`: For LLM integration
- `ChatOllama`: Chat interface for Ollama models
- Existing dependencies: MongoDB, FunASR, Transformers, etc.

## Benefits

1. **Streamlined workflow**: One button instead of two
2. **Better UX**: Automatic metadata extraction
3. **Meaningful labels**: Easy to understand who is speaking
4. **Transparency**: LLM reasoning is shown to users
5. **Fallback mechanism**: Shows both raw and labeled versions
6. **Cached results**: MongoDB caching for fast repeated processing

## Future Enhancements

Potential improvements:
1. Allow users to select different LLM models from UI
2. Add manual override for speaker identification
3. Save LLM identification results to MongoDB for caching
4. Support for more than 2 speakers
5. Multi-language support for speaker identification prompts

