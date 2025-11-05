# JSON Batch Analysis - README

## Overview

The **JSON Batch Analysis** tab allows you to process multiple conversations at once by providing a JSON array. Each conversation is analyzed sequentially to extract stock information using one or more LLM models.

## Features

✅ **Batch Processing**: Analyze multiple conversations in one go  
✅ **Multi-LLM Support**: Use multiple LLM models to analyze each conversation  
✅ **Vector Store Correction**: Automatically correct STT errors using Milvus stock database  
✅ **Comprehensive Metadata**: Track broker, client, datetime, and other information  
✅ **Combined JSON Output**: Get all results in a single, structured JSON format  
✅ **Sequential Processing**: Conversations are processed one by one to avoid VRAM issues  

## JSON Input Format

The input should be a JSON array with conversation objects. Each conversation object should have:

```json
[
  {
    "conversation_number": 1,
    "filename": "example.wav",
    "metadata": {
      "broker_name": "Dickson Lau",
      "broker_id": "0489",
      "client_name": "CHENG SUK HING",
      "client_id": "P77197",
      "hkt_datetime": "2025-10-20T10:01:20"
    },
    "transcriptions": {
      "sensevoice": "經紀: 你好\n客戶: 我想買騰訊"
    }
  },
  {
    "conversation_number": 2,
    "filename": "example2.wav",
    "metadata": {
      "broker_name": "John Doe",
      "client_name": "Jane Smith"
    },
    "transcriptions": {
      "whisper": "經紀: 買咩股票?\n客戶: 買小米"
    }
  }
]
```

### Required Fields

- **conversation_number**: Integer identifier for the conversation
- **filename**: String filename of the audio file
- **transcriptions**: Dictionary of transcription sources or a string

### Optional Fields

- **metadata**: Dictionary with additional information like:
  - `broker_name`
  - `broker_id`
  - `client_name`
  - `client_id`
  - `hkt_datetime`
  - `client_number`
  - etc.

### Transcriptions Field

The `transcriptions` field can be:

1. **Dictionary** (recommended):
   ```json
   "transcriptions": {
     "sensevoice": "經紀: 你好...",
     "whisper": "經紀: 你好...",
     "funasr": "經紀: 你好..."
   }
   ```
   The system will use the first available transcription.

2. **String**:
   ```json
   "transcriptions": "經紀: 你好\n客戶: 我想買股票"
   ```

## How to Use

### Step 1: Prepare Your JSON

1. Create a JSON file or string with all your conversations
2. Each conversation should follow the format above
3. You can use the provided `example_json_batch.json` as a template

### Step 2: Select LLMs

1. Choose one or more LLM models from the checkbox list
2. Multiple LLMs will analyze each conversation independently
3. Default: First LLM in the list

### Step 3: Configure Settings

**Vector Store Correction** (Recommended: ✅ Enabled)
- Automatically corrects stock names that may have STT errors
- Uses your Milvus stock database for matching
- Provides confidence scores for corrections

**System Message**
- Customize the prompt for the LLM
- Default message is optimized for Cantonese stock conversations

**Temperature**
- Lower (0.1) = More deterministic and focused
- Higher (1.0+) = More creative but less consistent
- Recommended: 0.1 for stock extraction

**Ollama URL**
- Default: `http://localhost:11434`
- Change if your Ollama server is on a different host

### Step 4: Run Analysis

1. Click "🚀 Analyze All Conversations"
2. The system will process each conversation sequentially
3. Results will appear in real-time
4. Combined JSON output will be available after all conversations are processed

## Output Format

### Formatted Results Display

The formatted results show:
- Conversation details (number, filename, metadata)
- Transcription preview
- Stock extraction results for each LLM
- Stock information including:
  - Stock code/number
  - Stock name
  - Original word (if corrected)
  - Corrected stock name and code
  - Confidence level
  - Relevance score
  - Reasoning

### Combined JSON Output

The combined JSON output includes all conversations with their extracted stocks:

```json
[
  {
    "conversation_number": 1,
    "filename": "example.wav",
    "metadata": { ... },
    "transcription_source": "sensevoice",
    "analysis_timestamp": "2025-11-05 12:00:00",
    "llms_used": ["qwen2.5:32b-instruct", "llama3.3:70b-instruct"],
    "stocks": [
      {
        "stock_number": "00700",
        "stock_name": "騰訊",
        "confidence": "high",
        "relevance_score": 2,
        "corrected_stock_name": "騰訊控股",
        "corrected_stock_number": "00700",
        "correction_confidence": 0.95,
        "reasoning": "Client explicitly mentioned buying Tencent",
        "llm_model": "qwen2.5:32b-instruct"
      }
    ]
  }
]
```

## Example Workflow

### Example 1: Analyze 2 Conversations with 1 LLM

**Input:**
```json
[
  {
    "conversation_number": 1,
    "filename": "call1.wav",
    "metadata": {"broker_name": "John", "client_name": "Mary"},
    "transcriptions": {"sensevoice": "經紀: 你好\n客戶: 買騰訊"}
  },
  {
    "conversation_number": 2,
    "filename": "call2.wav",
    "metadata": {"broker_name": "John", "client_name": "Tom"},
    "transcriptions": {"sensevoice": "經紀: 你好\n客戶: 買小米"}
  }
]
```

**Steps:**
1. Paste the JSON in the input box
2. Select 1 LLM (e.g., "qwen2.5:32b-instruct")
3. Enable Vector Store Correction
4. Click "Analyze All Conversations"

**Expected Processing:**
- Conversation 1 analyzed with qwen2.5:32b-instruct
- Conversation 2 analyzed with qwen2.5:32b-instruct
- Total time: ~30-60 seconds (depending on LLM speed)

### Example 2: Analyze 5 Conversations with 2 LLMs

**Steps:**
1. Prepare JSON with 5 conversations
2. Select 2 LLMs (e.g., "qwen2.5:32b-instruct", "llama3.3:70b-instruct")
3. Click "Analyze All Conversations"

**Expected Processing:**
- Each conversation analyzed twice (once per LLM)
- Total analyses: 5 conversations × 2 LLMs = 10 analyses
- Sequential processing to avoid VRAM issues
- Total time: ~5-10 minutes

## Best Practices

### 1. Batch Size
- Recommended: 10-50 conversations per batch
- Larger batches will take longer but are more efficient
- Consider your LLM server's capacity

### 2. LLM Selection
- **Single LLM**: Faster, good for quick analysis
- **Multiple LLMs**: More reliable, cross-validation of results
- Recommended models:
  - `qwen2.5:32b-instruct` (Fast, good accuracy)
  - `llama3.3:70b-instruct` (Slower, high accuracy)

### 3. Vector Store Correction
- Always enable for production use
- Ensures stock names are accurate
- Corrects common STT errors (e.g., "金碟" → "金蝶國際")

### 4. Temperature
- Use 0.1-0.3 for stock extraction (deterministic)
- Higher temperatures may introduce variability

### 5. Transcription Quality
- Ensure transcriptions include speaker labels
- Include timestamps if available
- Clean up obvious errors before batch processing

## Integration with Other Tools

### From STT Tab
1. Use the STT tab to transcribe audio files
2. Export transcriptions with metadata
3. Format as JSON array
4. Use JSON Batch Analysis tab

### To Transaction Analysis Tab
1. Get combined JSON output from batch analysis
2. Extract stock information
3. Use Transaction Analysis tab to verify trades

### To Text Correction Tab
1. If transcriptions have errors, use Text Correction tab first
2. Then use corrected transcriptions in JSON batch analysis

## Troubleshooting

### "Invalid JSON format"
- Check JSON syntax (missing commas, brackets)
- Use a JSON validator (e.g., jsonlint.com)
- Ensure proper escaping of special characters

### "No transcription text found"
- Verify the `transcriptions` field is not empty
- Check that at least one transcription source has text
- Ensure transcription text is not just whitespace

### "Error with LLM"
- Verify Ollama server is running
- Check Ollama URL is correct
- Ensure the selected model is downloaded

### Slow Processing
- Reduce number of conversations per batch
- Use fewer LLMs
- Use faster models (e.g., smaller parameter counts)

### Out of Memory
- Process fewer conversations at a time
- Use only 1 LLM at a time
- Close other applications using VRAM

## Performance Tips

1. **Pre-filter conversations**: Remove conversations without valuable content
2. **Use efficient models**: Smaller models (7B-32B) are much faster
3. **Batch similar conversations**: Group by broker or client for better context
4. **Monitor VRAM**: Keep an eye on GPU memory usage
5. **Sequential processing**: The tool already processes sequentially to avoid overload

## API Reference

### `process_json_batch()`

**Parameters:**
- `json_input` (str): JSON string containing array of conversation objects
- `selected_llms` (list[str]): List of LLM model names
- `system_message` (str): System message for the LLMs
- `ollama_url` (str): Ollama server URL
- `temperature` (float): Temperature parameter (0.0-2.0)
- `use_vector_correction` (bool): Enable vector store correction

**Returns:**
- `tuple[str, str]`: (formatted_results, combined_json)

## Support

For issues or questions:
1. Check this README
2. Review the example JSON files
3. Check Ollama server logs
4. Verify Milvus vector store is initialized

## Changelog

### Version 1.0 (2025-11-05)
- Initial release
- Support for JSON batch processing
- Multi-LLM analysis
- Vector store correction
- Comprehensive metadata tracking
