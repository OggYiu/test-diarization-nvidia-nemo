# JSON Batch Analysis - Implementation Notes

## Architecture Overview

The JSON Batch Analysis feature is built on top of the existing `tab_stt_stock_comparison.py` module, reusing its core functionality while adding batch processing capabilities.

### Key Components

```
tab_json_batch_analysis.py
├── process_json_batch()           # Main processing function
│   ├── JSON parsing & validation
│   ├── Conversation iteration
│   └── Result aggregation
│
└── Reused from tab_stt_stock_comparison.py:
    ├── extract_stocks_with_single_llm()
    ├── format_extraction_result()
    ├── DEFAULT_SYSTEM_MESSAGE
    └── LLM_OPTIONS
```

## Implementation Details

### 1. JSON Processing Flow

```python
Input JSON String
    ↓
json.loads() → Parse to Python objects
    ↓
Validate structure (must be array)
    ↓
For each conversation:
    ├── Extract metadata
    ├── Extract transcription text
    ├── For each selected LLM:
    │   ├── Call extract_stocks_with_single_llm()
    │   └── Store results
    └── Aggregate results
    ↓
Format outputs:
    ├── Human-readable results
    └── Combined JSON
```

### 2. Sequential Processing Strategy

**Why Sequential?**
- Prevents VRAM overflow on GPU
- More predictable memory usage
- Better error handling per conversation
- Real-time progress visibility

**Processing Order:**
```
For LLM_1:
    Process Conv_1
    Process Conv_2
    ...
    Process Conv_N
For LLM_2:
    Process Conv_1
    Process Conv_2
    ...
    Process Conv_N
```

**Alternative (NOT used):**
```
For Conv_1:
    Process with LLM_1
    Process with LLM_2
    ...
For Conv_2:
    Process with LLM_1
    Process with LLM_2
    ...
```

**Current Implementation:**
Actually uses conversation-first iteration:
```python
for conv in conversations:
    for llm in selected_llms:
        process(conv, llm)
```

This allows:
- Complete analysis per conversation before moving to next
- Better organization of results
- Easier to resume if interrupted

### 3. Transcription Field Handling

The `transcriptions` field is flexible:

**Case 1: Dictionary (Preferred)**
```python
"transcriptions": {
    "sensevoice": "text...",
    "whisper": "text...",
    "funasr": "text..."
}
```

**Processing:**
```python
if isinstance(transcriptions, dict):
    for source_name, text in transcriptions.items():
        if text and text.strip():
            transcription_text = text
            transcription_source = source_name
            break  # Use first available
```

**Case 2: String**
```python
"transcriptions": "text..."
```

**Processing:**
```python
elif isinstance(transcriptions, str):
    transcription_text = transcriptions
    transcription_source = "default"
```

### 4. Error Handling

**Conversation-Level Errors:**
```python
try:
    # Process conversation
except Exception as conv_error:
    # Log error
    # Continue to next conversation
    # Don't fail entire batch
```

**LLM-Level Errors:**
- Handled by `extract_stocks_with_single_llm()`
- Returns error message instead of raising exception
- Allows partial results

**JSON Parsing Errors:**
```python
try:
    conversations = json.loads(json_input)
except json.JSONDecodeError as e:
    return f"❌ Error: Invalid JSON format\n\n{str(e)}", ""
```

### 5. Result Aggregation

**Per-Conversation Results:**
```python
conversation_result = {
    "conversation_number": conv_number,
    "filename": filename,
    "metadata": metadata,
    "transcription_source": transcription_source,
    "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "llms_used": selected_llms,
    "stocks": conv_stocks  # All stocks from all LLMs
}
```

**Stock Tagging:**
Each stock is tagged with the LLM that found it:
```python
for stock in stocks:
    stock["llm_model"] = model
```

This allows:
- Cross-LLM comparison
- Confidence aggregation
- Model performance analysis

### 6. Vector Store Correction Integration

**When Enabled:**
```python
if use_vector_correction:
    vector_store = get_vector_store()
    if vector_store.initialize():
        correction_result = verify_and_correct_stock(
            stock_name=stock.stock_name,
            stock_code=stock.stock_number,
            vector_store=vector_store,
            strategy=SearchStrategy.OPTIMIZED,
        )
```

**Benefits:**
- Corrects STT errors (e.g., "金碟" → "金蝶國際")
- Provides confidence scores
- Uses semantic search via Milvus

**Always Populated Fields:**
```python
stock.corrected_stock_name = ...
stock.corrected_stock_number = ...
stock.correction_confidence = ...
```

Even when correction is disabled, these fields are populated with original values and 1.0 confidence.

### 7. Performance Optimization

**Memory Management:**
- Sequential processing prevents VRAM overflow
- No parallel LLM calls (intentional)
- Results streamed to output as they complete

**Time Complexity:**
- O(C × L) where C = conversations, L = LLMs
- Each analysis: ~3-30 seconds depending on model
- Total time: C × L × avg_time_per_analysis

**Example:**
```
10 conversations × 2 LLMs × 5 seconds = 100 seconds (~1.7 minutes)
```

**Progress Tracking:**
```python
msg = f"[Conversation {conv_number}/{total_conversations}] [LLM {llm_idx}/{total_llms}] ..."
logging.info(msg)
print(msg)  # Real-time console output
```

### 8. Output Format Design

**Formatted Results:**
- Human-readable
- Hierarchical structure
- Progress indicators
- Time tracking
- Metadata display

**JSON Output:**
- Machine-readable
- Complete data structure
- Ready for further processing
- Includes all metadata

### 9. Integration Points

**Imports:**
```python
from tabs.tab_stt_stock_comparison import (
    extract_stocks_with_single_llm,
    format_extraction_result,
    DEFAULT_SYSTEM_MESSAGE,
    LLM_OPTIONS,
)
```

**Benefits:**
- Code reuse
- Consistent behavior
- Single source of truth
- Easy maintenance

**Shared Configuration:**
- `model_config.py`: LLM models and Ollama URL
- `stock_verifier_module`: Vector store correction

## Design Decisions

### 1. Why Not Parallel Processing?

**Considered:**
```python
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process, conv, llm) 
               for conv in conversations 
               for llm in llms]
```

**Rejected Because:**
- GPU memory contention
- Ollama server may not support concurrent requests
- Harder to track progress
- Risk of VRAM overflow
- No significant speed benefit (GPU is the bottleneck)

### 2. Why Conversation-First Iteration?

**Alternative 1: LLM-first**
```python
for llm in llms:
    for conv in conversations:
        process(conv, llm)
```

**Alternative 2: Conversation-first (CHOSEN)**
```python
for conv in conversations:
    for llm in llms:
        process(conv, llm)
```

**Reasons for Choice:**
- Better result organization
- Complete analysis per conversation
- Easier to resume if interrupted
- More intuitive output structure

### 3. Why Flexible Transcription Field?

**Rationale:**
- Different STT systems use different formats
- Allow gradual migration from string to dict
- Support legacy formats
- More user-friendly

### 4. Why Always Populate Corrected Fields?

**Before (Inconsistent):**
```json
{
  "stock_name": "騰訊",
  "stock_number": "00700",
  "corrected_stock_name": null,  // Sometimes missing
  "corrected_stock_number": null  // Sometimes missing
}
```

**After (Consistent):**
```json
{
  "stock_name": "騰訊",
  "stock_number": "00700",
  "corrected_stock_name": "騰訊控股",
  "corrected_stock_number": "00700",
  "correction_confidence": 0.95
}
```

**Benefits:**
- Consistent JSON schema
- Easier to parse downstream
- Clear indication of correction status
- No null checks needed

## Code Quality

### Type Hints
```python
def process_json_batch(
    json_input: str,
    selected_llms: list[str],
    system_message: str,
    ollama_url: str,
    temperature: float,
    use_vector_correction: bool = True,
) -> tuple[str, str]:
```

### Docstrings
```python
"""
Process a JSON batch of conversations and extract stock information.

Args:
    json_input: JSON string containing array of conversation objects
    ...

Returns:
    tuple[str, str]: (formatted_results, combined_json)
"""
```

### Error Messages
```python
if not json_input or not json_input.strip():
    return "❌ Error: Please provide a JSON input", ""
```

### Logging
```python
logging.info(msg)
print(msg)  # Console output for monitoring
```

## Testing Recommendations

### Unit Tests
1. Test JSON parsing with various formats
2. Test error handling for invalid JSON
3. Test transcription field extraction
4. Test result aggregation

### Integration Tests
1. Test with real LLM models
2. Test vector store correction
3. Test with multiple conversations
4. Test with multiple LLMs

### Performance Tests
1. Measure time per conversation
2. Test with large batches (100+ conversations)
3. Monitor VRAM usage
4. Test with different LLM models

## Future Enhancements

### Potential Improvements

1. **Parallel Processing (with safeguards)**
   ```python
   max_concurrent = 2  # Limit concurrent LLM calls
   ```

2. **Progress Bar**
   ```python
   with tqdm(total=total_analyses) as pbar:
       # Update progress
   ```

3. **Resumable Processing**
   ```python
   # Save intermediate results
   # Resume from checkpoint
   ```

4. **Result Caching**
   ```python
   # Cache LLM responses
   # Avoid re-processing same text
   ```

5. **Advanced Aggregation**
   ```python
   # Merge stocks from multiple LLMs
   # Calculate consensus
   # Identify discrepancies
   ```

6. **Export Options**
   ```python
   # Export to CSV
   # Export to Excel
   # Export to database
   ```

## Dependencies

### Direct Dependencies
```python
import json
import traceback
import logging
import time
from typing import List, Dict, Any
from datetime import datetime
import gradio as gr
```

### Module Dependencies
```python
from tabs.tab_stt_stock_comparison import ...
from model_config import DEFAULT_OLLAMA_URL
```

### Transitive Dependencies
- langchain_ollama
- pydantic
- stock_verifier_module
- Milvus (for vector store correction)

## Configuration

### Default Settings
```python
DEFAULT_SYSTEM_MESSAGE = """..."""  # From tab_stt_stock_comparison
LLM_OPTIONS = [...]  # From model_config
DEFAULT_OLLAMA_URL = "http://localhost:11434"  # From model_config
```

### User-Configurable Settings
- Selected LLMs
- System message
- Temperature
- Ollama URL
- Vector store correction toggle

## Monitoring & Logging

### Log Levels
```python
logging.info()   # Progress messages
logging.warning() # Skipped conversations
logging.error()  # Errors
```

### Console Output
```python
print(msg)  # Real-time progress
```

### Structured Logging
```python
msg = f"[Conversation {conv_number}/{total_conversations}] [LLM {llm_idx}/{total_llms}] ..."
```

## Summary

The JSON Batch Analysis feature provides:
- ✅ Efficient batch processing
- ✅ Robust error handling
- ✅ Flexible input format
- ✅ Comprehensive output
- ✅ Code reuse
- ✅ Easy maintenance
- ✅ Good performance
- ✅ Clear monitoring

The implementation prioritizes reliability and usability over raw speed, making it suitable for production use.
