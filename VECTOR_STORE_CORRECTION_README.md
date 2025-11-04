# Vector Store Stock Name Correction Feature

## Overview

This feature enhances the STT & Stock Comparison tab by automatically correcting stock names and codes that may have been incorrectly transcribed by Speech-to-Text (STT) models. It uses semantic similarity search powered by Milvus vector store to find the most likely correct stock information.

## How It Works

### Workflow

1. **STT Processing**: Audio is transcribed by one or more STT models (which may introduce errors)
2. **LLM Extraction**: LLMs analyze the transcription and extract stock information (stock codes and names)
3. **Vector Store Correction** (NEW): Each identified stock is validated against the Milvus vector store
4. **Results Display**: Both original and corrected information are displayed with confidence scores

### Correction Process

For each stock identified by the LLM:

1. **Query Generation**: The system searches the vector store using both:
   - Stock name (e.g., "騰訊")
   - Stock code (e.g., "00700")

2. **Similarity Matching**: Top 3 most similar results are retrieved for each query

3. **Confidence Scoring**: Results are scored based on semantic similarity
   - Confidence > 0.5: Correction is suggested
   - Higher confidence = more likely to be correct

4. **Data Extraction**: Stock information is extracted from the matched results:
   - Metadata fields: `stock_code`, `InstrumentCd`, `code`, `stock_name`, `AliasName`, `name`
   - Content parsing: Supports multiple formats (CSV, space-separated, etc.)

5. **Display**: If corrections are found, both original and corrected values are shown

## Features

### New StockInfo Fields

- `corrected_stock_name`: Corrected stock name (if different from original)
- `corrected_stock_number`: Corrected stock code (if different from original)
- `correction_confidence`: Confidence score for the correction (0.0-1.0)

### Enhanced Display

Results now show:
```
   1. ✅ 股票 #1
      • 股票代號: 00700 (original from LLM)
      • 股票名稱: 騰訊 (original from LLM)
      🔧 修正後:
         ◦ 股票代號: 00700
         ◦ 股票名稱: 騰訊控股
         ◦ 修正信心: 85.32%
      • 置信度: HIGH
      • 相關程度: 🟢 2/2
      • 推理: [Vector Store Correction: 騰訊 → 騰訊控股 (confidence: 85.32%)]
```

## Configuration

### Milvus Connection

The system connects to your Milvus vector store:
- **Endpoint**: Configured in `MILVUS_CLUSTER_ENDPOINT`
- **Database**: `stocks`
- **Collection**: `phone_calls`
- **Embedding Model**: `qwen3-embedding:8b` (via Ollama)

### UI Controls

In the "⚙️ Advanced Settings" section:

- **Enable Vector Store Correction** (checkbox)
  - Default: Enabled
  - When disabled, only LLM extraction is performed (no vector store correction)

## Benefits

### 1. Error Correction
Fixes common STT errors:
- Homophone errors: "百" → "八"
- Similar sounding words: "孤/沽" → correct stock name
- Incomplete names: "騰訊" → "騰訊控股"

### 2. Validation
Confirms that identified stocks actually exist in your database

### 3. Confidence Metrics
Provides transparency about the correction quality

### 4. Flexibility
Can be enabled/disabled per request

## Data Format Support

The correction function can parse multiple data formats:

### CSV Format
```
00700,騰訊控股
```

### Space-Separated
```
00700 騰訊控股
```

### Metadata
```json
{
  "stock_code": "00700",
  "stock_name": "騰訊控股"
}
```

### Alternative Field Names
- Stock code: `stock_code`, `InstrumentCd`, `code`
- Stock name: `stock_name`, `AliasName`, `name`

## Performance Considerations

### Initialization
- Vector store is initialized once per session (lazy loading)
- Embedding model is loaded into memory

### Per-Request
- Each stock performs 2 similarity searches (by name and code)
- Top 3 results retrieved per query
- Minimal latency impact (<1s per stock)

### Memory
- Vector store connection persists across requests
- Embedding model stays in memory

## Troubleshooting

### Vector Store Not Connecting

If you see errors like "Vector store not available":
1. Check Ollama is running: `ollama list`
2. Verify embedding model is installed: `ollama pull qwen3-embedding:8b`
3. Check Milvus endpoint is accessible
4. Verify network connectivity

### No Corrections Shown

Possible reasons:
1. Confidence threshold not met (< 0.5)
2. LLM extracted correct information (no correction needed)
3. Stock not found in vector store
4. Vector store correction is disabled

### Low Confidence Scores

If correction confidence is consistently low:
1. Check vector store has relevant stock data
2. Verify embedding model quality
3. Consider updating stock database

## Example Usage

### Input Transcription (with STT error)
```
券商：你好，請問需要什麼幫助？
客戶：我想買泡泡沬特
券商：好的，一百一三八，買多少？
```

### LLM Extraction
```
股票代號: 18138
股票名稱: 泡泡沬特
```

### After Vector Correction
```
股票代號: 18138
股票名稱: 泡泡瑪特
🔧 修正後:
   ◦ 股票名稱: 泡泡瑪特
   ◦ 修正信心: 78.45%
```

## Technical Details

### Dependencies
- `langchain-ollama`: Ollama embeddings integration
- `langchain-milvus`: Milvus vector store integration
- `pydantic`: Data validation

### Key Functions

1. `initialize_vector_store()`: Lazy initialization of Milvus connection
2. `correct_stock_with_vector_store()`: Main correction logic
3. `extract_stocks_with_single_llm()`: Enhanced to apply corrections

## Future Enhancements

Potential improvements:
1. Adjustable confidence threshold
2. Multiple embedding models
3. Correction history tracking
4. Batch correction optimization
5. Alternative vector stores support

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Verify all dependencies are installed
3. Test vector store connection separately using the Milvus Search tab
4. Review Milvus collection schema

---

**Last Updated**: November 4, 2025

