# Stock Verifier Module

## Overview

`stock_verifier.py` is a standalone, reusable module for verifying and correcting stock names and codes that may have been incorrectly transcribed by Speech-to-Text (STT) systems. It uses semantic similarity search powered by Milvus vector store and Ollama embeddings.

## Features

### ✨ Core Capabilities

- **Semantic Search**: Uses embeddings to find similar stock names/codes
- **Multiple Query Strategies**: Searches by name, code, and variations
- **Smart Parsing**: Handles multiple data formats (CSV, space-separated, metadata)
- **Confidence Scoring**: Transparent scoring with high/medium/low thresholds
- **STT Error Handling**: Built-in knowledge of common transcription errors
- **Batch Processing**: Efficient batch verification of multiple stocks
- **Singleton Pattern**: Shared vector store instance across calls

### 🎯 Accuracy Optimizations

1. **Multiple Search Strategies**
   - Primary query by name and code
   - Automatic name variations for common STT errors
   - Weighted confidence scoring based on query type

2. **Common STT Error Patterns**
   - Homophone substitutions (百 ↔ 八, 孤 ↔ 沽)
   - Similar-sounding characters (轮 → 輪, 星 → 升)
   - Character variations (號 ↔ 毫 ↔ 豪)

3. **Robust Data Extraction**
   - Metadata-first approach (more reliable)
   - Multiple format parsers (CSV, space, tab, labeled)
   - Stock code normalization (padding to 5 digits)

4. **Confidence Thresholds**
   - High: ≥ 0.8 (very confident)
   - Medium: ≥ 0.6 (moderately confident)
   - Low: ≥ 0.4 (possible correction)
   - Below 0.4: No correction suggested

## Installation

### Prerequisites

```bash
# Ensure Ollama is installed and running
ollama pull qwen3-embedding:8b

# Python dependencies (should already be installed)
pip install langchain-ollama langchain-milvus
```

### Configuration

Edit the constants in `stock_verifier.py`:

```python
MILVUS_CLUSTER_ENDPOINT = "your_milvus_endpoint"
MILVUS_TOKEN = "your_token"
MILVUS_DB_NAME = "stocks"
MILVUS_COLLECTION_NAME = "phone_calls"
EMBEDDING_MODEL = "qwen3-embedding:8b"

# Adjust thresholds if needed
CONFIDENCE_THRESHOLD_HIGH = 0.8
CONFIDENCE_THRESHOLD_MEDIUM = 0.6
CONFIDENCE_THRESHOLD_LOW = 0.4
```

## Usage

### Basic Usage

```python
from stock_verifier import verify_and_correct_stock

# Verify a single stock
result = verify_and_correct_stock(
    stock_name="泡泡沬特",  # STT error
    stock_code=None
)

print(f"Original: {result.original_stock_name}")
print(f"Corrected: {result.corrected_stock_name}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Applied: {result.correction_applied}")
```

### Advanced Usage

```python
from stock_verifier import (
    get_vector_store,
    verify_and_correct_stock,
    batch_verify_stocks,
    format_correction_summary,
)

# Initialize vector store once (optional, auto-initializes on first use)
vector_store = get_vector_store()
vector_store.initialize()

# Verify with custom confidence threshold
result = verify_and_correct_stock(
    stock_name="騰訊",
    stock_code="700",
    vector_store=vector_store,
    confidence_threshold=0.5,  # More lenient
)

# Format for display
summary = format_correction_summary(result)
print(summary)
```

### Batch Processing

```python
# Verify multiple stocks efficiently
stocks = [
    {"stock_name": "騰訊", "stock_code": "700"},
    {"stock_name": "泡泡沬特", "stock_code": None},
    {"stock_name": "小米", "stock_code": "1810"},
]

results = batch_verify_stocks(
    stocks=stocks,
    confidence_threshold=0.4,
)

for result in results:
    if result.correction_applied:
        print(f"Corrected: {result.original_stock_name} → {result.corrected_stock_name}")
```

### Integration Example

```python
# In your LLM extraction pipeline
from stock_verifier import verify_and_correct_stock

def extract_and_verify_stocks(transcription_text):
    # 1. Extract stocks with LLM
    stocks = llm_extract_stocks(transcription_text)
    
    # 2. Verify and correct each stock
    for stock in stocks:
        correction = verify_and_correct_stock(
            stock_name=stock['name'],
            stock_code=stock['code'],
        )
        
        # 3. Apply corrections if confident
        if correction.correction_applied:
            stock['name'] = correction.corrected_stock_name or stock['name']
            stock['code'] = correction.corrected_stock_code or stock['code']
            stock['was_corrected'] = True
            stock['correction_confidence'] = correction.confidence
    
    return stocks
```

## API Reference

### Classes

#### `StockVectorStore`

Manages connection to Milvus vector store.

**Methods:**
- `initialize()` → `bool`: Initialize connection
- `search(query: str, k: int)` → `List[Tuple[Doc, float]]`: Search vector store
- `is_available` → `bool`: Check if store is ready

#### `StockCorrectionResult`

Data class containing verification results.

**Attributes:**
- `original_stock_name`: Original name from STT/LLM
- `original_stock_code`: Original code from STT/LLM
- `corrected_stock_name`: Corrected name (if different)
- `corrected_stock_code`: Corrected code (if different)
- `confidence`: Confidence score (0.0-1.0)
- `correction_applied`: Whether correction was applied
- `confidence_level`: String level ("high", "medium", "low", "none")
- `reasoning`: Explanation of the correction/verification
- `matched_content`: Preview of matched vector store content
- `metadata`: Additional metadata from match

### Functions

#### `verify_and_correct_stock()`

Main verification function.

```python
def verify_and_correct_stock(
    stock_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    vector_store: Optional[StockVectorStore] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_LOW,
    top_k: int = TOP_K_RESULTS,
) -> StockCorrectionResult:
```

**Parameters:**
- `stock_name`: Stock name to verify
- `stock_code`: Stock code to verify
- `vector_store`: Vector store instance (auto-created if None)
- `confidence_threshold`: Minimum confidence to apply correction
- `top_k`: Number of vector search results to retrieve

**Returns:** `StockCorrectionResult` object

#### `batch_verify_stocks()`

Batch verification of multiple stocks.

```python
def batch_verify_stocks(
    stocks: List[Dict[str, Optional[str]]],
    vector_store: Optional[StockVectorStore] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_LOW,
) -> List[StockCorrectionResult]:
```

**Parameters:**
- `stocks`: List of dicts with 'stock_name' and/or 'stock_code' keys
- `vector_store`: Vector store instance
- `confidence_threshold`: Minimum confidence threshold

**Returns:** List of `StockCorrectionResult` objects

#### `get_vector_store()`

Get or create the global vector store singleton.

```python
def get_vector_store() -> StockVectorStore:
```

**Returns:** Global `StockVectorStore` instance

#### Utility Functions

```python
# Normalize stock code (e.g., "700" → "00700")
def normalize_stock_code(code: str) -> str

# Check if string looks like a stock code
def is_valid_stock_code(code: str) -> bool

# Parse stock from content string
def parse_stock_from_content(content: str) -> Tuple[Optional[str], Optional[str]]

# Extract from metadata dict
def extract_from_metadata(metadata: Dict) -> Tuple[Optional[str], Optional[str]]

# Generate name variations for better matching
def generate_name_variations(name: str) -> List[str]

# Format result as readable summary
def format_correction_summary(result: StockCorrectionResult) -> str
```

## Performance

### Initialization (One-time)
- Vector store connection: ~1-2 seconds
- Embedding model loading: ~2-3 seconds
- **Total**: ~3-5 seconds (one time per session)

### Per Stock Verification
- Single search: ~0.2-0.5 seconds
- With variations: ~0.5-1.0 seconds
- Batch of 10 stocks: ~3-5 seconds

### Memory Usage
- Embedding model: ~500MB
- Vector store connection: ~50MB
- **Total**: ~550MB (persistent)

## Configuration Tuning

### Adjusting Confidence Thresholds

```python
# In stock_verifier.py
CONFIDENCE_THRESHOLD_HIGH = 0.9   # More strict (fewer false positives)
CONFIDENCE_THRESHOLD_MEDIUM = 0.7
CONFIDENCE_THRESHOLD_LOW = 0.5
```

### Adjusting Search Parameters

```python
# Retrieve more results for better matching
TOP_K_RESULTS = 10  # Default: 5

# Try more search variations
MAX_SEARCH_ATTEMPTS = 5  # Default: 3
```

### Custom STT Error Patterns

```python
# Add to generate_name_variations()
substitutions = {
    '百': ['八', '伯', '白'],  # Add more variations
    '特': ['得', '德', '鐵'],
    # Add your custom patterns
}
```

## Testing

Run the built-in tests:

```bash
python stock_verifier.py
```

Example output:
```
================================================================================
Stock Verifier - Example Usage
================================================================================

Verifying stocks...

Test 1: {'stock_name': '騰訊', 'stock_code': '700'}
✓ 已驗證: 騰訊
Reasoning: Verified correct (85.3% confidence)
--------------------------------------------------------------------------------
Test 2: {'stock_name': '泡泡沬特', 'stock_code': None}
🔧 修正建議 (HIGH, 78.5%):
  名稱: 泡泡沬特 → 泡泡瑪特
Reasoning: Vector store correction (78.5% confidence): 名稱: 泡泡沬特 → 泡泡瑪特
--------------------------------------------------------------------------------
```

## Troubleshooting

### Issue: "Vector store not available"

**Solution:**
1. Check Ollama is running: `ollama list`
2. Verify embedding model: `ollama pull qwen3-embedding:8b`
3. Test Milvus connectivity
4. Check credentials in configuration

### Issue: Low confidence scores

**Causes:**
- Stock not in database
- Poor quality embeddings
- Data format mismatch

**Solutions:**
1. Lower confidence threshold
2. Add more stock data to Milvus
3. Verify data format in collection
4. Try different embedding model

### Issue: Wrong corrections

**Causes:**
- Similar stock names in database
- Confidence threshold too low

**Solutions:**
1. Increase confidence threshold
2. Improve data quality in Milvus
3. Add more context to queries
4. Review and filter results manually

### Issue: Slow performance

**Solutions:**
1. Reduce `TOP_K_RESULTS`
2. Use batch processing
3. Cache frequent queries
4. Optimize Milvus indexes

## Advanced Topics

### Custom Embedding Models

```python
# In stock_verifier.py
EMBEDDING_MODEL = "your-custom-model:tag"
```

### Multiple Vector Stores

```python
# Create separate instances
hk_store = StockVectorStore(collection_name="hk_stocks")
us_store = StockVectorStore(collection_name="us_stocks")

# Use specific store
result = verify_and_correct_stock(
    stock_name="AAPL",
    vector_store=us_store
)
```

### Custom Confidence Calculation

Modify the confidence calculation in `verify_and_correct_stock()`:

```python
# Change from:
confidence = 1.0 / (1.0 + score)

# To your custom formula:
confidence = custom_confidence_function(score, query, result)
```

## Best Practices

1. **Initialize Once**: Use `get_vector_store()` singleton for shared instance
2. **Batch When Possible**: Use `batch_verify_stocks()` for multiple stocks
3. **Set Appropriate Thresholds**: Balance precision vs recall
4. **Log Results**: Monitor correction accuracy over time
5. **Handle Edge Cases**: Check for None values before using results
6. **Update Database**: Keep Milvus data current
7. **Monitor Performance**: Track search times and optimize

## Future Enhancements

- [ ] Support for multiple languages
- [ ] Custom correction rules engine
- [ ] Machine learning feedback loop
- [ ] Correction history and analytics
- [ ] A/B testing framework
- [ ] Integration with more vector stores
- [ ] Fuzzy matching algorithms
- [ ] Real-time database updates

## License

Part of the test-diarization project.

## Support

For issues or questions:
1. Check logs for detailed errors
2. Review configuration settings
3. Test with example code
4. Verify Milvus data format

---

**Created**: November 4, 2025  
**Last Updated**: November 4, 2025  
**Version**: 1.0.0

