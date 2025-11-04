# Stock Verifier - Usage Guide

Comprehensive guide to using the Stock Verifier module in your applications.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [API Reference](#api-reference)
- [Advanced Examples](#advanced-examples)
- [Integration Examples](#integration-examples)
- [Best Practices](#best-practices)

## Installation

### Prerequisites

```bash
pip install langchain-ollama langchain-milvus
```

### Ollama Setup

Ensure Ollama is running with the embedding model:

```bash
ollama pull qwen3-embedding:8b
```

## Basic Usage

### Simple Verification

```python
from stock_verifier_improved import verify_and_correct_stock

# Verify a stock with both name and code
result = verify_and_correct_stock(
    stock_name="騰訊",
    stock_code="700"
)

# Access the results
if result.correction_applied:
    print(f"Original: {result.original_stock_name}")
    print(f"Corrected: {result.corrected_stock_name}")
    print(f"Confidence: {result.confidence:.1%}")
else:
    print("Stock verified - no correction needed")
```

### Using Different Strategies

```python
from stock_verifier_improved import verify_and_correct_stock, SearchStrategy

# Optimized strategy (default - recommended)
result = verify_and_correct_stock(
    stock_name="騰訊升認購證",
    stock_code="18138",
    strategy=SearchStrategy.OPTIMIZED  # Prioritizes exact code match
)

# Semantic only (for fuzzy matching)
result = verify_and_correct_stock(
    stock_name="泡泡沬特",  # STT error: 沬 should be 瑪
    stock_code=None,
    strategy=SearchStrategy.SEMANTIC_ONLY
)

# Exact only (strict matching)
result = verify_and_correct_stock(
    stock_name="騰訊控股",
    stock_code="00700",
    strategy=SearchStrategy.EXACT_ONLY
)
```

## API Reference

### Main Functions

#### `verify_and_correct_stock()`

Verify and correct a single stock.

```python
def verify_and_correct_stock(
    stock_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    vector_store: Optional[StockVectorStore] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_LOW,
    top_k: int = TOP_K_RESULTS,
    strategy: SearchStrategy = SearchStrategy.OPTIMIZED,
) -> StockCorrectionResult
```

**Parameters:**
- `stock_name` (str, optional): Original stock name from STT/LLM
- `stock_code` (str, optional): Original stock code from STT/LLM
- `vector_store` (StockVectorStore, optional): Reusable vector store instance
- `confidence_threshold` (float): Minimum confidence to apply correction (default: 0.4)
- `top_k` (int): Number of candidates to retrieve (default: 10)
- `strategy` (SearchStrategy): Search strategy to use (default: OPTIMIZED)

**Returns:**
- `StockCorrectionResult`: Object containing correction details

**Example:**
```python
result = verify_and_correct_stock(
    stock_name="騰訊升認購證",
    stock_code="18138",
    confidence_threshold=0.6,  # Higher threshold
    top_k=20,  # More candidates
    strategy=SearchStrategy.OPTIMIZED
)
```

#### `batch_verify_stocks()`

Verify multiple stocks in a single batch (more efficient).

```python
def batch_verify_stocks(
    stocks: List[Dict[str, Optional[str]]],
    vector_store: Optional[StockVectorStore] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_LOW,
    strategy: SearchStrategy = SearchStrategy.OPTIMIZED,
) -> List[StockCorrectionResult]
```

**Parameters:**
- `stocks` (List[Dict]): List of stock dictionaries with 'stock_name' and/or 'stock_code' keys
- `vector_store` (StockVectorStore, optional): Reusable vector store instance
- `confidence_threshold` (float): Minimum confidence threshold
- `strategy` (SearchStrategy): Search strategy to use

**Returns:**
- `List[StockCorrectionResult]`: List of correction results

**Example:**
```python
from stock_verifier_improved import batch_verify_stocks

stocks = [
    {"stock_name": "騰訊", "stock_code": "700"},
    {"stock_name": "小米", "stock_code": "1810"},
    {"stock_name": "阿里巴巴", "stock_code": "9988"},
]

results = batch_verify_stocks(stocks)

for stock, result in zip(stocks, results):
    print(f"{stock['stock_name']}: {result.corrected_stock_name} ({result.confidence:.1%})")
```

### Data Classes

#### `StockCorrectionResult`

Result object returned by verification functions.

```python
@dataclass
class StockCorrectionResult:
    original_stock_name: Optional[str]      # Input stock name
    original_stock_code: Optional[str]      # Input stock code (normalized)
    corrected_stock_name: Optional[str]     # Corrected name (if different)
    corrected_stock_code: Optional[str]     # Corrected code (if different)
    confidence: float                        # Confidence score (0.0 to 1.0)
    correction_applied: bool                 # Whether correction was applied
    confidence_level: str                    # "none", "low", "medium", "high"
    reasoning: str                           # Explanation of the decision
    matched_content: str                     # Preview of matched content
    metadata: Dict[str, Any]                 # Metadata from matched document
    search_strategy: str                     # Strategy used
    all_candidates: List[Dict[str, Any]]    # Top candidates (for debugging)
```

**Example Usage:**
```python
result = verify_and_correct_stock(stock_name="騰訊", stock_code="700")

# Access fields
print(f"Confidence: {result.confidence:.2%}")
print(f"Level: {result.confidence_level}")  # "high", "medium", "low", "none"
print(f"Applied: {result.correction_applied}")
print(f"Reasoning: {result.reasoning}")

# Get corrected values or original if no correction
final_name = result.corrected_stock_name or result.original_stock_name
final_code = result.corrected_stock_code or result.original_stock_code
```

#### `SearchStrategy` Enum

Available search strategies.

```python
class SearchStrategy(Enum):
    OPTIMIZED = "optimized"          # Exact code match + semantic (default)
    SEMANTIC_ONLY = "semantic_only"  # Pure semantic similarity
    EXACT_ONLY = "exact_only"        # Only exact matches
```

### Vector Store Management

#### `get_vector_store()`

Get or create global vector store instance (singleton).

```python
from stock_verifier_improved import get_vector_store

vector_store = get_vector_store()

# Check if initialized
if vector_store.is_available:
    print("Vector store ready")
else:
    if vector_store.initialize():
        print("Vector store initialized")
```

#### `StockVectorStore` Class

Manages connection to Milvus vector store.

```python
from stock_verifier_improved import StockVectorStore

# Create custom instance
vector_store = StockVectorStore(
    endpoint="your-endpoint",
    token="your-token",
    db_name="your-db",
    collection_name="your-collection",
    embedding_model="qwen3-embedding:8b"
)

# Initialize
if vector_store.initialize():
    # Use in verifications
    result = verify_and_correct_stock(
        stock_name="騰訊",
        stock_code="700",
        vector_store=vector_store
    )
```

### Utility Functions

#### `format_correction_summary()`

Format result as readable summary.

```python
from stock_verifier_improved import format_correction_summary

result = verify_and_correct_stock(stock_name="騰訊", stock_code="700")
summary = format_correction_summary(result)
print(summary)
# Output:
# ✓ 已驗證: 騰訊
# or
# 🔧 修正建議 (HIGH, 95.0%):
#   名稱: 騰訊升認購證 → 騰訊摩通六一購Ｂ
```

#### `normalize_stock_code()`

Normalize stock code to standard format.

```python
from stock_verifier_improved import normalize_stock_code

code = normalize_stock_code("700")
print(code)  # Output: 00700

code = normalize_stock_code("18138")
print(code)  # Output: 18138
```

## Advanced Examples

### Reusing Vector Store for Multiple Verifications

```python
from stock_verifier_improved import verify_and_correct_stock, get_vector_store

# Initialize once
vector_store = get_vector_store()
vector_store.initialize()

# Use for multiple verifications (more efficient)
stocks = ["騰訊", "小米", "阿里巴巴"]

for stock_name in stocks:
    result = verify_and_correct_stock(
        stock_name=stock_name,
        vector_store=vector_store  # Reuse connection
    )
    print(f"{stock_name} -> {result.corrected_stock_name}")
```

### Custom Confidence Thresholds

```python
# High precision (only very confident corrections)
result = verify_and_correct_stock(
    stock_name="騰訊",
    stock_code="700",
    confidence_threshold=0.8  # Only apply corrections with 80%+ confidence
)

# Low threshold (accept more corrections)
result = verify_and_correct_stock(
    stock_name="泡泡沬特",  # STT error
    confidence_threshold=0.3  # More lenient
)
```

### Handling Different Input Formats

```python
# Name only
result = verify_and_correct_stock(stock_name="騰訊控股")

# Code only
result = verify_and_correct_stock(stock_code="700")

# Both name and code
result = verify_and_correct_stock(
    stock_name="騰訊",
    stock_code="700"
)

# Partial name
result = verify_and_correct_stock(stock_name="騰訊")  # Will find "騰訊控股"
```

### Analyzing All Candidates

```python
result = verify_and_correct_stock(
    stock_name="騰訊升認購證",
    stock_code="18138"
)

# Check all candidates (useful for debugging)
print("Top candidates:")
for i, candidate in enumerate(result.all_candidates, 1):
    print(f"{i}. Confidence: {candidate['confidence']:.2%}")
    print(f"   Match Type: {candidate.get('match_type')}")
    print(f"   Query: {candidate.get('query')}")
    
    # Extract info from candidate
    doc = candidate['doc']
    if hasattr(doc, 'metadata'):
        print(f"   Code: {doc.metadata.get('InstrumentCd')}")
        print(f"   Name: {doc.metadata.get('AliasName')}")
```

## Integration Examples

### Integration with STT Pipeline

```python
from stock_verifier_improved import verify_and_correct_stock, batch_verify_stocks

def process_stt_output(transcription: str) -> dict:
    """Process STT transcription and correct stock mentions"""
    
    # Extract stocks from transcription (your extraction logic)
    detected_stocks = extract_stocks_from_text(transcription)
    
    # Verify and correct
    results = batch_verify_stocks(detected_stocks)
    
    # Build corrected output
    corrected_stocks = []
    for result in results:
        if result.correction_applied:
            corrected_stocks.append({
                'code': result.corrected_stock_code,
                'name': result.corrected_stock_name,
                'confidence': result.confidence,
                'original_name': result.original_stock_name,
            })
        else:
            corrected_stocks.append({
                'code': result.original_stock_code,
                'name': result.original_stock_name,
                'confidence': result.confidence,
            })
    
    return {
        'transcription': transcription,
        'stocks': corrected_stocks
    }
```

### Integration with LLM Analysis

```python
from stock_verifier_improved import verify_and_correct_stock

def enhance_llm_output(llm_analysis: dict) -> dict:
    """Enhance LLM output with verified stock information"""
    
    extracted_stocks = llm_analysis.get('stocks', [])
    
    verified_stocks = []
    for stock in extracted_stocks:
        result = verify_and_correct_stock(
            stock_name=stock.get('name'),
            stock_code=stock.get('code')
        )
        
        verified_stocks.append({
            'name': result.corrected_stock_name or result.original_stock_name,
            'code': result.corrected_stock_code or result.original_stock_code,
            'confidence': result.confidence,
            'verification_applied': result.correction_applied,
            'reasoning': result.reasoning,
        })
    
    llm_analysis['verified_stocks'] = verified_stocks
    return llm_analysis
```

### Building a REST API

```python
from flask import Flask, request, jsonify
from stock_verifier_improved import verify_and_correct_stock, SearchStrategy

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify_stock():
    """Verify stock endpoint"""
    data = request.json
    
    result = verify_and_correct_stock(
        stock_name=data.get('stock_name'),
        stock_code=data.get('stock_code'),
        strategy=SearchStrategy[data.get('strategy', 'OPTIMIZED').upper()],
        confidence_threshold=data.get('confidence_threshold', 0.4)
    )
    
    return jsonify({
        'original': {
            'name': result.original_stock_name,
            'code': result.original_stock_code
        },
        'corrected': {
            'name': result.corrected_stock_name,
            'code': result.corrected_stock_code
        },
        'correction_applied': result.correction_applied,
        'confidence': result.confidence,
        'confidence_level': result.confidence_level,
        'reasoning': result.reasoning,
        'strategy': result.search_strategy
    })

if __name__ == '__main__':
    app.run(debug=True)
```

## Best Practices

### 1. Initialize Once, Use Many Times

```python
# ✅ Good: Reuse vector store
vector_store = get_vector_store()
vector_store.initialize()

for stock in stocks:
    result = verify_and_correct_stock(stock_name=stock, vector_store=vector_store)

# ❌ Bad: Re-initialize every time
for stock in stocks:
    result = verify_and_correct_stock(stock_name=stock)  # Creates new connection each time
```

### 2. Use Batch Operations When Possible

```python
# ✅ Good: Batch processing
stocks = [{"stock_name": name} for name in stock_names]
results = batch_verify_stocks(stocks)

# ❌ Less efficient: Individual calls
results = [verify_and_correct_stock(stock_name=name) for name in stock_names]
```

### 3. Choose Appropriate Strategy

```python
# ✅ Good: Use optimized for production
result = verify_and_correct_stock(
    stock_name="騰訊升認購證",
    stock_code="18138",
    strategy=SearchStrategy.OPTIMIZED  # Prioritizes exact code match
)

# ❌ Less accurate: Semantic only when you have reliable codes
result = verify_and_correct_stock(
    stock_name="騰訊升認購證",
    stock_code="18138",
    strategy=SearchStrategy.SEMANTIC_ONLY  # Might match wrong stock
)
```

### 4. Set Appropriate Confidence Thresholds

```python
# For critical applications (high precision)
result = verify_and_correct_stock(
    stock_name=name,
    confidence_threshold=0.8  # Only very confident corrections
)

# For exploratory analysis (high recall)
result = verify_and_correct_stock(
    stock_name=name,
    confidence_threshold=0.3  # More suggestions
)
```

### 5. Handle Errors Gracefully

```python
try:
    result = verify_and_correct_stock(
        stock_name=stock_name,
        stock_code=stock_code
    )
    
    if result.correction_applied:
        # Use corrected values
        final_code = result.corrected_stock_code
        final_name = result.corrected_stock_name
    else:
        # Use original values
        final_code = result.original_stock_code
        final_name = result.original_stock_name
        
except Exception as e:
    logging.error(f"Verification failed: {str(e)}")
    # Fallback to original values
    final_code = stock_code
    final_name = stock_name
```

### 6. Log for Debugging

```python
import logging

logging.basicConfig(level=logging.INFO)

result = verify_and_correct_stock(
    stock_name="騰訊升認購證",
    stock_code="18138"
)

# The module logs important events:
# - Vector store initialization
# - Exact matches found
# - Search fallbacks
# - Verification results
```

## Common Patterns

### Pattern 1: Validate User Input

```python
def validate_user_stock_input(user_code: str, user_name: str) -> dict:
    """Validate and correct user input"""
    result = verify_and_correct_stock(
        stock_name=user_name,
        stock_code=user_code,
        strategy=SearchStrategy.OPTIMIZED,
        confidence_threshold=0.6
    )
    
    if result.correction_applied and result.confidence_level in ['high', 'medium']:
        # Suggest correction to user
        return {
            'valid': False,
            'suggestion': {
                'code': result.corrected_stock_code,
                'name': result.corrected_stock_name
            },
            'message': f"Did you mean {result.corrected_stock_name}?"
        }
    
    return {'valid': True, 'code': user_code, 'name': user_name}
```

### Pattern 2: STT Post-Processing

```python
def post_process_stt(stt_stocks: List[dict]) -> List[dict]:
    """Post-process STT output to fix transcription errors"""
    
    # Use semantic search for STT errors
    results = batch_verify_stocks(
        stt_stocks,
        strategy=SearchStrategy.SEMANTIC_ONLY,  # Good for fuzzy matching
        confidence_threshold=0.5
    )
    
    corrected = []
    for original, result in zip(stt_stocks, results):
        corrected.append({
            'code': result.corrected_stock_code or result.original_stock_code,
            'name': result.corrected_stock_name or result.original_stock_name,
            'confidence': result.confidence,
            'was_corrected': result.correction_applied
        })
    
    return corrected
```

### Pattern 3: Progressive Verification

```python
def progressive_verification(stock_name: str, stock_code: str) -> dict:
    """Try multiple strategies progressively"""
    
    # Try exact match first
    result = verify_and_correct_stock(
        stock_name=stock_name,
        stock_code=stock_code,
        strategy=SearchStrategy.EXACT_ONLY
    )
    
    if result.correction_applied and result.confidence > 0.9:
        return {'code': result.corrected_stock_code, 'name': result.corrected_stock_name}
    
    # Fall back to optimized
    result = verify_and_correct_stock(
        stock_name=stock_name,
        stock_code=stock_code,
        strategy=SearchStrategy.OPTIMIZED
    )
    
    if result.correction_applied and result.confidence > 0.7:
        return {'code': result.corrected_stock_code, 'name': result.corrected_stock_name}
    
    # Use original
    return {'code': stock_code, 'name': stock_name}
```

## Troubleshooting

### Issue: Low Confidence Scores

```python
# Increase top_k to get more candidates
result = verify_and_correct_stock(
    stock_name=name,
    stock_code=code,
    top_k=20  # Default is 10
)

# Check all candidates to see what's available
for candidate in result.all_candidates:
    print(f"Confidence: {candidate['confidence']:.2%}")
    print(f"Type: {candidate.get('match_type')}")
```

### Issue: Wrong Corrections

```python
# Use exact-only strategy if codes are reliable
result = verify_and_correct_stock(
    stock_name=name,
    stock_code=code,
    strategy=SearchStrategy.EXACT_ONLY
)

# Increase confidence threshold
result = verify_and_correct_stock(
    stock_name=name,
    stock_code=code,
    confidence_threshold=0.8  # Higher threshold
)
```

### Issue: Vector Store Connection Errors

```python
from stock_verifier_improved import get_vector_store

vector_store = get_vector_store()

# Check connection
if not vector_store.is_available:
    success = vector_store.initialize()
    if not success:
        print("Failed to connect to vector store")
        # Check:
        # 1. Ollama is running
        # 2. qwen3-embedding:8b model is available
        # 3. Milvus endpoint is accessible
```

## Performance Optimization

### Benchmark Different Strategies

```python
import time

strategies = [
    SearchStrategy.OPTIMIZED,
    SearchStrategy.SEMANTIC_ONLY,
    SearchStrategy.EXACT_ONLY
]

for strategy in strategies:
    start = time.time()
    
    result = verify_and_correct_stock(
        stock_name="騰訊",
        stock_code="700",
        strategy=strategy
    )
    
    elapsed = (time.time() - start) * 1000
    print(f"{strategy.value}: {elapsed:.2f}ms, confidence: {result.confidence:.2%}")
```

### Measure Batch Performance

```python
import time

stocks = [{"stock_name": f"Stock{i}", "stock_code": str(i)} for i in range(100)]

start = time.time()
results = batch_verify_stocks(stocks)
elapsed = time.time() - start

print(f"Processed {len(stocks)} stocks in {elapsed:.2f}s")
print(f"Average: {elapsed / len(stocks) * 1000:.2f}ms per stock")
```

