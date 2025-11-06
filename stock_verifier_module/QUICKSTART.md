# Quick Start Guide

Get started with the Stock Verifier Module in 5 minutes!

## Prerequisites

1. **Install Ollama**
   ```bash
   # Visit https://ollama.ai and install Ollama
   ```

2. **Pull the embedding model**
   ```bash
   ollama pull qwen3-embedding:8b
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

The fastest way to verify everything works:

```bash
cd stock_verifier_module
python test_runner.py
```

You should see output like:

```
================================================================================
STOCK VERIFIER TEST REPORT
================================================================================
Total Tests: 6
Passed:      6 ✅ PASSED
Pass Rate:   100.00%
================================================================================
```

## Basic Usage

### Example 1: Verify a Stock

```python
from stock_verifier_improved import verify_and_correct_stock

# The problematic case mentioned by the user
result = verify_and_correct_stock(
    stock_name="騰訊升認購證",
    stock_code="18138"
)

print(f"Original: {result.original_stock_name}")
print(f"Corrected: {result.corrected_stock_name}")
print(f"Confidence: {result.confidence:.1%}")
```

**Expected Output:**
```
Original: 騰訊升認購證
Corrected: 騰訊摩通六一購Ｂ
Confidence: 95.0%
```

### Example 2: Batch Processing

```python
from stock_verifier_improved import batch_verify_stocks

stocks = [
    {"stock_name": "騰訊", "stock_code": "700"},
    {"stock_name": "小米", "stock_code": "1810"},
]

results = batch_verify_stocks(stocks)

for result in results:
    print(f"{result.original_stock_name} -> {result.corrected_stock_name}")
```

### Example 3: Try All Examples

Run the comprehensive examples:

```bash
python example_usage.py
```

This will demonstrate:
- Basic verification
- Different search strategies
- Batch processing
- Code-only and name-only lookups
- Confidence thresholds
- Performance optimization
- And more!

## Adding Your Own Test Cases

Edit `test_cases.json` and add:

```json
{
  "id": "TC007",
  "description": "My custom test",
  "input": {
    "stock_code": "12345",
    "stock_name": "測試股票"
  },
  "expected": {
    "stock_code": "12345",
    "stock_name": "正確名稱"
  },
  "tags": ["custom"],
  "notes": "Testing my specific case"
}
```

Then run:

```bash
python test_runner.py --tags custom
```

## Search Strategies

The module supports three strategies:

### 1. Optimized (Default) ⭐ Recommended

```python
from stock_verifier_improved import verify_and_correct_stock, SearchStrategy

result = verify_and_correct_stock(
    stock_name="騰訊升認購證",
    stock_code="18138",
    strategy=SearchStrategy.OPTIMIZED  # Default
)
```

- **When to use:** Always use this for production
- **How it works:** Prioritizes exact code match, falls back to semantic search
- **Best for:** Warrants, derivatives, similar-named stocks

### 2. Semantic Only

```python
result = verify_and_correct_stock(
    stock_name="泡泡沬特",  # STT error
    strategy=SearchStrategy.SEMANTIC_ONLY
)
```

- **When to use:** When stock codes might be wrong or missing
- **How it works:** Pure semantic similarity search
- **Best for:** STT transcription errors, fuzzy matching

### 3. Exact Only

```python
result = verify_and_correct_stock(
    stock_name="騰訊控股",
    stock_code="00700",
    strategy=SearchStrategy.EXACT_ONLY
)
```

- **When to use:** When you need guaranteed exact matches only
- **How it works:** Only returns result if exact code match found
- **Best for:** High-precision validation

## Understanding Results

Every verification returns a `StockCorrectionResult` with:

```python
result = verify_and_correct_stock(stock_name="騰訊", stock_code="700")

# Key attributes
result.corrected_stock_name    # Corrected name (or None)
result.corrected_stock_code    # Corrected code (or None)
result.confidence              # 0.0 to 1.0
result.confidence_level        # "high", "medium", "low", "none"
result.correction_applied      # True if correction was applied
result.reasoning               # Explanation of the decision
result.search_strategy         # Strategy used
```

### Confidence Levels

- **High (≥80%)**: Very confident, safe to use
- **Medium (≥60%)**: Moderately confident, review recommended
- **Low (≥40%)**: Low confidence, manual review needed
- **None (<40%)**: Not confident enough to suggest

## Common Use Cases

### Use Case 1: Validate User Input

```python
def validate_stock(user_code, user_name):
    result = verify_and_correct_stock(
        stock_name=user_name,
        stock_code=user_code
    )
    
    if result.correction_applied and result.confidence_level == "high":
        return {
            "valid": False,
            "message": f"Did you mean {result.corrected_stock_name}?"
        }
    
    return {"valid": True}
```

### Use Case 2: Post-Process STT Output

```python
def fix_stt_errors(stt_stocks):
    results = batch_verify_stocks(
        stt_stocks,
        strategy=SearchStrategy.SEMANTIC_ONLY,
        confidence_threshold=0.5
    )
    
    return [
        {
            "code": r.corrected_stock_code or r.original_stock_code,
            "name": r.corrected_stock_name or r.original_stock_name,
            "confidence": r.confidence
        }
        for r in results
    ]
```

### Use Case 3: Enhance LLM Output

```python
def verify_llm_stocks(llm_output):
    stocks = llm_output.get("stocks", [])
    
    verified = []
    for stock in stocks:
        result = verify_and_correct_stock(
            stock_name=stock.get("name"),
            stock_code=stock.get("code")
        )
        
        verified.append({
            "name": result.corrected_stock_name or result.original_stock_name,
            "code": result.corrected_stock_code or result.original_stock_code,
            "confidence": result.confidence
        })
    
    return verified
```

## Performance Tips

### Tip 1: Reuse Vector Store

```python
from stock_verifier_improved import get_vector_store

# Initialize once
vector_store = get_vector_store()
vector_store.initialize()

# Use for all verifications
for stock in many_stocks:
    result = verify_and_correct_stock(
        stock_name=stock,
        vector_store=vector_store  # Reuse!
    )
```

### Tip 2: Use Batch Operations

```python
# ✅ Good: One batch operation
stocks = [{"stock_name": name} for name in stock_names]
results = batch_verify_stocks(stocks)

# ❌ Bad: Multiple individual operations
results = [verify_and_correct_stock(stock_name=name) for name in stock_names]
```

## Testing Specific Scenarios

### Test Only Critical Cases

```bash
python test_runner.py --tags critical
```

### Test Warrants

```bash
python test_runner.py --tags warrant
```

### Test with Different Strategy

```bash
python test_runner.py --strategy semantic_only
```

### Stop on First Failure

```bash
python test_runner.py --stop-on-failure
```

## Troubleshooting

### Problem: "Failed to initialize vector store"

**Solution:**
1. Check Ollama is running: `ollama list`
2. Pull embedding model: `ollama pull qwen3-embedding:8b`
3. Check Milvus endpoint is accessible

### Problem: Low confidence scores

**Solution:**
```python
# Increase top_k to get more candidates
result = verify_and_correct_stock(
    stock_name=name,
    stock_code=code,
    top_k=20  # Default is 10
)
```

### Problem: Wrong corrections

**Solution:**
```python
# Use stricter strategy
result = verify_and_correct_stock(
    stock_name=name,
    stock_code=code,
    strategy=SearchStrategy.EXACT_ONLY,
    confidence_threshold=0.8  # Higher threshold
)
```

## Next Steps

1. ✅ Run tests: `python test_runner.py`
2. ✅ Try examples: `python example_usage.py`
3. ✅ Add your own test cases in `test_cases.json`
4. ✅ Read full documentation: `README.md` and `USAGE.md`
5. ✅ Integrate into your application

## Getting Help

- **Test reports**: Check `test_report.json` for detailed results
- **Console logs**: Set `logging.basicConfig(level=logging.INFO)` for debug info
- **Documentation**: See `README.md` for overview, `USAGE.md` for API details

## Summary

You now know how to:
- ✅ Run tests to verify accuracy
- ✅ Verify stocks with different strategies
- ✅ Add and test your own cases
- ✅ Understand confidence levels
- ✅ Optimize for performance

**Ready to use in production!** 🚀




