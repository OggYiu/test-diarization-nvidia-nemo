# Stock Verifier Module

An improved stock name and code verifier using Milvus vector store with enhanced search strategies for better accuracy.

## 🎯 Key Features

- **Prioritized Exact Matching**: When stock code is provided, prioritizes exact code matches over semantic similarity
- **Multiple Search Strategies**: Choose between optimized (default), semantic-only, or exact-only strategies
- **Flexible Test Framework**: JSON-based test cases that are easy to add and maintain
- **Detailed Reporting**: Comprehensive test reports with confidence scores and reasoning
- **Better Warrant/Derivative Handling**: Improved accuracy for similar-named securities

## 📁 Module Structure

```
stock_verifier_module/
├── stock_verifier_improved.py  # Main verifier with improved search strategies
├── test_cases.json              # Flexible test case definitions
├── test_runner.py               # Test execution and reporting framework
├── README.md                    # This file
├── USAGE.md                     # Usage examples and API documentation
└── test_report.json            # Generated test report (after running tests)
```

## 🚀 Quick Start

### 1. Running Tests

```bash
# Run all tests with default optimized strategy
python test_runner.py

# Run with specific strategy
python test_runner.py --strategy optimized
python test_runner.py --strategy semantic_only
python test_runner.py --strategy exact_only

# Run only specific test cases by tags
python test_runner.py --tags critical
python test_runner.py --tags warrant exact_code_match

# Stop on first failure (useful for debugging)
python test_runner.py --stop-on-failure

# Specify custom test file and output
python test_runner.py --test-file my_tests.json --output my_report.json
```

### 2. Using in Your Code

```python
from stock_verifier_improved import verify_and_correct_stock, SearchStrategy

# Basic usage with optimized strategy (default)
result = verify_and_correct_stock(
    stock_name="騰訊升認購證",
    stock_code="18138"
)

print(f"Corrected: {result.corrected_stock_name}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Reasoning: {result.reasoning}")

# Use specific strategy
result = verify_and_correct_stock(
    stock_name="騰訊",
    stock_code="700",
    strategy=SearchStrategy.EXACT_ONLY
)

# Batch verification
from stock_verifier_improved import batch_verify_stocks

stocks = [
    {"stock_name": "騰訊", "stock_code": "700"},
    {"stock_name": "小米", "stock_code": "1810"},
]

results = batch_verify_stocks(stocks, strategy=SearchStrategy.OPTIMIZED)
for result in results:
    print(f"{result.original_stock_name} -> {result.corrected_stock_name}")
```

## 📝 Adding Test Cases

Test cases are defined in `test_cases.json`. To add a new test case:

```json
{
  "id": "TC007",
  "description": "Your test description",
  "input": {
    "stock_code": "12345",
    "stock_name": "測試股票"
  },
  "expected": {
    "stock_code": "12345",
    "stock_name": "正確股票名稱"
  },
  "tags": ["your_tag", "another_tag"],
  "notes": "Any additional notes about this test case"
}
```

### Test Case Structure

- **id**: Unique identifier (e.g., TC001, TC002)
- **description**: Human-readable description
- **input**: The input data to test
  - `stock_code`: Input stock code (can be null)
  - `stock_name`: Input stock name (can be null)
- **expected**: Expected output after correction
  - `stock_code`: Expected corrected code
  - `stock_name`: Expected corrected name
- **tags**: Array of tags for filtering (e.g., "critical", "warrant", "stt_error")
- **notes**: Additional context or explanation

### Useful Tags

- `critical`: Tests that must pass
- `warrant`: Warrant/derivative securities
- `exact_code_match`: Tests requiring exact code matching
- `stt_error`: Common STT transcription errors
- `basic`: Basic stock lookups
- `code_only`: Tests with code but no name
- `name_only`: Tests with name but no code

## 🔍 Search Strategies

### 1. Optimized (Default - Recommended)

```python
strategy=SearchStrategy.OPTIMIZED
```

**How it works:**
1. If stock code is provided, searches for exact code match first
2. If exact match found, returns with high confidence (0.95)
3. If no exact match, falls back to semantic similarity search
4. Best for real-world usage with mixed quality data

**Use when:**
- You want the best balance of accuracy and recall
- You have stock codes that should be trusted
- Working with warrants or similar-named securities

### 2. Semantic Only

```python
strategy=SearchStrategy.SEMANTIC_ONLY
```

**How it works:**
1. Pure semantic similarity search using embeddings
2. Considers name variations for common transcription errors
3. Weights code searches at 0.9, name searches at 1.0

**Use when:**
- Stock codes might be unreliable or incorrect
- You want to find semantically similar stocks
- Testing STT transcription error handling

### 3. Exact Only

```python
strategy=SearchStrategy.EXACT_ONLY
```

**How it works:**
1. Only returns results if exact code match is found
2. No fallback to semantic search
3. Very high confidence when match found

**Use when:**
- You need guaranteed exact matches only
- Code must be absolutely correct
- Building high-precision validation systems

## 📊 Understanding Test Reports

### Console Report

The test runner prints a detailed console report:

```
================================================================================
STOCK VERIFIER TEST REPORT
================================================================================
Execution Time: 2025-11-04 12:30:45
Total Tests: 6
Strategy: optimized
================================================================================

TEST RESULTS:
--------------------------------------------------------------------------------

1. ✅ PASSED TC001: 騰訊認購證 - exact code match priority test
   Tags: warrant, exact_code_match, critical
   Input:    Code=18138, Name=騰訊升認購證
   Expected: Code=18138, Name=騰訊摩通六一購Ｂ
   Actual:   Code=18138, Name=騰訊摩通六一購Ｂ
   Confidence: 95.00% (high)
   Reasoning: exact_code match (95.0% confidence): 名稱: 騰訊升認購證 → 騰訊摩通六一購Ｂ
   Execution Time: 523.45 ms

...

================================================================================
SUMMARY
================================================================================
Total Tests:     6
Passed:          6 ✅ PASSED
Failed:          0 ❌ FAILED
Errors:          0 💥 ERROR
Skipped:         0 ⏭️  SKIPPED
Pass Rate:       100.00%
Total Time:      2156.78 ms
================================================================================
```

### JSON Report

A detailed JSON report is saved to `test_report.json`:

```json
{
  "timestamp": "2025-11-04T12:30:45.123456",
  "summary": {
    "total_tests": 6,
    "passed": 6,
    "failed": 0,
    "errors": 0,
    "skipped": 0,
    "pass_rate": 100.0,
    "execution_time_ms": 2156.78
  },
  "results": [
    {
      "test_id": "TC001",
      "description": "騰訊認購證 - exact code match priority test",
      "status": "✅ PASSED",
      "input_data": {
        "stock_code": "18138",
        "stock_name": "騰訊升認購證"
      },
      "expected": {
        "stock_code": "18138",
        "stock_name": "騰訊摩通六一購Ｂ"
      },
      "actual": {
        "stock_code": "18138",
        "stock_name": "騰訊摩通六一購Ｂ"
      },
      "correction_result": {
        "confidence": 0.95,
        "confidence_level": "high",
        "reasoning": "exact_code match (95.0% confidence): 名稱: 騰訊升認購證 → 騰訊摩通六一購Ｂ"
      }
    }
  ]
}
```

## 🔧 Configuration

Configuration is stored in `test_cases.json` under `test_configuration`:

```json
{
  "test_configuration": {
    "default_strategy": "optimized",
    "confidence_threshold": 0.4,
    "top_k": 10,
    "available_strategies": ["optimized", "semantic_only", "exact_only"]
  }
}
```

## 🐛 Debugging Failed Tests

If a test fails:

1. **Check the confidence score**: Low confidence might indicate ambiguous data
2. **Review the reasoning**: See why the verifier made its choice
3. **Look at all_candidates**: Check other potential matches
4. **Verify the expected data**: Ensure expected values are correct
5. **Try different strategies**: See if another strategy performs better

Example of debugging:

```python
from stock_verifier_improved import verify_and_correct_stock, SearchStrategy

# Run with optimized strategy
result = verify_and_correct_stock(
    stock_name="騰訊升認購證",
    stock_code="18138",
    strategy=SearchStrategy.OPTIMIZED
)

print(f"Strategy: {result.search_strategy}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
print(f"\nTop candidates:")
for i, candidate in enumerate(result.all_candidates[:3], 1):
    print(f"{i}. Confidence: {candidate['confidence']:.2%}, Type: {candidate.get('match_type')}")
```

## 📈 Performance Tips

1. **Reuse vector store**: Initialize once for multiple verifications
2. **Use batch operations**: `batch_verify_stocks()` is more efficient
3. **Filter by tags**: Run only relevant tests during development
4. **Optimize K value**: Increase `top_k` if exact matches are missed

## 🔒 Security Notes

- The Milvus credentials are currently hardcoded in the module
- For production use, consider moving credentials to environment variables
- The module uses strong consistency for data accuracy

## 📚 Related Documentation

- See `USAGE.md` for detailed API documentation
- See `test_cases.json` for example test case format
- Check test reports for real performance data

## 🤝 Contributing

To add new features or test cases:

1. Add test cases to `test_cases.json`
2. Run tests to ensure they pass
3. Update documentation if adding new features
4. Review test reports for accuracy

## ❓ FAQ

**Q: Why is my test failing with high confidence?**  
A: The verifier found a strong match, but it doesn't match your expected values. Check if expected values are correct or if the vector store data is up to date.

**Q: What strategy should I use?**  
A: Use `OPTIMIZED` (default) for best results. It prioritizes exact code matches while falling back to semantic search.

**Q: How do I handle STT transcription errors?**  
A: Add character substitutions to `generate_name_variations()` in the verifier. Common substitutions are already included.

**Q: Can I run tests in CI/CD?**  
A: Yes! The test runner exits with code 0 on success and 1 on failure, making it CI/CD friendly.

## 📞 Support

For issues or questions:
1. Check the test reports for detailed error information
2. Review the console output for warnings
3. Verify vector store connectivity
4. Check test case format in JSON file

