# Stock Name Similarity Testing Guide

## Quick Start Testing

### 1. Prerequisites Check

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull the embedding model
ollama pull qwen3-embedding:8b

# Verify the model is available
ollama list | grep qwen3-embedding
```

### 2. Start the Application

```bash
# Run the unified GUI
python unified_gui.py
```

### 3. Navigate to Trade Verification Tab

In the UI: **🔍 Trade Verification**

## Test Scenarios

### Test Case 1: Exact Match (Baseline)

**Transaction JSON:**
```json
{
  "transactions": [
    {
      "transaction_type": "sell",
      "stock_code": "3337",
      "stock_name": "安東油田服務",
      "quantity": "20000",
      "price": "1.23",
      "hkt_datetime": "2025-10-09T09:30:52"
    }
  ]
}
```

**Metadata JSON:**
```json
{
  "metadata": {
    "client_id": "M9136",
    "broker_id": "0489",
    "hkt_datetime": "2025-10-09T09:30:52"
  }
}
```

**Expected Result:**
- ✅ Stock code matches
- ✅ Stock name exact match (1.00)
- High confidence score (80%+)

---

### Test Case 2: Stock Name Variation (Partial Match)

**Transaction JSON:**
```json
{
  "transactions": [
    {
      "transaction_type": "sell",
      "stock_code": "3337",
      "stock_name": "安東油田",
      "quantity": "20000",
      "price": "1.23",
      "hkt_datetime": "2025-10-09T09:30:52"
    }
  ]
}
```

**Expected Result:**
- ✅ Stock code matches
- ✅ Stock name partial match (0.90) - "安東油田" is in "安東油田服務"
- High confidence score

---

### Test Case 3: Code Mismatch with High Name Similarity

**Scenario:** STT mishears stock code but gets name mostly right

**Transaction JSON:**
```json
{
  "transactions": [
    {
      "transaction_type": "sell",
      "stock_code": "3338",
      "stock_name": "安東油田服務集團",
      "quantity": "20000",
      "price": "1.23",
      "hkt_datetime": "2025-10-09T09:30:52"
    }
  ]
}
```

**Expected Result:**
- ❌ Stock code doesn't match (3338 vs 3337)
- ✅ Stock name high similarity (≥0.8)
- ⚠️ Warning: "possible STT error in code"
- Medium-high confidence (with +15 bonus)

---

### Test Case 4: Both Code and Name Mismatch

**Transaction JSON:**
```json
{
  "transactions": [
    {
      "transaction_type": "sell",
      "stock_code": "175",
      "stock_name": "吉利汽車",
      "quantity": "2000",
      "price": "19.63",
      "hkt_datetime": "2025-10-09T09:31:31"
    }
  ]
}
```

**Expected Result:**
- ❌ Stock code doesn't match (looking at record for M9136)
- ❌ Stock name low similarity
- Low confidence score

---

### Test Case 5: Multiple Trades (Find Best Match)

**Transaction JSON:**
```json
{
  "transactions": [
    {
      "transaction_type": "sell",
      "stock_code": "358",
      "stock_name": "江西銅業",
      "quantity": "6000",
      "price": "34.7",
      "hkt_datetime": "2025-10-09T09:32:45"
    }
  ]
}
```

**Metadata JSON:**
```json
{
  "metadata": {
    "client_id": "M57727",
    "broker_id": "0489",
    "hkt_datetime": "2025-10-09T09:32:45"
  }
}
```

**Expected Result:**
- Should find the sell (A) trade at 09:32:45 with stock 358
- Stock name "江西銅業股份" should match "江西銅業" with high similarity
- High confidence match

---

## What to Look For

### In Text Output

1. **Stock Name Similarity Messages:**
   ```
   ✅ MATCHES:
     ✅ High similarity (0.88): '騰訊認購證' ↔ '騰訊認購權證'
   
   ⚠️ PARTIAL MATCHES:
     ⚠️ Stock codes don't match, but names are very similar 
        (similarity: 0.88) - possible STT error in code
   ```

2. **Trade Record Details:**
   ```
   📄 Trade Record Details:
     Stock Code: 3337
     Stock Name: 安東油田服務  ← Should be visible now
   ```

3. **Confidence Scores:**
   - With name match: Should be higher
   - With name + code match: Very high (80%+)
   - With name match only: Medium-high (60-75%)

### In JSON Output

```json
{
  "match_index": 1,
  "confidence_score": 72.5,
  "stock_name_similarity": 0.88,  ← Check this field
  "comparison": {
    "matches": {
      "stock_name": "✅ High similarity (0.88): ..."
    }
  }
}
```

## Debugging

### Check Embeddings are Working

Add this to your test:

1. Watch the console/logs for:
   ```
   ✅ Embeddings model initialized successfully
   ```

2. If you see:
   ```
   WARNING: Embeddings not available. Stock name similarity will be disabled.
   ```
   Then embeddings are not working.

### Check Ollama Connection

```bash
# Test Ollama embedding endpoint
curl http://localhost:11434/api/embeddings \
  -d '{
    "model": "qwen3-embedding:8b",
    "prompt": "test"
  }'
```

Expected response: JSON with embedding array

### Common Issues

1. **Similarity always 0.0**
   - Embeddings not available
   - Check Ollama is running
   - Check model is pulled

2. **Slow performance**
   - First embedding generation is slower (model loading)
   - Subsequent calls should be faster
   - Check Ollama logs for issues

3. **Unexpected similarity scores**
   - Traditional vs Simplified Chinese mismatch
   - Extra whitespace in names
   - Model may need warm-up

## Performance Benchmarks

Expected timing (approximate):

| Operation | Time |
|-----------|------|
| First embedding (cold start) | 200-500ms |
| Subsequent embeddings | 50-200ms |
| Similarity computation | < 1ms |
| Total per transaction | 100-400ms |

For 5 transactions with 3 trades each = ~15 comparisons = 2-6 seconds total

## Validation Checklist

- [ ] Stock name appears in trade record details
- [ ] Similarity scores appear in matches/mismatches
- [ ] High similarity (≥0.8) triggers "possible STT error" warning
- [ ] Confidence scores include stock name matching
- [ ] JSON output includes `stock_name_similarity` field
- [ ] System works even without stock_name (graceful degradation)
- [ ] System works without embeddings (falls back to exact match)
- [ ] No crashes or errors in logs

## Advanced Testing

### Test with Real Audio

1. Process audio file with STT
2. Run transaction analysis (LLM)
3. Verify transaction JSON has stock_name
4. Run trade verification
5. Check if name similarity helps find matches

### Test Edge Cases

```json
// Empty stock name
{"stock_name": ""}

// Special characters
{"stock_name": "H股 - 中國銀行"}

// Numbers in name
{"stock_name": "中國移動2018認購證"}

// English names
{"stock_name": "HSBC Holdings"}

// Mixed language
{"stock_name": "騰訊 Tencent"}
```

## Reporting Issues

If you find issues, please note:

1. Transaction JSON used
2. Trade record being compared
3. Expected vs actual similarity score
4. Confidence score impact
5. Console/log messages
6. Ollama version and model version

## Next Steps After Testing

1. **Adjust Thresholds** if needed:
   - High similarity: Currently ≥0.8
   - Medium similarity: Currently ≥0.6
   - Bonus points: Currently +15

2. **Tune Weights** in confidence calculation:
   - Stock name: Currently 20.0
   - Stock code: Currently 20.0

3. **Add Caching** for frequently seen stocks

4. **Monitor Performance** in production

---

**Created**: 2025-11-06  
**Purpose**: Testing guide for stock name similarity feature  
**Module**: `tabs/tab_trade_verification.py`

