# Stock Name Similarity Feature for Trade Verification

## Overview

The Trade Verification tab now includes **stock name similarity matching** using AI embeddings. This enhancement helps identify matching trades even when Speech-to-Text (STT) systems mis-transcribe stock codes or names, which is especially common in Cantonese audio processing.

## Problem Statement

When verifying transactions from call recordings against trade records, we often encounter:
1. **Stock Code Errors**: STT may mishear numbers (e.g., "18138" vs "10138")
2. **Stock Name Variations**: Names may be transcribed with slight variations
3. **Homophone Confusion**: Cantonese has many homophones that sound similar but are written differently

Previously, the system only matched stock codes by exact comparison (after normalizing leading zeros). If the STT made an error in transcribing the stock code, the trade would not be matched even if all other criteria aligned.

## Solution

The new feature uses **semantic embeddings** to compute similarity between stock names, providing a robust fallback when stock codes don't match perfectly.

### Key Features

1. **Embedding-Based Similarity**: Uses Ollama `qwen3-embedding:8b` model to compute cosine similarity between stock names
2. **Smart Matching Logic**:
   - Exact match (case-insensitive): 100% similarity
   - Partial match (substring): 90% similarity
   - Semantic similarity via embeddings: 0-100%

3. **Confidence Adjustment**:
   - Stock name similarity is now part of the confidence score calculation (20% weight)
   - Bonus points (+15%) when stock names are highly similar (≥0.8) even if codes don't match
   - Helps surface potential matches that would have been missed

4. **Clear Feedback**:
   - Shows stock name similarity scores in results
   - Flags cases where "codes don't match but names are very similar"
   - Helps identify likely STT errors

## How It Works

### Matching Algorithm

```python
# Step 1: Check stock code (exact match after normalization)
if tx_stock_code == trade_stock_code:
    ✅ Stock code matches
else:
    ❌ Stock code doesn't match
    
# Step 2: Compute stock name similarity
similarity = compute_text_similarity(tx_stock_name, trade_stock_name)

# Step 3: Interpret similarity
if similarity >= 0.8:
    ✅ High similarity - likely match
    if stock codes don't match:
        ⚠️ Possible STT error in stock code
elif similarity >= 0.6:
    ⚠️ Medium similarity - potential match
else:
    ❌ Low similarity - unlikely match

# Step 4: Adjust confidence score
confidence += stock_name_weight * similarity_category
if similarity >= 0.8 and codes_dont_match:
    confidence += 15  # Bonus for high name similarity
```

### Similarity Thresholds

| Similarity Score | Interpretation | Action |
|-----------------|----------------|--------|
| 1.0 | Exact match | ✅ Added to matches |
| 0.9 | Partial match (substring) | ✅ Added to matches |
| 0.8 - 0.89 | High similarity | ✅ Added to matches |
| 0.6 - 0.79 | Medium similarity | ⚠️ Partial match |
| < 0.6 | Low similarity | ❌ Mismatch |

## Usage

### Prerequisites

1. **Ollama with embedding model**:
   ```bash
   ollama pull qwen3-embedding:8b
   ```

2. **Required packages** (already in `requirements.txt`):
   - `langchain-ollama>=0.1.0`
   - `numpy>=1.21.0`

### Input Data Requirements

Your transaction JSON must include **stock_name**:

```json
{
  "transactions": [
    {
      "transaction_type": "buy",
      "stock_code": "18138",
      "stock_name": "騰訊認購證",  // REQUIRED for similarity matching
      "quantity": "5000",
      "price": "0.45",
      "hkt_datetime": "2025-10-20T10:15:30"
    }
  ]
}
```

The `trades.csv` file must have a **stock_name** column (which it already does).

### Example Scenarios

#### Scenario 1: Exact Code Match with Name Confirmation
```
Transaction: Code=3337, Name="安東油田服務"
Trade Record: Code=3337, Name="安東油田服務"

Result:
✅ Stock code matches
✅ Stock name exact match (1.00)
Confidence: High
```

#### Scenario 2: Code Mismatch with High Name Similarity (STT Error)
```
Transaction: Code=18138, Name="騰訊認購證"
Trade Record: Code=18128, Name="騰訊認購權證"

Result:
❌ Stock code doesn't match (18138 vs 18128)
✅ Stock name high similarity (0.88)
⚠️ Possible STT error in stock code
Confidence: Medium-High (bonus points applied)
```

#### Scenario 3: Both Code and Name Mismatch
```
Transaction: Code=3337, Name="安東油田"
Trade Record: Code=175, Name="吉利汽車"

Result:
❌ Stock code doesn't match
❌ Stock name low similarity (0.12)
Confidence: Low
```

## Benefits

### 1. Better Match Coverage
- Catches matches that would have been missed due to STT errors
- More robust against transcription variations

### 2. Error Detection
- Identifies likely transcription errors in stock codes
- Helps validate and correct transaction data

### 3. Cantonese-Friendly
- Handles homophone confusion common in Cantonese STT
- Semantic understanding beyond character matching

### 4. Confidence Transparency
- Shows similarity scores in results
- Helps users understand matching decisions

## Technical Details

### Embedding Model
- **Model**: `qwen3-embedding:8b` via Ollama
- **Dimensions**: 8192 (typically for 8b models)
- **Similarity Metric**: Cosine similarity
- **Language Support**: Chinese (Traditional/Simplified) and English

### Performance
- **Embedding Generation**: ~50-200ms per query (depends on Ollama setup)
- **Similarity Computation**: < 1ms
- **Caching**: Embedding model is initialized once and reused

### Fallback Behavior
If embeddings are not available:
- Feature degrades gracefully
- Only exact and substring matching is used
- No embedding-based similarity computed
- Warning logged but execution continues

## Configuration

### Field Weights in Confidence Calculation

The new weight distribution (totals 100%):

```python
field_weights = {
    "broker_id": 20.0,      # Broker match
    "stock_code": 20.0,     # Stock code match
    "stock_name": 20.0,     # 🆕 Stock name similarity
    "order_side": 12.0,     # Buy/Sell direction
    "quantity": 12.0,       # Order quantity
    "price": 8.0,           # Order price
    "time": 8.0,            # Time difference
}
base_confidence = 20.0      # Client/date/window match
```

### Bonus Points
- **+15 points**: High name similarity (≥0.8) when code doesn't match
- Helps surface potential matches with transcription errors

## Output Examples

### Text Output
```
📌 POTENTIAL MATCH #1 - Confidence: 72.5%
────────────────────────────────────────────────────────

📄 Trade Record Details:
  Stock Code: 3337
  Stock Name: 安東油田服務
  ...

✅ MATCHES:
  ✅ Broker ID 0489 matches AECode CK489
  ✅ 3337 matches 3337
  ✅ High similarity (0.95): '安東油田服務' ↔ '安東油田服務'
  ✅ buy matches OrderSide=B

⚠️ PARTIAL MATCHES:
  ⚠️ Time difference: 12.3 minutes
```

### JSON Output
```json
{
  "match_index": 1,
  "confidence_score": 72.5,
  "stock_name_similarity": 0.95,
  "comparison": {
    "matches": {
      "broker_id": "✅ Broker ID 0489 matches AECode CK489",
      "stock_code": "✅ 3337 matches 3337",
      "stock_name": "✅ High similarity (0.95): '安東油田服務' ↔ '安東油田服務'"
    },
    "mismatches": {},
    "partial_matches": {
      "time": "⚠️ Time difference: 12.3 minutes"
    }
  }
}
```

## Troubleshooting

### Issue: "Embeddings not available"
**Solution**: 
1. Install Ollama: `https://ollama.ai`
2. Pull embedding model: `ollama pull qwen3-embedding:8b`
3. Start Ollama service
4. Restart the application

### Issue: Slow performance
**Solution**:
1. Check Ollama service is running locally
2. Consider using GPU acceleration for Ollama
3. Monitor Ollama logs for performance issues

### Issue: Similarity scores seem incorrect
**Possible causes**:
1. Stock names have extra whitespace or special characters
2. Traditional vs Simplified Chinese mismatch
3. Model not suitable for domain-specific terminology

**Solution**: 
- Normalize stock names (trim whitespace)
- Use consistent character encoding
- Consider fine-tuning embeddings (advanced)

## Future Enhancements

Potential improvements:
1. **Phonetic Similarity**: Add Cantonese phonetic matching for better homophone handling
2. **Custom Embeddings**: Fine-tune embeddings on stock market terminology
3. **Fuzzy Code Matching**: Allow single-digit variations in stock codes with high name similarity
4. **Caching**: Cache embeddings for frequently seen stock names
5. **Batch Processing**: Generate embeddings in batches for better performance

## References

- Ollama Embeddings: https://ollama.ai/
- LangChain Ollama Integration: https://python.langchain.com/docs/integrations/text_embedding/ollama
- Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity

## Changelog

### Version 1.0 (Current)
- ✅ Added stock name similarity matching using embeddings
- ✅ Integrated similarity into confidence scoring
- ✅ Updated UI to show similarity scores
- ✅ Added bonus points for high similarity with code mismatch
- ✅ Graceful degradation when embeddings unavailable

---

**Created**: 2025-11-06  
**Author**: AI Assistant  
**Module**: `tabs/tab_trade_verification.py`

