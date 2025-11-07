# Complete Tab Chaining Guide

## Overview

The Phone Call Analysis Suite now supports **complete end-to-end chaining** across 4 tabs, allowing data to flow seamlessly from audio transcription to trade verification.

## 🔗 The Complete Chain

```
┌─────────────────────┐
│  1. STT Tab         │  Upload Audio → Generate Transcription
│  (Speech-to-Text)   │
└──────────┬──────────┘
           │ Conversation JSON
           ↓
┌─────────────────────┐
│  2. JSON Batch      │  Extract Stocks → Deduplicate & Merge
│     Analysis        │
└──────────┬──────────┘
           │ Conversation JSON + Merged Stocks JSON
           ↓
┌─────────────────────┐
│  3. Transaction     │  Identify Transactions → Add Metadata
│     Analysis JSON   │
└──────────┬──────────┘
           │ Transaction JSON
           ↓
┌─────────────────────┐
│  4. Trade           │  Verify Against Trades.csv
│     Verification    │
└─────────────────────┘
```

## 📊 Data Flow

### Chain 1: STT → JSON Batch Analysis
**Data**: `shared_conversation_json`

**Format**:
```json
[
  {
    "conversation_number": 1,
    "filename": "call1.wav",
    "metadata": {
      "hkt_datetime": "2025-10-20T10:15:30",
      "broker_name": "Dickson Lau",
      "broker_id": "B001",
      "client_name": "CHENG SUK HING",
      "client_id": "C123"
    },
    "transcriptions": {
      "sensevoice": "經紀: 你好\n客戶: 我想買騰訊",
      "whisperv3_cantonese": "經紀: 你好\n客戶: 我想買騰訊"
    }
  }
]
```

### Chain 2: JSON Batch Analysis → Transaction Analysis JSON
**Data**: 
- Input: `shared_conversation_json` (same as above)
- Output: `shared_merged_stocks_json`

**Merged Stocks Format**:
```json
{
  "stocks": [
    {
      "stock_number": "00700",
      "stock_name": "騰訊控股",
      "relevance_score": 0.85,
      "original_word": "買入騰訊",
      "corrected_stock_number": "00700",
      "corrected_stock_name": "騰訊控股",
      "correction_confidence": 1.0,
      "confidence": "high",
      "quantity": "1000",
      "price": "350.5"
    }
  ],
  "metadata": {
    "total_conversations": 2,
    "total_analyses": 4,
    "unique_stocks_found": 3
  }
}
```

### Chain 3: Transaction Analysis JSON → Trade Verification
**Data**: `shared_transaction_json`

**Transaction Format**:
```json
{
  "transactions": [
    {
      "transaction_type": "buy",
      "confidence_score": 0.95,
      "conversation_number": 1,
      "hkt_datetime": "2025-10-20T10:15:30",
      "broker_id": "B001",
      "broker_name": "Dickson Lau",
      "client_id": "C123",
      "client_name": "CHENG SUK HING",
      "stock_code": "00700",
      "stock_name": "騰訊控股",
      "quantity": "1000",
      "price": "350.5",
      "explanation": "..."
    }
  ],
  "conversation_analysis": "...",
  "overall_summary": "..."
}
```

## 🎯 How to Use

### Complete Workflow (4 Steps)

#### Step 1: Transcribe Audio
1. Go to **"3️⃣ Auto-Diarize & Transcribe"** tab
2. Upload your audio file(s)
3. Click **"🚀 Transcribe Audio"**
4. Wait for completion
5. ✅ JSON output appears at bottom

#### Step 2: Extract Stocks
1. Switch to **"🔟 JSON Batch Analysis"** tab
2. Click **"📥 Load from STT Tab"**
3. Configure LLM settings (optional)
4. Click **"🚀 Analyze All Conversations"**
5. ✅ Merged stocks JSON appears

#### Step 3: Analyze Transactions
1. Switch to **"📊 Transaction Analysis (JSON)"** tab
2. Click **"📥 Load Conversation from Previous Tab"**
3. Click **"📥 Load Stocks from Previous Tab"**
4. Configure LLM settings (optional)
5. Click **"🚀 Analyze Transactions"**
6. ✅ Transaction JSON appears

#### Step 4: Verify Trades
1. Switch to **"🔍 Trade Verification"** tab
2. Click **"📥 Load Transactions from Previous Tab"**
3. Configure settings (trades.csv path, time window)
4. Click **"🔍 Verify Transactions"**
5. ✅ Verification results appear

### Quick Workflow (One-Click Per Tab)

After setting up your settings once:
1. Upload audio → Click "Transcribe"
2. Click "Load" → Click "Analyze"
3. Click "Load" (×2) → Click "Analyze"
4. Click "Load" → Click "Verify"

**Done!** Complete pipeline executed in 4 clicks.

## 🔑 Key Features

### Automatic Data Transfer
- ✅ No manual copy/paste needed
- ✅ Data preserved perfectly between tabs
- ✅ All metadata maintained

### Load Buttons
Each tab has clear "📥 Load from Previous Tab" buttons:
- JSON Batch Analysis: Loads conversation JSON
- Transaction Analysis JSON: Loads conversation + stocks JSON
- Trade Verification: Loads transaction JSON

### Manual Override
You can still manually paste JSON if needed:
- ✅ Load buttons don't replace manual input
- ✅ Useful for testing specific data
- ✅ Flexible workflow

### Multiple Files Support
Process multiple audio files at once:
- ✅ All conversations maintained in JSON
- ✅ Stocks deduplicated across conversations
- ✅ Transactions tracked by conversation number

## 📝 Example Scenario

**Input**: 3 audio files from phone calls

1. **STT Tab**: Processes all 3 files
   - Output: JSON with 3 conversations

2. **JSON Batch Analysis**: Extracts stocks from all 3
   - Found: 騰訊 (00700), 阿里巴巴 (09988), 比亞迪 (01211)
   - Output: Merged JSON with 3 unique stocks

3. **Transaction Analysis JSON**: Identifies transactions
   - Conversation 1: Buy 騰訊 1000 shares
   - Conversation 2: Sell 阿里巴巴 500 shares
   - Conversation 3: No transaction (just inquiry)
   - Output: JSON with 2 transactions

4. **Trade Verification**: Verifies against trades.csv
   - Transaction 1: ✅ Matched (found in trades.csv)
   - Transaction 2: ❌ Not found (possible issue)
   - Output: Verification report

## 🛠️ Technical Details

### Shared States (Gradio)

```python
# In unified_gui.py
shared_conversation_json = gr.State(None)   # STT → JSON Batch Analysis
shared_merged_stocks_json = gr.State(None)  # JSON Batch Analysis → Transaction Analysis
shared_transaction_json = gr.State(None)    # Transaction Analysis → Trade Verification
```

### Wrapper Functions

Each tab uses wrapper functions to duplicate outputs:

```python
# Example: JSON Batch Analysis
def process_with_stock_state(*args):
    result = process_json_batch(*args)
    # result = (formatted_results, combined_json, merged_json, verification_results)
    return result + (result[2],)  # Duplicate merged_json for state
```

This ensures:
- Original function remains unchanged
- Output appears in textbox AND state
- Backward compatible

### Modified Tabs

| Tab | Input States | Output States | Load Buttons |
|-----|-------------|---------------|-------------|
| STT | None | conversation_json | 0 |
| JSON Batch Analysis | conversation_json | merged_stocks_json | 1 |
| Transaction Analysis JSON | conversation_json, merged_stocks_json | transaction_json | 2 |
| Trade Verification | transaction_json | None | 1 |

## 🐛 Troubleshooting

### "No data from previous tab"

**Cause**: Previous tab hasn't been run yet

**Solution**: 
1. Run each tab in sequence
2. Wait for completion before moving to next tab

### Old Data Appears

**Cause**: State wasn't updated

**Solution**: Re-run the previous tab to update state

### Data Looks Wrong

**Cause**: Incorrect tab sequence

**Solution**: Follow the correct order:
1. STT
2. JSON Batch Analysis
3. Transaction Analysis JSON
4. Trade Verification

### Can't Find Load Button

**Cause**: Tab doesn't support chaining (yet)

**Solution**: Check if the tab is in the chain (see table above)

## 📚 Related Documentation

- `QUICK_START_CHAINING.md` - Quick 3-step guide (original 2-tab chain)
- `TAB_CHAINING_GUIDE.md` - Detailed user documentation
- `CHAINING_SUMMARY.md` - Technical implementation details
- `TROUBLESHOOTING_TAB_CHAINING.md` - Error resolution guide

## 🧪 Testing

Run the automated test suite:

```bash
python test_chaining.py
```

Expected output:
```
✓ PASS: unified_gui.py state
✓ PASS: create_stt_tab signature
✓ PASS: create_json_batch_analysis_tab signature
✓ PASS: tab_stt.py wrapper function
✓ PASS: JSON Batch Analysis chaining
✓ PASS: Transaction Analysis JSON signature
✓ PASS: Trade Verification signature
✓ PASS: All shared states

Total: 8/8 tests passed
🎉 All tests passed!
```

## 🎉 Benefits

### Time Savings
- ⏱️ No manual data copying
- ⏱️ Faster workflow execution
- ⏱️ Reduced human error

### Data Integrity
- ✅ Perfect data transfer
- ✅ No formatting issues
- ✅ Complete metadata preservation

### User Experience
- 😊 Simple one-click loading
- 😊 Clear visual flow
- 😊 Intuitive interface

### Flexibility
- 🔄 Can use any tab independently
- 🔄 Can chain any combination
- 🔄 Manual input still available

---

**Implementation Date**: November 7, 2025  
**Version**: 2.0 (Complete 4-Tab Chain)  
**Status**: ✅ Complete, Tested, and Production Ready

