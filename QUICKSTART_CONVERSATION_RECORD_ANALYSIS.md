# Quick Start Guide - Conversation Record Analysis

## 🚀 Get Started in 5 Minutes

### Step 1: Start Ollama (if not running)
```bash
ollama serve
```

### Step 2: Pull a recommended model (if needed)
```bash
# Option 1: Best accuracy (requires ~8GB RAM)
ollama pull qwen2.5:14b

# Option 2: Faster, less accurate (requires ~4GB RAM)
ollama pull qwen2.5:7b
```

### Step 3: Launch the GUI
```bash
cd c:\projects\test-diarization
python unified_gui.py
```

### Step 4: Open the new tab
Navigate to: **🎯 Conversation Record Analysis**

### Step 5: Load sample conversation
Copy the contents of `sample_conversation_for_record_analysis.json`:

```json
{
  "metadata": {
    "filename": "sample_call_20251009.wav",
    "broker_id": "0489",
    "client_id": "M9136",
    "hkt_datetime": "2025-10-09T09:30:00",
    "call_duration": "5:30",
    "recording_quality": "good"
  },
  "conversations": [
    [
      {
        "speaker": "Broker",
        "text": "喂，你好！"
      },
      {
        "speaker": "Client", 
        "text": "我想沽啲安東油田服務，三三三七。"
      },
      {
        "speaker": "Broker",
        "text": "安東油田服務，三三三七，沽幾多？"
      },
      {
        "speaker": "Client",
        "text": "兩萬股，一蚊二三沽。"
      },
      {
        "speaker": "Broker",
        "text": "好，兩萬股安東油田服務三三三七，一蚊二三沽，收到。"
      },
      {
        "speaker": "Client",
        "text": "係，唔該。"
      }
    ]
  ],
  "transcriptions": [
    {
      "model": "wsyue-asr",
      "text": "喂你好我想沽啲安東油田服務三三三七。安東油田服務三三三七沽幾多？兩萬股一蚊二三沽。好兩萬股安東油田服務三三三七一蚊二三沽收到。係唔該。",
      "confidence": 0.92
    }
  ]
}
```

Paste this into the **"Conversation JSON"** textbox.

### Step 6: Configure settings

Leave defaults as-is:
- **Trades CSV File Path**: `trades.csv`
- **Client ID Filter**: `M9136` (or leave empty to see all clients)
- **LLM Model**: `qwen2.5:14b`
- **Ollama API URL**: `http://localhost:11434`
- **Temperature**: `0.3`

### Step 7: Click "🎯 Analyze Records"

Wait 10-30 seconds for analysis to complete.

### Step 8: Review Results

You should see:

#### ✅ High Confidence (0.95) for Record #78239686
```
OrderNo: 78239686
Stock: 3337 (安東油田服務)
Side: Sell
Quantity: 20000
Price: 1.23
```

**Why?** This trade matches the conversation exactly:
- Stock: "安東油田服務三三三七" ✓
- Action: "沽" (sell) ✓
- Quantity: "兩萬股" (20000) ✓
- Price: "一蚊二三" (1.23) ✓

#### ❌ Low Confidence (~0.1) for other records
Other trades on that date won't match because they weren't discussed.

---

## 📋 What You'll See

### Formatted Text Output
```
================================================================================
📊 CONVERSATION RECORD ANALYSIS
================================================================================

📅 Date: 2025-10-09
👤 Client: M9136
📂 Trades File: trades.csv
🤖 Model: qwen2.5:14b
📋 Total Records Found: 10

================================================================================
💬 CONVERSATION SUMMARY
================================================================================
The conversation is a phone call between a broker and client discussing
the sale of 安東油田服務 (stock code 3337). The client requested to sell
20,000 shares at a price of HK$1.23, which the broker confirmed.

================================================================================
📈 OVERALL ASSESSMENT
================================================================================
Out of 10 trade records found for client M9136 on 2025-10-09, only 1 record
(OrderNo: 78239686) clearly matches the conversation with high confidence (0.95).
The remaining 9 records show no evidence of being discussed in this call and
should be flagged for review to ensure they were properly authorized.

================================================================================
📊 CONFIDENCE SUMMARY
================================================================================
average_confidence: 0.185
high_confidence_count: 1
medium_confidence_count: 0
low_confidence_count: 9

================================================================================
🔍 INDIVIDUAL RECORD ANALYSIS (10 records)
================================================================================

✅ Record #1 - OrderNo: 78239686
────────────────────────────────────────────────────────────────────────────────
Confidence Score: 0.95 (95%)

Reasoning:
This trade matches the conversation almost perfectly. The client explicitly
requested to sell 安東油田服務 (stock code 3337), quantity 20,000 shares at
price HK$1.23. The broker repeated and confirmed all these details. The only
minor difference is the exact time - the conversation was at 09:30:00 while
the trade was executed at 09:30:52, a 52-second difference which is
reasonable for order processing time.

Matched Conversation Segments:
  • 我想沽啲安東油田服務，三三三七
  • 兩萬股，一蚊二三沽
  • 好，兩萬股安東油田服務三三三七，一蚊二三沽，收到

────────────────────────────────────────────────────────────────────────────────

❌ Record #2 - OrderNo: 78239778
────────────────────────────────────────────────────────────────────────────────
Confidence Score: 0.05 (5%)

Reasoning:
This trade is for 吉利汽車 (stock code 175). There is no mention of this
stock anywhere in the conversation. The conversation only discusses
安東油田服務. This trade should be investigated.

Matched Conversation Segments:
  (none)

────────────────────────────────────────────────────────────────────────────────

... (8 more low-confidence records)
```

### JSON Output
```json
{
  "status": "success",
  "analysis_info": {
    "date": "2025-10-09",
    "client_id": "M9136",
    "trades_file": "trades.csv",
    "model": "qwen2.5:14b",
    "total_records": 10
  },
  "analysis_result": {
    "records_analyzed": [
      {
        "order_no": "78239686",
        "confidence_score": 0.95,
        "reasoning": "This trade matches the conversation almost perfectly...",
        "matched_conversation_segments": [
          "我想沽啲安東油田服務，三三三七",
          "兩萬股，一蚊二三沽",
          "好，兩萬股安東油田服務三三三七，一蚊二三沽，收到"
        ]
      },
      {
        "order_no": "78239778",
        "confidence_score": 0.05,
        "reasoning": "No mention of this stock in conversation...",
        "matched_conversation_segments": []
      }
    ],
    "total_confidence_summary": {
      "average_confidence": 0.185,
      "high_confidence_count": 1,
      "medium_confidence_count": 0,
      "low_confidence_count": 9
    },
    "conversation_summary": "The conversation discusses selling 安東油田服務...",
    "overall_assessment": "Only 1 out of 10 trades matches..."
  },
  "trade_records": [...]
}
```

---

## 🎯 Understanding the Results

### Confidence Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| **0.9-1.0** | ✅ Definitely discussed | Authorized - no action needed |
| **0.7-0.9** | ✅ Likely discussed | Review - probably OK |
| **0.4-0.7** | ⚠️ Possibly discussed | Review - needs clarification |
| **0.1-0.4** | ❌ Probably NOT discussed | Flag - investigate |
| **0.0-0.1** | ❌ Definitely NOT discussed | Alert - likely unauthorized |

### What to Do Next

#### If you see HIGH confidence scores (0.7+)
✅ Good! These trades were properly authorized in the conversation.

#### If you see MEDIUM confidence scores (0.4-0.7)
⚠️ Review these manually. Could be:
- Ambiguous conversation
- Speech recognition errors
- Implied rather than explicit authorization

#### If you see LOW confidence scores (<0.4)
❌ **Action required!** These trades were NOT discussed. Investigate:
- Were they authorized in a different call?
- Standing orders or pre-authorization?
- Potential unauthorized trading?
- Data errors (wrong client ID, wrong date)?

---

## 💡 Tips for Best Results

### 1. **Use Complete Conversations**
More dialogue = better analysis
```
❌ BAD:  "買股票"
✅ GOOD: "我想買騰訊，三千股，四百蚊"
```

### 2. **Include Broker Confirmations**
Confirmations boost confidence
```
Client: "買三千股騰訊"
Broker: "收到，三千股騰訊" ← Important!
```

### 3. **Accurate Timestamps**
Ensure `hkt_datetime` is correct
```
❌ Wrong date → No records found
✅ Correct date → All records loaded
```

### 4. **Use Client ID Filter**
More focused = faster analysis
```
Without filter: Analyzes 100+ trades (all clients)
With filter:    Analyzes 10 trades (one client)
```

### 5. **Choose Right Model**
Balance speed vs accuracy
```
qwen2.5:7b  → Fast (5s)  but less accurate
qwen2.5:14b → Medium (15s) and accurate ✅ Recommended
qwen2.5:32b → Slow (45s)  but most accurate
```

---

## 🔧 Troubleshooting

### Problem: "No records found"
**Solutions:**
- ✅ Check `hkt_datetime` format: `2025-10-09T09:30:00`
- ✅ Verify date exists in `trades.csv`
- ✅ Try removing client_id filter
- ✅ Check trades.csv has data for that date

### Problem: "Cannot parse JSON"
**Solutions:**
- ✅ Validate JSON syntax (use jsonlint.com)
- ✅ Ensure `hkt_datetime` field exists in metadata
- ✅ Check for missing quotes, commas, brackets

### Problem: "All confidence scores are 0"
**Solutions:**
- ✅ Check conversation is not empty
- ✅ Verify LLM is running (`ollama list`)
- ✅ Try different model
- ✅ Check if model supports Chinese text

### Problem: "Structured output failed"
**Solutions:**
- ✅ Update Ollama to latest version
- ✅ Try different model (qwen2.5 series recommended)
- ✅ Check Ollama logs for errors

### Problem: Analysis is too slow
**Solutions:**
- ✅ Use smaller model (qwen2.5:7b)
- ✅ Reduce number of records with client_id filter
- ✅ Ensure Ollama is using GPU (if available)

---

## 📚 Additional Resources

- **Full Documentation**: `CONVERSATION_RECORD_ANALYSIS_README.md`
- **Workflow Diagram**: `CONVERSATION_RECORD_ANALYSIS_WORKFLOW.md`
- **Feature Summary**: `NEW_TAB_SUMMARY.md`
- **Sample Data**: `sample_conversation_for_record_analysis.json`

---

## ✅ Quick Checklist

- [ ] Ollama is running
- [ ] Model is downloaded (qwen2.5:14b)
- [ ] GUI is launched (unified_gui.py)
- [ ] Sample conversation JSON is ready
- [ ] trades.csv exists and has data
- [ ] Client ID M9136 exists in trades.csv for 2025-10-09
- [ ] Ready to test!

---

## 🎉 Success Criteria

After running the analysis, you should have:
- ✅ Confidence scores for all trade records
- ✅ Detailed reasoning for each score
- ✅ Conversation segments cited as evidence
- ✅ Summary statistics (average, high/medium/low counts)
- ✅ Overall assessment of match quality
- ✅ Actionable insights (which trades to review)

---

## 🆘 Need Help?

1. Check error message in output textbox
2. Review troubleshooting section above
3. Check Ollama logs: `ollama logs`
4. Verify JSON format: paste into jsonlint.com
5. Check file paths are correct
6. Ensure Chinese text encoding is UTF-8

---

**Ready to start? Follow Step 1 above! 🚀**

