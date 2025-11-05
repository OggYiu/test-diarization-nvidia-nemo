# JSON Batch Analysis - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Prepare Your JSON

Create a JSON file with your conversations. Here's a minimal example:

```json
[
  {
    "conversation_number": 1,
    "filename": "call1.wav",
    "metadata": {
      "broker_name": "John Doe",
      "client_name": "Jane Smith"
    },
    "transcriptions": {
      "sensevoice": "經紀: 你好，請問需要什麼幫助？\n客戶: 我想買騰訊"
    }
  }
]
```

### Step 2: Open the Tab

1. Launch the Unified GUI
2. Navigate to **"🔟 JSON Batch Analysis"** tab

### Step 3: Run Analysis

1. **Paste JSON** into the "JSON Conversations" textbox
2. **Select LLM** (default is fine for testing)
3. **Click** "🚀 Analyze All Conversations"
4. **Wait** for results to appear
5. **Copy** the combined JSON output if needed

That's it! 🎉

---

## 📋 Complete Example

Use the provided `example_json_batch.json`:

**Content:**
```json
[
  {
    "conversation_number": 1,
    "filename": "Dickson Lau 0489_8330-96674941_202510200201201108.wav",
    "metadata": {
      "broker_name": "Dickson Lau",
      "broker_id": "0489",
      "client_name": "CHENG SUK HING"
    },
    "transcriptions": {
      "sensevoice": "經紀 Dickson Lau: 請到時點啊。\n客戶 CHENG SUK HING: 劉生啊，我想買騰訊個輪啊買個聲得唔得啊嗯。"
    }
  },
  {
    "conversation_number": 2,
    "filename": "Dickson Lau_8330-96674941_202510200608412868.wav",
    "metadata": {
      "broker_name": "Dickson Lau",
      "client_name": "CHENG SUK HING"
    },
    "transcriptions": {
      "sensevoice": "經紀 Dickson Lau: \n客戶 CHENG SUK HING: 阿劉生。\n經紀 Dickson Lau: 啊幾好思啊，嗰只系輪買咗一百即三百。"
    }
  }
]
```

**Expected Output:**
- Conversation 1: Should detect "騰訊" (Tencent) with stock code 00700
- Conversation 2: Should detect "輪" (warrant/option) related discussions

---

## ⚙️ Common Settings

### For Quick Testing
- **LLM**: qwen2.5:32b-instruct (fast)
- **Vector Store Correction**: ✅ Enabled
- **Temperature**: 0.1

### For Production
- **LLM**: Select 2-3 models for cross-validation
- **Vector Store Correction**: ✅ Enabled
- **Temperature**: 0.1
- **System Message**: Keep default or customize for your use case

---

## 🎯 What You Get

### 1. Formatted Results Display
```
================================
📞 CONVERSATION #1 / 2
================================
📁 Filename: call1.wav
🎤 Transcription Source: sensevoice
👤 Broker: John Doe
👥 Client: Jane Smith

📝 Transcription:
---
經紀: 你好，請問需要什麼幫助？
客戶: 我想買騰訊
---

🤖 Analyzing with LLM 1/1: qwen2.5:32b-instruct

┌─ RESULTS
│  📊 股票提取結果
│  🤖 LLM 模型: qwen2.5:32b-instruct
│  
│  🔍 找到 1 個股票:
│  
│     1. ✅ 股票 #1
│        • 股票代號: 00700
│        • 股票名稱: 騰訊
│        🔧 修正後:
│           ◦ 股票代號: 00700
│           ◦ 股票名稱: 騰訊控股
│           ◦ 修正信心: 95.00%
│        • 置信度: HIGH
│        • 相關程度: 🟢 2/2
└───────────────────────────────────
```

### 2. Combined JSON Output
```json
[
  {
    "conversation_number": 1,
    "filename": "call1.wav",
    "metadata": {
      "broker_name": "John Doe",
      "client_name": "Jane Smith"
    },
    "transcription_source": "sensevoice",
    "analysis_timestamp": "2025-11-05 12:00:00",
    "llms_used": ["qwen2.5:32b-instruct"],
    "stocks": [
      {
        "stock_number": "00700",
        "stock_name": "騰訊",
        "confidence": "high",
        "relevance_score": 2,
        "corrected_stock_name": "騰訊控股",
        "corrected_stock_number": "00700",
        "correction_confidence": 0.95,
        "reasoning": "Client mentioned buying Tencent",
        "llm_model": "qwen2.5:32b-instruct"
      }
    ]
  }
]
```

---

## 🔧 Troubleshooting

### Problem: "Invalid JSON format"
**Solution:** Validate your JSON using jsonlint.com

### Problem: "No transcription text found"
**Solution:** Ensure the `transcriptions` field is not empty

### Problem: Very slow processing
**Solution:** 
- Use only 1 LLM for testing
- Reduce batch size
- Use a faster model

### Problem: Out of memory
**Solution:**
- Process fewer conversations at once
- Close other GPU applications
- Use a smaller LLM model

---

## 📚 Next Steps

1. ✅ Try the example JSON file
2. ✅ Process your own conversations
3. ✅ Experiment with multiple LLMs
4. ✅ Review the full README for advanced features
5. ✅ Integrate with other tabs in the suite

---

## 💡 Pro Tips

1. **Start small**: Test with 2-3 conversations first
2. **Use vector correction**: It significantly improves accuracy
3. **Multiple LLMs**: Use 2-3 models for important analyses
4. **Save outputs**: Copy the JSON output for further processing
5. **Iterate**: Refine your system message based on results

---

## 🎓 Learning Resources

- **Full README**: `JSON_BATCH_ANALYSIS_README.md`
- **Implementation Details**: `JSON_BATCH_ANALYSIS_IMPLEMENTATION.md`
- **Example File**: `example_json_batch.json`

Happy analyzing! 🚀
