# Supported Conversation JSON Formats

## Overview

The Conversation Record Analysis tab now supports **multiple JSON formats** for maximum flexibility.

## Format 1: Single Conversation Object (Original)

```json
{
  "conversation_number": 1,
  "filename": "call_001.wav",
  "metadata": {
    "hkt_datetime": "2025-10-20T10:01:20",
    "client_id": "P77197",
    "broker_id": "0489"
  },
  "transcriptions": [
    {
      "model": "wsyue-asr",
      "text": "Full transcription text..."
    }
  ],
  "conversations": [
    [
      {"speaker": "Broker", "text": "Hello"},
      {"speaker": "Client", "text": "Hi"}
    ]
  ]
}
```

**Use this when**: You have a single conversation to analyze

---

## Format 2: Array of Conversations (NEW!)

```json
[
  {
    "conversation_number": 1,
    "filename": "call_001.wav",
    "metadata": {
      "hkt_datetime": "2025-10-20T10:01:20",
      "client_id": "P77197"
    },
    "transcriptions": {...}
  },
  {
    "conversation_number": 2,
    "filename": "call_002.wav",
    "metadata": {
      "hkt_datetime": "2025-10-20T14:08:41",
      "client_id": "P77197"
    },
    "transcriptions": {...}
  }
]
```

**Use this when**: You have multiple conversations from batch processing

**Note**: The tool will **analyze the first conversation** in the array. To analyze others, paste them individually or extract them from the array.

---

## Format 3: Transcriptions as Dictionary (NEW!)

Your transcriptions can be either:

### Array Format (Original)
```json
"transcriptions": [
  {
    "model": "wsyue-asr",
    "text": "Transcription text here..."
  },
  {
    "model": "sensevoice",
    "text": "Another transcription..."
  }
]
```

### Dictionary Format (NEW!)
```json
"transcriptions": {
  "sensevoice": "Transcription text from sensevoice...",
  "wsyue-asr": "Transcription text from wsyue-asr..."
}
```

**Both formats are supported!**

---

## Complete Example: Your Format

This is the **exact format** from your data, and it's **fully supported**:

```json
[
  {
    "conversation_number": 1,
    "filename": "Dickson Lau 0489_8330-96674941_202510200201201108.wav",
    "metadata": {
      "filename": "Dickson Lau 0489_8330-96674941_202510200201201108.wav",
      "broker_name": "Dickson Lau",
      "broker_id": "0489",
      "client_number": "96674941",
      "client_name": "CHENG SUK HING",
      "client_id": "P77197",
      "utc_datetime": "2025-10-20T02:01:20",
      "hkt_datetime": "2025-10-20T10:01:20",
      "timestamp": "2025-11-05 18:21:19"
    },
    "transcriptions": {
      "sensevoice": "經紀 Dickson Lau (start_time: 0.62): 請到時點啊。\n客戶 CHENG SUK HING (start_time: 1.995): 劉生啊，我想買騰訊個窩輪啊..."
    }
  },
  {
    "conversation_number": 2,
    "filename": "Dickson Lau_8330-96674941_202510200608412868.wav",
    "metadata": {
      "filename": "Dickson Lau_8330-96674941_202510200608412868.wav",
      "broker_name": "Dickson Lau",
      "broker_id": "N/A",
      "client_number": "96674941",
      "client_name": "CHENG SUK HING",
      "client_id": "P77197",
      "utc_datetime": "2025-10-20T06:08:41",
      "hkt_datetime": "2025-10-20T14:08:41",
      "timestamp": "2025-11-05 18:21:57"
    },
    "transcriptions": {
      "sensevoice": "經紀 Dickson Lau (start_time: 0.0): \n客戶 CHENG SUK HING (start_time: 1.26): 阿劉生。..."
    }
  }
]
```

### What Happens

1. ✅ Tool detects this is an **array** with 2 conversations
2. ✅ Tool extracts the **first conversation** (conversation_number: 1)
3. ✅ Tool shows a note: "Analyzing conversation #1: Dickson Lau 0489_8330-96674941_202510200201201108.wav"
4. ✅ Tool extracts `hkt_datetime: "2025-10-20T10:01:20"`
5. ✅ Tool loads all trades for **2025-10-20** for client **P77197**
6. ✅ Tool analyzes each trade against the conversation

### To Analyze Conversation #2

Simply paste just the second object:
```json
{
  "conversation_number": 2,
  "filename": "Dickson Lau_8330-96674941_202510200608412868.wav",
  "metadata": {
    "hkt_datetime": "2025-10-20T14:08:41",
    "client_id": "P77197",
    ...
  },
  "transcriptions": {...}
}
```

---

## Required Fields

### Absolutely Required

1. **`metadata.hkt_datetime`** - Must be present and valid
   - Format: `"2025-10-20T10:01:20"` or `"2025-10-20 10:01:20"` or `"2025-10-20"`
   - Cannot be empty, null, or "N/A"

2. **`transcriptions` OR `conversations`** - At least one must have content
   - Can be array or dictionary format
   - Must contain actual text (not empty)

### Optional but Recommended

- `metadata.client_id` - Filters trades to specific client
- `metadata.broker_id` - Included in output for reference
- `conversation_number` - Shows which conversation is being analyzed
- `filename` - Shows in output for tracking

---

## Field Mapping

| Your Field | Used For | Notes |
|------------|----------|-------|
| `metadata.hkt_datetime` | ✅ Date extraction | **Required** |
| `metadata.client_id` | ✅ Trade filtering | Optional (can filter manually) |
| `metadata.broker_id` | ℹ️ Display only | Not used for matching |
| `metadata.client_name` | ℹ️ Display only | Not used for matching |
| `transcriptions` | ✅ LLM analysis | Required (or conversations) |
| `conversations` | ✅ LLM analysis | Required (or transcriptions) |
| `conversation_number` | ℹ️ Display only | Helps identify which one |
| `filename` | ℹ️ Display only | Helps identify which one |

---

## Error Handling

### If You See: "Cannot find valid date"

**Check**:
1. Is `metadata` present? ✓
2. Is `metadata.hkt_datetime` present? ✓
3. Is the value not empty/null/"N/A"? ✓
4. Is the format correct? ✓

**Debug Output**: The error message now shows what was found in your JSON!

### If You See: "Conversation appears empty"

**Check**:
1. Is `transcriptions` OR `conversations` present? ✓
2. Does it have actual text content? ✓
3. Is it formatted correctly (dict or array)? ✓

---

## Quick Tips

### Tip 1: Analyzing Multiple Conversations

If you have an array with 5 conversations:

**Option A**: Paste the whole array
- Tool analyzes conversation #1
- Repeat with each individual conversation

**Option B**: Extract individually
```javascript
// In browser console or Python
conversations = [...your array...];
console.log(JSON.stringify(conversations[0]));  // First
console.log(JSON.stringify(conversations[1]));  // Second
// etc.
```

### Tip 2: Batch Processing (Future Feature)

Currently, the tool analyzes **one conversation at a time**. A future update may add:
- Process all conversations in array
- Generate combined report
- Compare confidence across conversations

### Tip 3: Different Dates

If your array has conversations from **different dates**, each one will load trades for its respective date:

- Conversation 1: `hkt_datetime = "2025-10-20T10:01:20"` → Loads trades for 2025-10-20
- Conversation 2: `hkt_datetime = "2025-10-20T14:08:41"` → Loads trades for 2025-10-20 (same day)
- Conversation 3: `hkt_datetime = "2025-10-21T09:00:00"` → Loads trades for 2025-10-21 (different day)

---

## Examples of Valid Inputs

### ✅ Minimal Valid (Single)
```json
{
  "metadata": {
    "hkt_datetime": "2025-10-20T10:01:20",
    "client_id": "P77197"
  },
  "transcriptions": {
    "sensevoice": "客戶說要買股票..."
  }
}
```

### ✅ Minimal Valid (Array)
```json
[
  {
    "metadata": {
      "hkt_datetime": "2025-10-20T10:01:20"
    },
    "transcriptions": {
      "text": "conversation text"
    }
  }
]
```

### ✅ Your Full Format (Array)
```json
[
  {
    "conversation_number": 1,
    "filename": "...",
    "metadata": {
      "hkt_datetime": "2025-10-20T10:01:20",
      "client_id": "P77197",
      ...
    },
    "transcriptions": {
      "sensevoice": "..."
    }
  }
]
```

---

## Summary

✅ **Single conversation object** - Supported  
✅ **Array of conversations** - Supported (analyzes first)  
✅ **Dictionary transcriptions** - Supported  
✅ **Array transcriptions** - Supported  
✅ **Your exact format** - Fully supported!  

**Just paste your JSON and click "Analyze Records"!** 🎯

