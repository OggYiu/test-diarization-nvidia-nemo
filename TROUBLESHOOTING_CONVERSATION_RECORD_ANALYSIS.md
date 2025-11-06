# Troubleshooting - Conversation Record Analysis

## Common Errors and Solutions

### ❌ Error: "Cannot find date in conversation JSON"

This error means the tool cannot find a valid `hkt_datetime` field in your JSON.

#### Required Structure

Your JSON **MUST** have:

```json
{
  "metadata": {
    "hkt_datetime": "2025-10-20T10:01:20"
  },
  "conversations": [
    [
      {"speaker": "Broker", "text": "..."},
      {"speaker": "Client", "text": "..."}
    ]
  ]
}
```

#### Common Issues

**Issue 1: Missing `hkt_datetime` field**
```json
❌ BAD:
{
  "metadata": {
    "filename": "test.wav"
    // Missing hkt_datetime!
  }
}

✅ GOOD:
{
  "metadata": {
    "filename": "test.wav",
    "hkt_datetime": "2025-10-20T10:01:20"
  }
}
```

**Issue 2: `hkt_datetime` is empty or null**
```json
❌ BAD:
{
  "metadata": {
    "hkt_datetime": ""  // Empty!
  }
}

❌ BAD:
{
  "metadata": {
    "hkt_datetime": null  // Null!
  }
}

❌ BAD:
{
  "metadata": {
    "hkt_datetime": "N/A"  // N/A is treated as invalid
  }
}

✅ GOOD:
{
  "metadata": {
    "hkt_datetime": "2025-10-20T10:01:20"
  }
}
```

**Issue 3: Wrong datetime format**
```json
❌ BAD:
{
  "metadata": {
    "hkt_datetime": "20/10/2025 10:01"  // Wrong format
  }
}

✅ GOOD - Supported formats:
{
  "metadata": {
    "hkt_datetime": "2025-10-20T10:01:20"  // ISO format (recommended)
  }
}

OR:
{
  "metadata": {
    "hkt_datetime": "2025-10-20 10:01:20"  // Also works
  }
}

OR:
{
  "metadata": {
    "hkt_datetime": "2025-10-20"  // Date only (also works)
  }
}
```

**Issue 4: `metadata` field is missing**
```json
❌ BAD:
{
  "filename": "test.wav",
  "hkt_datetime": "2025-10-20T10:01:20"  // hkt_datetime at wrong level!
}

✅ GOOD:
{
  "filename": "test.wav",
  "metadata": {
    "hkt_datetime": "2025-10-20T10:01:20"  // Inside metadata!
  }
}
```

#### Debugging Steps

1. **Check your JSON has `metadata` field**
   ```json
   {
     "metadata": { ... }
   }
   ```

2. **Check `hkt_datetime` is inside `metadata`**
   ```json
   {
     "metadata": {
       "hkt_datetime": "..."
     }
   }
   ```

3. **Check `hkt_datetime` value is not empty**
   - Not `""`
   - Not `null`
   - Not `"N/A"`

4. **Check datetime format**
   - Should be: `"2025-10-20T10:01:20"`
   - Or: `"2025-10-20 10:01:20"`
   - Or: `"2025-10-20"`

5. **Validate your JSON**
   - Use https://jsonlint.com to check for syntax errors
   - Make sure all quotes are correct
   - Make sure all commas are in the right place

---

### ❌ Error: "Conversation appears to be empty or too short"

This error means your JSON doesn't have conversation content.

#### Required: Add Conversations

Your JSON needs **EITHER**:

**Option 1: Conversations array**
```json
{
  "metadata": { ... },
  "conversations": [
    [
      {"speaker": "Broker", "text": "喂，你好！"},
      {"speaker": "Client", "text": "我想買股票"}
    ]
  ]
}
```

**Option 2: Transcriptions array**
```json
{
  "metadata": { ... },
  "transcriptions": [
    {
      "model": "wsyue-asr",
      "text": "喂你好我想買股票...",
      "confidence": 0.92
    }
  ]
}
```

**Option 3: Both (recommended)**
```json
{
  "metadata": { ... },
  "conversations": [ ... ],
  "transcriptions": [ ... ]
}
```

---

### ❌ Error: "No trade records found"

This error means no trades exist in `trades.csv` for that date/client.

#### Possible Causes

**Cause 1: Wrong date**
```
Your JSON: hkt_datetime = "2025-10-20T10:01:20"
trades.csv: Has data for 2025-10-09
Result: No match!

Solution: Check the correct date in trades.csv
```

**Cause 2: Wrong client ID**
```
Your JSON: client_id = "P77197"
trades.csv: Has no records for P77197 on that date
Result: No match!

Solution: 
- Remove client_id filter to see all clients
- Or check the correct client_id in trades.csv
```

**Cause 3: trades.csv doesn't have data**
```
Solution: Check that trades.csv exists and has data
```

#### How to Debug

1. **Check what date you're using**
   - Look at your `hkt_datetime` value
   - The tool extracts the DATE part (e.g., 2025-10-20)

2. **Check trades.csv for that date**
   - Open trades.csv
   - Search for that date in the OrderTime column

3. **Try without client_id filter**
   - Leave "Client ID Filter" textbox empty
   - This will show ALL trades for that date

4. **Check the date format in trades.csv**
   - Should be like: `2025-10-09 09:30:52.050`

---

### ❌ Error: "Cannot parse datetime"

This error means the `hkt_datetime` value cannot be parsed.

#### Supported Formats

```
✅ "2025-10-20T10:01:20"         (ISO format - recommended)
✅ "2025-10-20 10:01:20.123"     (trades.csv format)
✅ "2025-10-20 10:01:20"         (without milliseconds)
✅ "2025-10-20"                  (date only)

❌ "20/10/2025"                  (wrong format)
❌ "2025/10/20"                  (wrong separators)
❌ "10:01:20 2025-10-20"         (wrong order)
```

#### Solution

Change your datetime to one of the supported formats above.

---

## Complete Working Example

Here's a **complete, working JSON** you can copy and modify:

```json
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
  "conversations": [
    [
      {
        "speaker": "Broker",
        "text": "喂，你好！"
      },
      {
        "speaker": "Client",
        "text": "我想買騰訊，三千股。"
      },
      {
        "speaker": "Broker",
        "text": "好，三千股騰訊，幾多錢？"
      },
      {
        "speaker": "Client",
        "text": "四百蚊。"
      },
      {
        "speaker": "Broker",
        "text": "收到，三千股騰訊四百蚊，確認。"
      }
    ]
  ],
  "transcriptions": [
    {
      "model": "wsyue-asr",
      "text": "喂你好我想買騰訊三千股。好三千股騰訊幾多錢？四百蚊。收到三千股騰訊四百蚊確認。",
      "confidence": 0.92
    }
  ]
}
```

### How to Use This Example

1. Copy the above JSON
2. Modify these fields:
   - `metadata.hkt_datetime` → Your call date/time
   - `metadata.client_id` → Your client ID
   - `conversations` → Your actual conversation
3. Paste into the tool
4. Click "🎯 Analyze Records"

---

## Quick Checklist

Before clicking "Analyze Records", verify:

- [ ] JSON is valid (no syntax errors)
- [ ] `metadata` field exists
- [ ] `metadata.hkt_datetime` exists and is not empty
- [ ] `hkt_datetime` format is correct (e.g., "2025-10-20T10:01:20")
- [ ] `conversations` OR `transcriptions` field exists
- [ ] Conversations have actual text content
- [ ] `trades.csv` file exists
- [ ] `trades.csv` has data for that date
- [ ] (Optional) `client_id` matches a client in trades.csv

---

## Still Having Issues?

### Enable Debug Mode

The improved error messages now show:
- What fields were found in your JSON
- What values they contain
- What's missing

Read the error message carefully - it will tell you exactly what's wrong!

### Test with Sample File

Use the provided sample file first:
```
sample_conversation_for_record_analysis.json
```

This is a known-good example. If this works but your JSON doesn't, compare them to find the difference.

### Common JSON Syntax Errors

```json
❌ Missing comma:
{
  "metadata": { "hkt_datetime": "2025-10-20T10:01:20" }
  "conversations": []  // ERROR: Missing comma above!
}

✅ With comma:
{
  "metadata": { "hkt_datetime": "2025-10-20T10:01:20" },
  "conversations": []
}
```

```json
❌ Trailing comma:
{
  "metadata": { "hkt_datetime": "2025-10-20T10:01:20" },
  "conversations": [],  // ERROR: Trailing comma!
}

✅ No trailing comma:
{
  "metadata": { "hkt_datetime": "2025-10-20T10:01:20" },
  "conversations": []
}
```

```json
❌ Wrong quotes:
{
  'metadata': { 'hkt_datetime': '2025-10-20T10:01:20' }  // ERROR: Single quotes!
}

✅ Correct quotes:
{
  "metadata": { "hkt_datetime": "2025-10-20T10:01:20" }
}
```

---

## Need More Help?

1. Validate your JSON at: https://jsonlint.com
2. Compare with: `sample_conversation_for_record_analysis.json`
3. Check: `conversation_record_analysis_template.json`
4. Read: `CONVERSATION_RECORD_ANALYSIS_README.md`
5. Check error message details - they now show what was found in your JSON!

