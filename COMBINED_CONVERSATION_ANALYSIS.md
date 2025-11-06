# Combined Conversation Analysis Feature

## Overview

The **Combined Conversation Analysis** feature allows you to analyze multiple related conversations as one unified context when evaluating trade records. This is particularly useful when conversations are sequential and related (e.g., multiple calls throughout the same day with the same client).

## Why Use Combined Analysis?

### Problem Scenario

Imagine you have these two conversations on the same day:

**Conversation 1 (10:00 AM):**
```
Client: "I want to buy Tencent, 1000 shares at market price"
Broker: "Ok, I'll execute that for you"
```

**Conversation 2 (2:00 PM):**
```
Client: "How's that stock doing?"
Broker: "It's up 2%, do you want to sell?"
Client: "Yes, sell half"
```

### The Challenge

When analyzing Conversation 2 alone, the LLM doesn't know which stock "that stock" refers to. This could lead to:
- ❌ Low confidence scores for valid trades
- ❌ Inability to match trades to conversations
- ❌ False negatives in compliance checking

### The Solution

With **Combined Analysis enabled**, both conversations are analyzed together as one unified context. Now the LLM can:
- ✅ Understand "that stock" refers to Tencent from Conversation 1
- ✅ Correctly match the sell order in Conversation 2 to the context from Conversation 1
- ✅ Provide accurate confidence scores across all related conversations

## How to Use

### Step 1: Prepare Your JSON

Your conversation JSON should be an **array** of conversation objects:

```json
[
  {
    "metadata": {
      "hkt_datetime": "2025-10-20T10:00:00",
      "filename": "call_001.wav",
      "client_id": "P77197"
    },
    "transcriptions": {...},
    "conversations": [...]
  },
  {
    "metadata": {
      "hkt_datetime": "2025-10-20T14:00:00",
      "filename": "call_002.wav",
      "client_id": "P77197"
    },
    "transcriptions": {...},
    "conversations": [...]
  }
]
```

### Step 2: Enable Combined Analysis

1. Open the **Conversation Record Analysis** tab
2. Paste your conversation JSON array
3. ✅ **Check the "Enable Combined Analysis" checkbox**
4. Configure other settings (trades file, model, etc.)
5. Click "🎯 Analyze Records"

### Step 3: Review Results

The analysis results will show:

```
✨ COMBINED ANALYSIS MODE ENABLED
   Analyzing 2 conversations as one unified context.
   Context from earlier conversations helps understand later ones.
   
   Conversations analyzed:
   1. call_001.wav
   2. call_002.wav
```

## When to Use Combined Analysis

### ✅ Recommended Use Cases

- **Same Client, Same Day**: Multiple calls with the same client on the same trading day
- **Follow-up Conversations**: Later calls referring back to earlier discussions
- **Order Modifications**: Initial order followed by changes/cancellations
- **Related Trading Sessions**: Multiple discussions about the same stocks/positions
- **Clarification Calls**: Client calls back to clarify or confirm earlier orders

### ❌ When NOT to Use

- **Unrelated Conversations**: Different clients or completely separate topics
- **Different Days**: Conversations from different trading days (unless intentionally tracking multi-day strategies)
- **Single Conversation**: When you only have one conversation (checkbox has no effect)
- **Testing Individual Calls**: When you specifically want to test each conversation's standalone clarity

## Technical Details

### How It Works

1. **Date Extraction**: Uses the first conversation's `hkt_datetime` to determine the date
2. **Trade Loading**: Loads all trades for that date (and client, if specified)
3. **Context Merging**: Combines all conversations into one unified text with clear labels
4. **Unified Analysis**: LLM analyzes trades against the complete combined context
5. **Cross-Reference**: LLM can reference any conversation to match trades

### Input Format

The feature accepts two input formats:

**Format 1: Array (recommended for multiple conversations)**
```json
[
  {"metadata": {...}, "transcriptions": {...}, "conversations": [...]},
  {"metadata": {...}, "transcriptions": {...}, "conversations": [...]}
]
```

**Format 2: Single Object (traditional)**
```json
{"metadata": {...}, "transcriptions": {...}, "conversations": [...]}
```

*Note: Combined Analysis checkbox has no effect on single objects*

### Combined Text Format

When combined analysis is enabled, conversations are formatted as:

```
================================================================================
📞 COMBINED ANALYSIS OF 2 CONVERSATIONS
================================================================================
Note: This analysis treats all 2 conversations as one unified context.
Context from earlier conversations may help understand later ones.

================================================================================
📞 Conversation #1 (call_001.wav)
================================================================================
=== CALL METADATA ===
...

================================================================================
📞 Conversation #2 (call_002.wav)
================================================================================
=== CALL METADATA ===
...
```

## Output Information

### Text Output

The formatted text output includes:
- 🎯 Indication that combined analysis mode is enabled
- 📋 List of all conversations analyzed
- 📊 Unified confidence scores considering all conversations
- 💬 Overall assessment across all conversations

### JSON Output

The JSON output includes:

```json
{
  "status": "success",
  "analysis_info": {
    "combined_analysis_mode": true,
    "conversations_analyzed": 2,
    ...
  },
  "conversations_info": [
    {
      "index": 1,
      "filename": "call_001.wav",
      "hkt_datetime": "2025-10-20T10:00:00"
    },
    {
      "index": 2,
      "filename": "call_002.wav",
      "hkt_datetime": "2025-10-20T14:00:00"
    }
  ],
  "analysis_result": {...}
}
```

## Examples

### Example 1: Stock Name Carryover

**Input**: 2 conversations, combined analysis enabled

**Conversation 1**: "Buy 騰訊 (Tencent) 500 shares"  
**Conversation 2**: "Sell that stock, 200 shares"

**Result**: ✅ LLM correctly identifies "that stock" as Tencent from Conversation 1

### Example 2: Price Discussion Continuation

**Input**: 2 conversations, combined analysis enabled

**Conversation 1**: "I'm interested in 0700 at $320"  
**Conversation 2**: "Ok, execute at that price"

**Result**: ✅ LLM knows "that price" = $320 from Conversation 1

### Example 3: Quantity Modifications

**Input**: 2 conversations, combined analysis enabled

**Conversation 1**: "Buy 1000 shares of HSBC"  
**Conversation 2**: "Actually, make it 800 instead"

**Result**: ✅ LLM understands the modification reduces original order from 1000 to 800

## Tips for Best Results

1. **Chronological Order**: Ensure conversations are ordered chronologically in the array
2. **Same Client**: Best results when all conversations involve the same client
3. **Same Day**: Most effective for conversations on the same trading day
4. **Clear Metadata**: Include `filename` or other identifiers to track which conversation is which
5. **Quality Transcriptions**: Better transcription quality = better cross-conversation understanding

## Comparison: Individual vs Combined

| Scenario | Individual Analysis | Combined Analysis |
|----------|-------------------|-------------------|
| Standalone conversations | ✅ Perfect | ⚠️ Unnecessary overhead |
| Related conversations | ❌ May miss context | ✅ Full context understanding |
| Cross-references | ❌ Cannot resolve | ✅ Resolves correctly |
| Processing time | ⚡ Fast | 🕐 Slightly slower |
| Confidence accuracy | ⚠️ May be lower | ✅ More accurate |
| Use case | Testing individual calls | Production compliance |

## Troubleshooting

### Problem: Checkbox does nothing

**Solution**: Ensure your input is an **array** of conversations, not a single object.

### Problem: Only analyzing first conversation

**Solution**: Verify the checkbox is ✅ **checked** before clicking "Analyze Records"

### Problem: Date extraction error

**Solution**: Ensure the **first conversation** in the array has valid `hkt_datetime` in metadata

### Problem: Poor cross-references

**Solution**: 
- Check transcription quality
- Ensure conversations are chronologically ordered
- Verify they're actually related (same client/topic)

## Future Enhancements

Potential future improvements:
- [ ] Analyze each conversation individually AND combined (comparison view)
- [ ] Automatic detection of related conversations
- [ ] Smart chunking for very large conversation arrays
- [ ] Timeline view showing trades matched to specific conversations
- [ ] Client session grouping suggestions

## Related Documentation

- `CONVERSATION_RECORD_ANALYSIS_README.md` - Main feature documentation
- `CONVERSATION_JSON_FORMATS.md` - JSON format specifications
- `QUICKSTART_CONVERSATION_RECORD_ANALYSIS.md` - Quick start guide

---

**Version**: 1.0  
**Last Updated**: 2025-11-06  
**Feature Status**: ✅ Production Ready

