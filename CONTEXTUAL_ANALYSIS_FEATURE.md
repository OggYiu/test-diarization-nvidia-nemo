# 🔗 Contextual Analysis Feature for JSON Batch Processing

## Overview

The **Contextual Analysis** feature enhances the JSON Batch Analysis tool by passing analysis results from previous conversations to subsequent ones. This helps the AI understand references, abbreviations, and context that span multiple conversations.

## Problem Statement

In sequential conversations, speakers often refer back to topics discussed earlier without repeating full details:

**Example:**
- **Conversation 1**: Broker and client discuss "騰信窩輪" (Tencent warrant) in detail
- **Conversation 2**: They only mention "窩輪" (warrant) - but it's referring to "騰信窩輪" from Conversation 1

Without contextual awareness, the analysis of Conversation 2 might miss that "窩輪" specifically refers to Tencent, leading to incomplete or ambiguous results.

## Solution

The Contextual Analysis feature:
1. **Analyzes conversations sequentially** in the order they appear in the JSON array
2. **Captures key information** from each conversation (summary + stocks discussed)
3. **Passes this context** to the LLM when analyzing subsequent conversations
4. **Enables better understanding** of abbreviated references and implicit mentions

## How It Works

### 1. Sequential Processing with Context Accumulation

```
Conversation 1: Analyze normally
  ↓
  Extract: Summary + Stocks
  ↓
Conversation 2: Analyze with context from Conversation 1
  ↓
  Extract: Summary + Stocks
  ↓
Conversation 3: Analyze with context from Conversations 1 & 2
  ↓
  ... and so on
```

### 2. Context Format

For each subsequent conversation, the system message is augmented with:

```
**===== CONTEXT FROM PREVIOUS CONVERSATIONS =====**
The following are summaries of previous conversations in this session.
Use this information to understand references and context in the current conversation.

--- Previous Conversation #1 ---
Summary: [AI-generated summary of the conversation]
Stocks discussed:
  - 騰訊 (00700)
  - 小米 (01810)

--- Previous Conversation #2 ---
Summary: [AI-generated summary of the conversation]
Stocks discussed:
  - 騰訊窩輪 (18138)

**===== END OF PREVIOUS CONTEXT =====**

Now analyze the CURRENT conversation below. When you see abbreviated references
(like '窩輪' without a specific stock name), check if they might be referring to
stocks mentioned in the previous conversations above.
```

### 3. LLM Analysis

The LLM receives:
- The original system message (with extraction instructions)
- Context from all previous conversations
- The current conversation transcript

This allows the LLM to:
- Recognize abbreviated references
- Understand implicit mentions
- Maintain context across the entire session

## Usage

### Enabling/Disabling the Feature

In the JSON Batch Analysis tab UI:

1. Look for **Advanced Settings** section
2. Find the checkbox: **🔗 Enable Contextual Analysis**
3. **Checked (default)**: Context from previous conversations will be passed to subsequent analyses
4. **Unchecked**: Each conversation will be analyzed independently (previous behavior)

### When to Use Contextual Analysis

**✅ USE IT WHEN:**
- Conversations are part of a continuous session or call sequence
- Speakers reference earlier topics without full details
- Multiple conversations between the same participants
- Follow-up calls discussing the same stocks/trades

**❌ DISABLE IT WHEN:**
- Conversations are completely independent
- Each conversation involves different participants
- You want isolated analysis without cross-conversation influence
- Testing/debugging individual conversations

## Example Scenario

### JSON Input

```json
[
  {
    "conversation_number": 1,
    "filename": "call_001.wav",
    "metadata": {
      "broker_name": "張經紀",
      "client_name": "李先生",
      "hkt_datetime": "2025-01-15 10:30:00"
    },
    "transcriptions": {
      "sensevoice": "經紀：早晨李生，今日想買啲咩？\n客戶：我想買騰信窩輪，個八五三八號嗰隻。\n經紀：好，騰信窩輪18538，我幫你掛單。"
    }
  },
  {
    "conversation_number": 2,
    "filename": "call_002.wav",
    "metadata": {
      "broker_name": "張經紀",
      "client_name": "李先生",
      "hkt_datetime": "2025-01-15 14:15:00"
    },
    "transcriptions": {
      "sensevoice": "經紀：李生，嗰隻窩輪已經買咗喇。\n客戶：好，幾錢入咗？\n經紀：0.125，幫你買咗10手。"
    }
  }
]
```

### Analysis Results

**Without Contextual Analysis:**
- Conversation 2 might report: "窩輪 (warrant) - unclear which stock" ⚠️

**With Contextual Analysis:**
- Conversation 2 correctly identifies: "騰信窩輪 (18538) - Tencent warrant" ✅
- The LLM understands "嗰隻窩輪" (that warrant) refers to the Tencent warrant from Conversation 1

## Technical Details

### Implementation

1. **Context Storage**: After analyzing each conversation, the system stores:
   - Conversation number
   - AI-generated summary
   - List of all stocks discussed (with corrected names/numbers)

2. **Context Injection**: Before analyzing the next conversation:
   - Build a formatted context string from all previous conversations
   - Append to the system message
   - Pass to the LLM

3. **Context Accumulation**: Context grows with each conversation:
   - Conversation 1: No context
   - Conversation 2: Context from Conversation 1
   - Conversation 3: Context from Conversations 1 & 2
   - etc.

### Performance Considerations

- **Token Usage**: Each conversation's context adds tokens to the prompt. For very long sessions (20+ conversations), monitor token limits.
- **Processing Time**: Slightly increased due to larger prompts, but typically negligible.
- **Memory**: Context is stored in memory during processing and cleared after batch completion.

## Benefits

1. **🎯 Improved Accuracy**: Better understanding of implicit references
2. **📊 Better Context**: Maintains conversation flow across multiple interactions
3. **🔍 Fewer Ambiguities**: Resolves abbreviated mentions using prior context
4. **💼 Real-world Usage**: Matches how actual conversations work (with callbacks and follow-ups)

## Configuration

The feature works automatically when enabled. No additional configuration needed.

### System Message Compatibility

The contextual information is appended to your custom system message, so:
- ✅ Your custom instructions are preserved
- ✅ Context is added after your instructions
- ✅ Compatible with all LLM models
- ✅ Works with vector store correction

## Limitations

1. **Sequential Dependency**: Conversations must be in chronological order for best results
2. **Token Limits**: Very long sessions (30+ conversations) might approach model token limits
3. **Context Quality**: Depends on the quality of summaries generated for each conversation
4. **No Backward Context**: Later conversations don't affect earlier ones (processing is one-way)

## Future Enhancements

Potential improvements for future versions:
- **Configurable Context Depth**: Limit context to last N conversations instead of all
- **Context Summarization**: Compress older conversation context to save tokens
- **Relationship Tracking**: Track which stocks/topics carry over between conversations
- **Context Preview**: Show users what context is being passed to each analysis

## Troubleshooting

### Context Not Being Applied

**Check:**
1. ✅ "Enable Contextual Analysis" checkbox is checked
2. ✅ Conversations are in correct chronological order in JSON
3. ✅ Previous conversations have valid summaries and stocks

### Incorrect References

**Try:**
1. Review the summary quality of previous conversations
2. Ensure stock names are correctly extracted in earlier conversations
3. Check if vector store correction is enabled for better stock identification

### Performance Issues

**Solutions:**
1. Reduce the number of conversations per batch
2. Use fewer LLMs (to reduce total processing time)
3. Consider disabling contextual analysis for very large batches

## Conclusion

The Contextual Analysis feature significantly improves the accuracy and usefulness of batch conversation analysis by maintaining context across sequential conversations. This mirrors real-world conversation patterns where speakers reference earlier topics, making the analysis more practical and reliable for production use.

For questions or issues, refer to the main documentation or check the implementation in `tabs/tab_json_batch_analysis.py`.

