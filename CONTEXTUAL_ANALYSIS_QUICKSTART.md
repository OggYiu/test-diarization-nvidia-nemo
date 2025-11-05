# 🚀 Contextual Analysis - Quick Start Guide

## What is Contextual Analysis?

Contextual Analysis allows the AI to remember what was discussed in previous conversations when analyzing later ones. This is crucial for understanding abbreviated references and implicit mentions.

**Example:**
- Conversation 1: "我想買騰信窩輪，個八五三八號" (I want to buy Tencent warrant, number 18538)
- Conversation 2: "嗰隻窩輪買咗未？" (Has that warrant been bought yet?)

With contextual analysis **enabled**, the AI understands "嗰隻窩輪" (that warrant) refers to the Tencent warrant from Conversation 1. ✅

Without it, the AI might report "窩輪" as unclear or ambiguous. ❌

## How to Use

### Step 1: Prepare Your JSON Input

Create a JSON array with your conversations **in chronological order**:

```json
[
  {
    "conversation_number": 1,
    "filename": "call_001.wav",
    "metadata": {
      "broker_name": "Your Broker",
      "client_name": "Your Client",
      "hkt_datetime": "2025-01-15 10:00:00"
    },
    "transcriptions": {
      "sensevoice": "First conversation transcript..."
    }
  },
  {
    "conversation_number": 2,
    "filename": "call_002.wav",
    "metadata": {
      "broker_name": "Your Broker",
      "client_name": "Your Client",
      "hkt_datetime": "2025-01-15 14:00:00"
    },
    "transcriptions": {
      "sensevoice": "Second conversation transcript..."
    }
  }
]
```

**💡 Tip**: See `example_contextual_analysis.json` for a complete example.

### Step 2: Open the JSON Batch Analysis Tab

1. Launch the unified GUI: `python unified_gui.py`
2. Navigate to the **"🔟 JSON Batch Analysis"** tab

### Step 3: Configure Settings

#### Select Your LLM(s)
- Choose one or more LLMs from the checkbox list
- Multiple LLMs will each analyze every conversation

#### Advanced Settings

**🔗 Enable Contextual Analysis** (Recommended: ✅ Checked)
- Check this to pass context from previous conversations
- Uncheck for independent analysis of each conversation

**🔧 Enable Vector Store Correction** (Recommended: ✅ Checked)
- Corrects STT errors using the stock database
- Works together with contextual analysis

**Temperature** (Recommended: 0.1)
- Lower values (0.0-0.2) = more deterministic
- Higher values (0.5-1.0) = more creative

### Step 4: Paste Your JSON

Copy your JSON array and paste it into the **"JSON Conversations"** text box.

### Step 5: Click "🚀 Analyze All Conversations"

The system will:
1. Parse your JSON
2. Process Conversation 1 (no context)
3. Extract summary and stocks from Conversation 1
4. Process Conversation 2 (with context from Conversation 1)
5. Continue for all conversations...

### Step 6: Review Results

#### Analysis Results Box
- Shows detailed analysis for each conversation
- Displays which LLM was used
- Shows extracted stocks with confidence levels
- Indicates when context is being used: `🔗 Using context from N previous conversation(s)`

#### Combined JSON Output
- Contains all results in structured JSON format
- Can be saved and used for further processing
- Includes all metadata, timestamps, and extracted stocks

## Understanding the Output

### With Contextual Analysis Enabled

```
📞 CONVERSATION #2 / 3
=================================
🔗 Using context from 1 previous conversation(s)

🤖 Analyzing with LLM 1/1: qwen2.5:14b

┌─ RESULTS
│  Stocks Extracted: 1
│  
│  Stock 1:
│  - Stock Number: 18538
│  - Stock Name: 騰訊窩輪
│  - Confidence: high
│  - Relevance Score: 1.0 (actively discussed)
│  - Reasoning: Referenced from previous conversation as "嗰隻窩輪"
└──────────────────────────────────────
```

Notice:
- "🔗 Using context from 1 previous conversation(s)" appears
- The reasoning mentions "Referenced from previous conversation"
- The stock is correctly identified even though only "窩輪" was mentioned

### Without Contextual Analysis

```
📞 CONVERSATION #2 / 3
=================================

🤖 Analyzing with LLM 1/1: qwen2.5:14b

┌─ RESULTS
│  Stocks Extracted: 0 or 1
│  
│  Stock 1:
│  - Stock Number: Unknown
│  - Stock Name: 窩輪
│  - Confidence: low
│  - Relevance Score: 0.5 (mentioned briefly)
│  - Reasoning: Warrant mentioned but unclear which specific stock
└──────────────────────────────────────
```

Notice:
- No context indicator
- Stock identification is unclear or incomplete
- Lower confidence and relevance scores

## Best Practices

### ✅ DO:
1. **Keep conversations in chronological order** - The context flows from earlier to later conversations
2. **Use related conversations** - Works best when conversations reference each other
3. **Enable both features** - Contextual Analysis + Vector Store Correction work well together
4. **Review the first conversation carefully** - It sets the context for everything that follows

### ❌ DON'T:
1. **Mix unrelated conversations** - If conversations are independent, disable contextual analysis
2. **Reorder conversations randomly** - Context only flows forward in time
3. **Use extremely long batches** - Very large batches (30+ conversations) may hit token limits
4. **Ignore conversation_number** - Keep them sequential for clarity

## Troubleshooting

### Problem: Context not being applied

**Check:**
- ✅ Contextual Analysis checkbox is checked
- ✅ Conversations are in chronological order
- ✅ Previous conversation has valid stocks extracted

### Problem: Incorrect references

**Try:**
- Ensure the first conversation clearly mentions the full stock name
- Check if vector correction is enabled
- Verify transcription quality

### Problem: Slow processing

**Solutions:**
- Reduce number of conversations per batch
- Use fewer LLMs (e.g., just one instead of three)
- For independent conversations, disable contextual analysis

## Example Use Cases

### Use Case 1: Follow-up Calls
**Scenario**: Client calls in the morning to place an order, then calls back in the afternoon to check status.

**Solution**: Enable contextual analysis so the afternoon call's references to "that stock" or "the order" are correctly understood.

### Use Case 2: Multi-leg Trades
**Scenario**: Multiple conversations about the same warrant or stock over a trading day.

**Solution**: Use contextual analysis to track the full context of all discussions about that security.

### Use Case 3: Clarification Calls
**Scenario**: Initial conversation has details, follow-up is brief with just "yes" or "confirm" responses.

**Solution**: Context helps understand what is being confirmed without repeating all details.

## Testing the Feature

Use the provided example file to test:

```bash
# In your browser, navigate to the JSON Batch Analysis tab
# Click "Choose File" or paste the contents of:
example_contextual_analysis.json
```

Compare results with and without contextual analysis enabled to see the difference!

## Advanced Usage

### Custom System Messages with Context

Your custom system message is preserved. The context is appended automatically:

```
[Your Custom System Message]

**===== CONTEXT FROM PREVIOUS CONVERSATIONS =====**
[Automatically generated context]
**===== END OF PREVIOUS CONTEXT =====**
```

### Multiple LLMs with Context

When using multiple LLMs, each one receives the same context. This ensures consistent interpretation across different models.

## Need Help?

- **Full Documentation**: See `CONTEXTUAL_ANALYSIS_FEATURE.md`
- **Example File**: Check `example_contextual_analysis.json`
- **Implementation**: Review `tabs/tab_json_batch_analysis.py`

---

**Happy Analyzing! 🎉**

