# Conversation Record Analysis

## Overview

The **Conversation Record Analysis** tab provides LLM-powered analysis to determine which trade records from `trades.csv` were actually discussed in a phone conversation.

This is the **inverse** of the Trade Verification tab:
- **Trade Verification**: Takes detected transactions → finds matching records in trades.csv
- **Conversation Record Analysis**: Takes trade records → determines if they were mentioned in the conversation

## Use Cases

1. **Compliance & Audit**: Verify that executed trades were actually authorized in phone calls
2. **Fraud Detection**: Identify unauthorized or suspicious trades
3. **Quality Control**: Check if brokers are executing only what clients requested
4. **Dispute Resolution**: Provide evidence of what was discussed in conversations

## How It Works

### Step 1: Input Conversation JSON
Provide a conversation JSON that includes:
- `metadata.hkt_datetime` or `transactions[].hkt_datetime`: Date of the conversation
- `metadata.client_id` (optional): Filter trades by client
- `conversations` or `transcriptions`: The actual conversation text

Example:
```json
{
  "metadata": {
    "hkt_datetime": "2025-10-09T09:30:00",
    "client_id": "M9136",
    "broker_id": "0489"
  },
  "conversations": [
    [
      {"speaker": "Broker", "text": "你好！"},
      {"speaker": "Client", "text": "我想沽安東油田服務..."}
    ]
  ]
}
```

### Step 2: Load Trade Records
The tool will:
1. Extract the date from the conversation
2. Load all trade records from `trades.csv` for that date
3. Optionally filter by `client_id`

### Step 3: LLM Analysis
For each trade record, the LLM analyzes:
- Stock code/name matches
- Quantity matches
- Price matches
- Buy/Sell direction matches
- Timing and context

### Step 4: Get Results
You'll receive:

#### 📊 Formatted Text Output
- Conversation summary
- Overall assessment
- Confidence summary statistics
- Individual record analysis with reasoning

#### 📄 JSON Output
Complete structured data including:
```json
{
  "status": "success",
  "analysis_info": {
    "date": "2025-10-09",
    "client_id": "M9136",
    "total_records": 10
  },
  "analysis_result": {
    "records_analyzed": [
      {
        "order_no": "78239686",
        "confidence_score": 0.95,
        "reasoning": "Client explicitly mentioned selling 安東油田服務 (stock 3337) with quantity 20000 at price 1.23...",
        "matched_conversation_segments": [
          "我想沽啲安東油田服務，三三三七",
          "兩萬股，一蚊二三沽"
        ]
      },
      {
        "order_no": "78240439",
        "confidence_score": 0.1,
        "reasoning": "No mention of this stock (358 江西銅業股份) in the conversation...",
        "matched_conversation_segments": []
      }
    ],
    "total_confidence_summary": {
      "average_confidence": 0.525,
      "high_confidence_count": 1,
      "medium_confidence_count": 2,
      "low_confidence_count": 7
    },
    "conversation_summary": "Client called to sell 安東油田服務 stock...",
    "overall_assessment": "Only 1 out of 10 trades was clearly discussed..."
  }
}
```

## Confidence Score Interpretation

| Score | Meaning | Example |
|-------|---------|---------|
| **1.0** | Definitely mentioned with clear confirmation | Stock code, quantity, price all explicitly stated and confirmed |
| **0.7-0.9** | Strong evidence, likely mentioned | Most details match, minor ambiguities |
| **0.4-0.6** | Some evidence but not clear | Partial match, could be referring to this trade |
| **0.1-0.3** | Possibly related but very uncertain | Weak connection, likely not this trade |
| **0.0** | Definitely NOT mentioned | No evidence whatsoever |

## Settings

### Model Selection
- **Recommended**: `qwen2.5:32b` or `qwen2.5:14b` for best accuracy
- **Faster**: `qwen2.5:7b` for quicker analysis
- Temperature: 0.3 (default) for focused analysis

### Client ID Filter
- **Leave empty**: Analyze ALL trades for that date (all clients)
- **Specify client ID**: Only analyze trades for that specific client

## Example Workflow

### Example 1: Single Client Analysis
```
Input: Conversation JSON with client M9136 on 2025-10-09
Filter: client_id = "M9136"
Result: 10 trades found, 1 with high confidence (0.95), 9 with low confidence
Conclusion: Only 1 trade was actually discussed in the call
```

### Example 2: All Clients (Suspicious Activity)
```
Input: Conversation JSON on 2025-10-09
Filter: (empty - all clients)
Result: 50 trades found across 10 clients
Finding: 3 trades have high confidence, but they're for different clients!
Alert: Possible unauthorized trading or mixed-up orders
```

## Tips for Best Results

1. **Complete Transcriptions**: More detailed conversations = better analysis
2. **Accurate Timestamps**: Ensure `hkt_datetime` matches the actual call date
3. **Include Both Speakers**: Broker confirmations improve confidence detection
4. **Use Stock Names**: LLM can match even with STT errors in stock codes
5. **Review Medium Scores**: 0.4-0.6 scores may need manual review

## Comparison with Trade Verification

| Feature | Trade Verification | Conversation Record Analysis |
|---------|-------------------|------------------------------|
| **Input** | Detected transactions | Conversation + date |
| **Direction** | Transactions → Records | Records → Conversation |
| **Purpose** | Find matching records | Verify records were discussed |
| **Output** | Matching CSV records | Confidence scores |
| **Use Case** | "Did this trade execute?" | "Was this trade authorized?" |

## Sample Files

- `sample_conversation_for_record_analysis.json`: Example conversation JSON
- `trades.csv`: Trade records database

## Troubleshooting

**No records found**
- Check that `hkt_datetime` is correct
- Verify `client_id` exists in trades.csv
- Ensure trades.csv has data for that date

**All low confidence scores**
- Conversation may not discuss any specific trades
- Check if conversation is complete (not cut off)
- Try different LLM model

**Error parsing JSON**
- Ensure JSON is valid (use JSON validator)
- Check that `hkt_datetime` field exists
- Verify structure matches expected format

## Technical Details

### Date Extraction
The tool looks for `hkt_datetime` in this order:
1. `metadata.hkt_datetime`
2. `transactions[0].hkt_datetime`

### Record Matching
- Primary filter: Date (exact match on date, ignoring time)
- Secondary filter: Client ID (optional)
- No other pre-filtering - LLM analyzes all matching records

### LLM Analysis
- Uses structured output (Pydantic models) for reliable JSON
- Single batch analysis for efficiency
- System prompt guides confidence calibration
- Temperature 0.3 for consistent scoring

## Future Enhancements

- [ ] Batch analysis across multiple dates
- [ ] Anomaly detection (highlight unusual patterns)
- [ ] Comparative analysis (broker A vs broker B)
- [ ] Export to audit report format
- [ ] Integration with compliance systems

