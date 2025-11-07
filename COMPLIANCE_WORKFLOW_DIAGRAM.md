# 🛡️ Compliance Analysis Workflow Diagram

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    📞 START: Audio Recording                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    🎤 Tab 1: STT (Speech-to-Text)               │
│  • Transcribes audio to text                                    │
│  • Outputs: Conversation JSON                                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ shared_conversation_json
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               🔟 Tab 2: JSON Batch Analysis                      │
│  • Merges multiple transcription models                         │
│  • Outputs: Merged stocks JSON                                  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ shared_merged_stocks_json
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│           📊 Tab 3: Transaction Analysis JSON                    │
│  • Extracts transaction details from conversation               │
│  • Identifies: stock codes, prices, quantities, buy/sell        │
│  • Outputs: Transaction JSON                                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ shared_transaction_json
                                ▼
                      ┌─────────┴─────────┐
                      │                   │
                      ▼                   ▼
    ┌─────────────────────────┐   ┌─────────────────────────┐
    │ 🔍 Tab 4:               │   │ 🎯 Tab 5:               │
    │ Trade Verification      │   │ Conversation Record     │
    │                         │   │ Analysis                │
    │ Direction:              │   │ Direction:              │
    │ Conversation → Trades   │   │ Trades → Conversation   │
    │                         │   │                         │
    │ Inputs:                 │   │ Inputs:                 │
    │ • Transaction JSON      │   │ • Conversation JSON     │
    │ • trades.csv            │   │ • trades.csv            │
    │                         │   │ • Client ID (optional)  │
    │ Process:                │   │                         │
    │ • Match each            │   │ Process:                │
    │   transaction against   │   │ • LLM analyzes if       │
    │   trade records         │   │   each trade record     │
    │ • Check: code, price,   │   │   was mentioned in      │
    │   quantity, time        │   │   conversation          │
    │                         │   │                         │
    │ Outputs:                │   │ Outputs:                │
    │ • Verification JSON     │   │ • Analysis JSON         │
    │ • report.csv           │   │ • verify.csv            │
    │ • Confidence (0-100%)   │   │ • Confidence (0.0-1.0)  │
    └────────────┬────────────┘   └────────────┬────────────┘
                 │                             │
                 │ shared_trade_             │ shared_conversation_
                 │ verification_json         │ analysis_json
                 │                             │
                 └────────────┬────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────┐
        │      🛡️ Tab 6: Compliance Analysis             │
        │                                                 │
        │  Dual Analysis Integration                      │
        │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                 │
        │                                                 │
        │  Analyzes Results From Both:                    │
        │  • Trade Verification (Conv → Trades)          │
        │  • Conversation Record Analysis (Trades → Conv)│
        │                                                 │
        │  Calculates:                                    │
        │  ✓ Overall confidence score (0.0-1.0)          │
        │  ✓ Compliance level                            │
        │  ✓ Human review necessity                      │
        │  ✓ Risk factors                                │
        │  ✓ Actionable recommendations                  │
        │                                                 │
        │  Outputs:                                       │
        │  • Compliance Report (Text)                    │
        │  • Compliance JSON                             │
        │  • compliance.csv                              │
        └─────────────────────┬───────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────┐
        │          📋 COMPLIANCE DECISION                 │
        │                                                 │
        │  If confidence ≥ 0.8:                          │
        │    ✅ COMPLIANT → No review needed             │
        │                                                 │
        │  If confidence 0.6-0.79:                       │
        │    ⚠️ LIKELY COMPLIANT → Optional review       │
        │                                                 │
        │  If confidence 0.4-0.59:                       │
        │    ⚠️ UNCLEAR → Review recommended             │
        │                                                 │
        │  If confidence < 0.4:                          │
        │    ❌ NON-COMPLIANT → Mandatory review         │
        └─────────────────────────────────────────────────┘
```

## Two-Way Verification Explained

### Why Two Analyses?

The compliance system uses **bidirectional verification** to catch different types of issues:

#### 🔍 Analysis 1: Trade Verification (Conversation → Trades)
**Question**: *"Were the trades mentioned in the conversation actually executed?"*

**Detects**:
- ✅ Confirmed trades (mentioned AND executed)
- ❌ Missed trades (mentioned but NOT executed)
- 🔍 Needs investigation if conversation mentions trades not in records

**Example Issue**:
```
Client: "Buy 10,000 shares of 0700 at $400"
Broker: "OK, I'll place the order"
→ BUT: No matching trade in trades.csv
→ FLAG: Potential undocumented trade
```

#### 🎯 Analysis 2: Conversation Record Analysis (Trades → Conversation)
**Question**: *"Were the executed trades actually discussed in the conversation?"*

**Detects**:
- ✅ Authorized trades (discussed AND executed)
- ❌ Unauthorized trades (executed but NOT discussed)
- 🔍 Needs investigation if trades exist without conversation evidence

**Example Issue**:
```
trades.csv shows: Buy 5,000 shares of 0941 at $50
Conversation: No mention of stock 0941 at all
→ FLAG: Potential unauthorized trade
```

### Combining Both Analyses

The **Compliance Analysis** tab combines both to give the complete picture:

```
┌──────────────────────┬──────────────────────┬─────────────────────┐
│   Conversation       │   Trade Records      │   Assessment        │
│   Says               │   Show               │                     │
├──────────────────────┼──────────────────────┼─────────────────────┤
│ ✅ Mentioned         │ ✅ Executed          │ ✅ Perfect match    │
│ High confidence      │ High confidence      │ → Compliant         │
├──────────────────────┼──────────────────────┼─────────────────────┤
│ ✅ Mentioned         │ ❌ Not executed      │ ⚠️ Missing trade    │
│ High confidence      │ No match found       │ → Investigate       │
├──────────────────────┼──────────────────────┼─────────────────────┤
│ ❌ Not mentioned     │ ✅ Executed          │ ⚠️ Unauthorized?    │
│ Low confidence       │ Exists in records    │ → Investigate       │
├──────────────────────┼──────────────────────┼─────────────────────┤
│ ❌ Not mentioned     │ ❌ Not executed      │ ✅ Consistent       │
│ No evidence          │ No record            │ → OK (nothing done) │
└──────────────────────┴──────────────────────┴─────────────────────┘
```

## Data Flow Visualization

```
Input Data:
┌─────────────────┐
│ Audio File      │
│ + trades.csv    │
│ + client info   │
└────────┬────────┘
         │
         ▼
Processing Steps:
┌─────────────────┐
│ 1. Transcribe   │──┐
└─────────────────┘  │
                     │
┌─────────────────┐  │
│ 2. Merge        │◄─┘
└─────────────────┘  │
                     │
┌─────────────────┐  │
│ 3. Extract TX   │◄─┘
└─────────────────┘
         │
         ├──────────────┬────────────────┐
         ▼              ▼                │
┌──────────────┐  ┌──────────────┐     │
│ 4a. Verify   │  │ 4b. Analyze  │     │
│     Conv→Tr  │  │     Tr→Conv  │     │
│              │  │              │     │
│ report.csv   │  │ verify.csv   │     │
└──────┬───────┘  └──────┬───────┘     │
       │                 │             │
       └────────┬────────┘             │
                ▼                      │
       ┌─────────────────┐             │
       │ 5. Compliance   │◄────────────┘
       │    Analysis     │
       │                 │
       │ compliance.csv  │
       └─────────────────┘
                │
                ▼
        Decision Output:
        ┌─────────────────┐
        │ • Confidence    │
        │ • Review needed?│
        │ • Risk factors  │
        │ • Actions       │
        └─────────────────┘
```

## Output Files Summary

```
📁 Project Directory
│
├── 📄 report.csv ──────────── Trade Verification results
│   ├─ One row per transaction analyzed
│   ├─ Columns: client_id, broker_id, stock_code, etc.
│   └─ Confidence: 0-100% (how well tx matches trade)
│
├── 📄 verify.csv ──────────── Conversation Record Analysis results
│   ├─ One row per trade record analyzed
│   ├─ Columns: order_no, stock_code, confidence_score, etc.
│   └─ Confidence: 0.0-1.0 (how well trade matches conv)
│
└── 📄 compliance.csv ──────── Compliance Analysis results (NEW)
    ├─ One row per complete analysis
    ├─ Combines metrics from both report.csv and verify.csv
    ├─ Columns: overall_confidence, compliance_status, etc.
    └─ Tracks compliance history over time
```

## Quick Reference

### Confidence Thresholds

| Score Range | Level | Status | Action |
|------------|-------|--------|--------|
| ≥ 0.8 | High | ✅ COMPLIANT | No review |
| 0.6 - 0.79 | Medium | ⚠️ LIKELY COMPLIANT | Optional review |
| 0.4 - 0.59 | Low | ⚠️ UNCLEAR | Review recommended |
| < 0.4 | Very Low | ❌ NON-COMPLIANT | Mandatory review |

### When Human Review is Needed

- [ ] Overall confidence < 0.7
- [ ] Transactions mentioned but not in trades.csv
- [ ] Trade records not mentioned in conversation
- [ ] More low confidence than high confidence matches
- [ ] Count mismatch (transactions ≠ records)

If **ANY** checkbox is true → Human review recommended

### Key Metrics

**From Trade Verification** (report.csv):
- Total transactions analyzed
- Best match confidence per transaction
- Matched vs. unmatched transactions

**From Conversation Record Analysis** (verify.csv):
- Total trade records analyzed
- Confidence per record (0.0-1.0)
- Records with/without conversation evidence

**From Compliance Analysis** (compliance.csv):
- Overall confidence (average of all scores)
- High/medium/low confidence match counts
- Unmatched transactions and records
- Risk factors identified
- Review reasons

## Usage Tips

1. **Always run both prerequisite tabs**: The compliance analysis needs BOTH verification and conversation analysis results.

2. **Use the load buttons**: Instead of copying/pasting JSON, use the "Load" buttons for automatic data transfer.

3. **Check the timestamps**: Ensure all analyses are from the same conversation (same datetime).

4. **Review low confidence cases**: Even if overall confidence is acceptable, individual low-confidence items may need attention.

5. **Track over time**: Use compliance.csv to identify patterns across multiple conversations.

6. **Don't ignore warnings**: Risk factors and review reasons are there for a reason - investigate them.

7. **Human judgment is key**: The system provides guidance, but final compliance decisions should involve human review, especially for edge cases.

---

**This workflow ensures comprehensive compliance verification through dual analysis and intelligent automation while highlighting cases that need human attention.**

