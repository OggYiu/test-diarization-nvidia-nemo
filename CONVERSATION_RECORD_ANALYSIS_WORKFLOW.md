# Conversation Record Analysis - Workflow Diagram

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSATION RECORD ANALYSIS                  │
│             Verify which trades were discussed in calls          │
└─────────────────────────────────────────────────────────────────┘

INPUT                           PROCESSING                         OUTPUT
━━━━━                           ━━━━━━━━━━                         ━━━━━━

┌──────────────┐               ┌──────────────┐                ┌──────────────┐
│ Conversation │               │  Extract     │                │   Formatted  │
│     JSON     │──────────────>│    Date      │                │     Text     │
│              │               │              │                │   Results    │
│ • metadata   │               │ hkt_datetime │                │              │
│ • convos     │               └──────┬───────┘                │ • Summary    │
│ • trans      │                      │                        │ • Assessment │
└──────────────┘                      │                        │ • Individual │
                                      ▼                        │   Scores     │
┌──────────────┐               ┌──────────────┐                │ • Reasoning  │
│  trades.csv  │               │ Load Trades  │                └──────────────┘
│              │──────────────>│   for Date   │
│ • All trades │               │              │                ┌──────────────┐
│ • Full DB    │               │ Filter by:   │                │     JSON     │
└──────────────┘               │ • Date       │                │    Output    │
                               │ • Client ID  │                │              │
┌──────────────┐               └──────┬───────┘                │ • Structured │
│   Settings   │                      │                        │ • Complete   │
│              │                      │                        │ • Machine    │
│ • Model      │                      ▼                        │   Readable   │
│ • Temp       │               ┌──────────────┐                └──────────────┘
│ • Client ID  │──────────────>│  Format for  │
└──────────────┘               │     LLM      │
                               │              │
                               │ Conversation │
                               │   +          │
                               │ Trade List   │
                               └──────┬───────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │  LLM         │
                               │  Analysis    │
                               │              │
                               │ For EACH     │
                               │ trade:       │
                               │ • Compare    │
                               │ • Score      │
                               │ • Reason     │
                               │ • Extract    │
                               └──────┬───────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │  Aggregate   │
                               │   Results    │
                               │              │
                               │ • Statistics │
                               │ • Summary    │
                               │ • Overall    │
                               └──────────────┘
```

## Detailed Process Flow

### Phase 1: Input & Validation
```
User Input → Parse JSON → Validate Structure → Extract Date
                ↓                               ↓
            Error?                          Success
                ↓                               ↓
          Show Error                    Continue
```

### Phase 2: Data Loading
```
Target Date + Client ID (optional)
            ↓
    Open trades.csv
            ↓
    Read all rows
            ↓
    Filter by date (exact match on date part)
            ↓
    Filter by client_id (if provided)
            ↓
    Matching Records
            ↓
    Empty?  → Error: No records found
    
    Has records → Continue
```

### Phase 3: LLM Analysis
```
For each trade record:
    ┌─────────────────────────────────────┐
    │ 1. Format trade details             │
    │    • Stock code, name               │
    │    • Quantity, price                │
    │    • Buy/Sell, time                 │
    └───────────────┬─────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │ 2. Search conversation for mentions │
    │    • Stock code variants            │
    │    • Stock name (Cantonese/English) │
    │    • Quantity mentions              │
    │    • Price mentions                 │
    │    • Action words (buy/sell)        │
    └───────────────┬─────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │ 3. Calculate confidence             │
    │    • Exact matches → High (0.8-1.0) │
    │    • Partial match → Medium (0.4-0.7│
    │    • No match → Low (0.0-0.3)       │
    └───────────────┬─────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │ 4. Generate reasoning               │
    │    • List matching elements         │
    │    • Note discrepancies             │
    │    • Cite conversation segments     │
    └───────────────┬─────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │ 5. Extract evidence                 │
    │    • Quote conversation segments    │
    │    • List matched phrases           │
    └─────────────────────────────────────┘
```

### Phase 4: Aggregation & Output
```
Individual Analyses
        ↓
┌───────────────────┐
│ Calculate Stats   │
│ • Average score   │
│ • High count      │
│ • Medium count    │
│ • Low count       │
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Generate Summary  │
│ • What discussed  │
│ • Main topics     │
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Overall Assessment│
│ • Match quality   │
│ • Concerns        │
│ • Recommendations │
└────────┬──────────┘
         ↓
┌───────────────────┐
│ Format Outputs    │
│ • Text (readable) │
│ • JSON (machine)  │
└───────────────────┘
```

## Data Flow Example

### Example Scenario

**Input Conversation:**
```
Client: "我想沽安東油田服務三三三七，兩萬股，一蚊二三"
Broker: "收到，三三三七沽兩萬一蚊二三"
```

**Trades.csv Records (2025-10-09, Client M9136):**

| OrderNo | Stock | Name | Side | Qty | Price | Time |
|---------|-------|------|------|-----|-------|------|
| 78239686 | 3337 | 安東油田服務 | A (Sell) | 20000 | 1.23 | 09:30:52 |
| 78239778 | 175 | 吉利汽車 | A | 2000 | 19.63 | 09:31:31 |
| 78239926 | 358 | 江西銅業股份 | A | 6000 | 34.7 | 09:32:45 |

**LLM Analysis Flow:**

```
Record #1: 78239686
├─ Stock: 3337 安東油田服務
├─ Conversation mentions: "安東油田服務三三三七" ✓
├─ Side: Sell (A)
├─ Conversation mentions: "沽" (sell) ✓
├─ Quantity: 20000
├─ Conversation mentions: "兩萬" (20000) ✓
├─ Price: 1.23
├─ Conversation mentions: "一蚊二三" (1.23) ✓
└─ CONFIDENCE: 0.95 ✅ HIGH

Record #2: 78239778
├─ Stock: 175 吉利汽車
├─ Conversation mentions: (none) ✗
├─ No matching elements found
└─ CONFIDENCE: 0.05 ❌ LOW

Record #3: 78239926
├─ Stock: 358 江西銅業股份
├─ Conversation mentions: (none) ✗
├─ No matching elements found
└─ CONFIDENCE: 0.05 ❌ LOW

Summary Statistics:
├─ Total records: 3
├─ Average confidence: 0.35
├─ High confidence (≥0.7): 1
├─ Medium confidence (0.4-0.7): 0
└─ Low confidence (<0.4): 2

Overall Assessment:
"Only 1 out of 3 trades was clearly discussed in the conversation.
The other 2 trades (吉利汽車, 江西銅業股份) were NOT mentioned
and should be flagged for review."
```

## Confidence Score Decision Tree

```
                    Does conversation mention
                      this stock at all?
                            │
                ┌───────────┴───────────┐
               NO                       YES
                │                        │
           Score ≤ 0.2              Does quantity
                │                    match exactly?
                │                        │
                │            ┌───────────┴───────────┐
                │           NO                       YES
                │            │                        │
                │       Score 0.4-0.6            Does price
                │            │                    match?
                │            │                        │
                │            │            ┌───────────┴───────────┐
                │            │           NO                       YES
                │            │            │                        │
                │            │       Score 0.6-0.8           Is buy/sell
                │            │            │                  correct?
                │            │            │                        │
                │            │            │            ┌───────────┴───────────┐
                │            │            │           NO                       YES
                │            │            │            │                        │
                │            │            │       Score 0.7-0.9           Score 0.9-1.0
                │            │            │                                     │
                │            │            │                                CONFIRMED ✓
                └────────────┴────────────┴─────────────────────────────────────┘
```

## Integration with Other Tabs

```
                    ┌────────────────────┐
                    │ Audio Recording    │
                    └─────────┬──────────┘
                              ↓
                    ┌────────────────────┐
                    │   Diarization      │
                    │   (Speaker Sep)    │
                    └─────────┬──────────┘
                              ↓
                    ┌────────────────────┐
                    │   STT Tab          │
                    │   (Transcription)  │
                    └─────────┬──────────┘
                              ↓
                    ┌────────────────────┐
                    │ Transaction        │
                    │ Analysis Tab       │
                    │ (Extract TXs)      │
                    └─────────┬──────────┘
                              ↓
            ┌─────────────────┼─────────────────┐
            ↓                                    ↓
┌──────────────────────┐          ┌──────────────────────┐
│ Trade Verification   │          │ Conversation Record  │
│                      │          │ Analysis             │
│ TX → trades.csv      │          │ trades.csv → Conv    │
│ "Did TX execute?"    │          │ "Was trade discussed?"│
└──────────────────────┘          └──────────────────────┘
            │                                    │
            └─────────────────┬──────────────────┘
                              ↓
                    ┌────────────────────┐
                    │ Compliance Report  │
                    │ (Future)           │
                    └────────────────────┘
```

## Key Differences: This Tab vs Others

### Transaction Analysis Tab
```
INPUT:  Conversation
OUTPUT: Extracted transactions
GOAL:   What did they discuss?
```

### Trade Verification Tab
```
INPUT:  Transactions + trades.csv
OUTPUT: Matching records
GOAL:   Did these trades execute?
```

### Conversation Record Analysis Tab (NEW!)
```
INPUT:  Conversation + trades.csv
OUTPUT: Confidence scores for all trades
GOAL:   Which trades were discussed?
```

## Performance Characteristics

```
Number of Records vs Processing Time
(qwen2.5:14b on typical hardware)

Records     Time      Memory
────────────────────────────
1-5         5s        8GB
5-10        10s       8GB
10-20       20s       8GB
20-50       45s       8GB
50+         90s+      8GB

Note: All records analyzed in ONE LLM call
for efficiency (single prompt with all trades)
```

## Error Handling Flow

```
                    ┌─────────────┐
                    │  Start      │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Parse JSON  │
                    └──────┬──────┘
                           │
                ┌──────────┴──────────┐
               OK                  ERROR
                │                     │
                ↓                     ↓
         ┌─────────────┐      ┌─────────────┐
         │Extract Date │      │Show Error   │
         └──────┬──────┘      │Return Early │
                │             └─────────────┘
     ┌──────────┴──────────┐
    OK                   ERROR
     │                      │
     ↓                      ↓
┌─────────────┐      ┌─────────────┐
│Load Trades  │      │Show Error   │
└──────┬──────┘      │Return Early │
       │             └─────────────┘
┌──────┴──────┐
│             │
↓             ↓
Found      Not Found
│             │
│             ↓
│      ┌─────────────┐
│      │Show Error   │
│      │Return Early │
│      └─────────────┘
↓
┌─────────────┐
│LLM Analysis │
└──────┬──────┘
       │
┌──────┴──────┐
│             │
↓             ↓
Success   Structured Output Failed
│             │
│             ↓
│      ┌─────────────┐
│      │Fallback to  │
│      │Raw Output   │
│      └──────┬──────┘
│             │
└──────┬──────┘
       ↓
┌─────────────┐
│Return Result│
└─────────────┘
```

---

## Quick Reference

### What goes IN
- ✅ Conversation JSON with `hkt_datetime`
- ✅ Optional: client_id filter
- ✅ Settings: model, temperature, trades file path

### What comes OUT
- ✅ Confidence score (0.0-1.0) for EACH trade
- ✅ Detailed reasoning for EACH score
- ✅ Conversation segments that match
- ✅ Summary statistics
- ✅ Overall assessment

### What it DOES
- ✅ Finds all trades for that date
- ✅ Compares each against conversation
- ✅ Uses LLM to determine if discussed
- ✅ Provides evidence and reasoning

### What it's GOOD FOR
- ✅ Compliance verification
- ✅ Fraud detection
- ✅ Quality control
- ✅ Dispute resolution
- ✅ Audit trails

