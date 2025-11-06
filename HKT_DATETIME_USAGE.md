# HKT DateTime Usage for Transaction Matching

## Overview

The transaction JSON now includes a `hkt_datetime` field for each transaction, which is used to more accurately match transactions against trade records in `trades.csv`. This allows for precise time-based matching when verifying if a transaction mentioned in a call recording actually occurred.

## Key Changes

### 1. Transaction JSON Structure Update

Each transaction now includes:
- `conversation_number`: The conversation this transaction came from
- `hkt_datetime`: The Hong Kong time when the conversation/transaction occurred
- Other fields: `transaction_type`, `stock_code`, `stock_name`, `quantity`, `price`, `confidence_score`, `explanation`

**Example:**
```json
{
  "transactions": [
    {
      "transaction_type": "buy",
      "confidence_score": 2.0,
      "conversation_number": 1,
      "hkt_datetime": "2025-10-20T10:15:30",
      "stock_code": "18138",
      "stock_name": "騰訊認購證",
      "quantity": "20000",
      "price": "0.38",
      "explanation": "..."
    }
  ]
}
```

### 2. Automatic DateTime Extraction

The system automatically extracts `hkt_datetime` from conversation metadata and adds it to each transaction:

**In `tab_transaction_analysis_json.py` (lines 408-410, 650-656):**
- Extracts `hkt_datetime` from `metadata.hkt_datetime` in the conversation JSON
- Maps each `conversation_number` to its corresponding datetime
- Programmatically adds the datetime to each transaction based on its `conversation_number`

### 3. Enhanced Trade Verification

**In `tab_trade_verification.py` (line 82-87):**
- **Priority-based datetime selection**: Uses `hkt_datetime` from the transaction itself first, falls back to metadata if not present
- This allows transactions from different times within the same conversation to be matched correctly
- Creates a time window (±X hours) around each transaction's datetime
- Searches `trades.csv` for records within that time window

## How It Works

### Step 1: Conversation JSON with Metadata

Input conversation JSON must include `hkt_datetime` in metadata:

```json
[
  {
    "conversation_number": 1,
    "filename": "call_001.wav",
    "metadata": {
      "broker_name": "Dickson Lau",
      "client_name": "CHENG SUK HING",
      "client_id": "P77197",
      "broker_id": "0489",
      "hkt_datetime": "2025-10-20T10:15:30"
    },
    "transcriptions": {
      "sensevoice": "經紀: 你好\n客戶: 我想買騰訊"
    }
  }
]
```

### Step 2: Transaction Analysis

The Transaction Analysis tab:
1. Reads the conversation JSON
2. Extracts `hkt_datetime` from metadata
3. Analyzes the conversation to identify transactions
4. Automatically adds `hkt_datetime` to each identified transaction

### Step 3: Trade Verification

The Trade Verification tab:
1. Reads the transaction JSON (with `hkt_datetime` for each transaction)
2. For each transaction:
   - Uses the transaction's `hkt_datetime` (or falls back to metadata if not present)
   - Creates a time window (e.g., ±1 hour) around that datetime
   - Searches `trades.csv` for matching records within the time window
3. Matches based on:
   - ✅ **DateTime**: OrderTime within time window
   - ✅ **Client ID**: ACCode matches client_id
   - ✅ **Broker ID**: AECode matches broker_id
   - ✅ **Stock Code**: SCTYCode matches stock_code
   - ✅ **Order Side**: Buy (B) / Sell (A)
   - ✅ **Quantity**: OrderQty matches quantity
   - ✅ **Price**: OrderPrice matches price

## Benefits

### 1. **Accurate Time Matching**
Each transaction uses its specific datetime for matching, not a generic call datetime. This is crucial when:
- Multiple transactions occur at different times in the same call
- Calls span a long duration
- Transactions are discussed before or after they actually occur

### 2. **Flexible Fallback**
If a transaction doesn't have its own `hkt_datetime`, the system falls back to the metadata datetime, ensuring backward compatibility.

### 3. **Better Time Window Management**
The time window is applied individually to each transaction, not to the entire call, resulting in more accurate matches.

## DateTime Format

The system accepts multiple datetime formats:
- ISO format: `"2025-10-20T10:15:30"`
- Extended format: `"2025-10-20 10:15:30.123456"`
- Simple format: `"2025-10-20 10:15:30"`

All times are assumed to be in Hong Kong Time (HKT).

## Configuration

### Time Window Setting

In the Trade Verification tab, you can adjust the time window:
- **Default**: ±1.0 hours
- **Range**: 0.5 to 24.0 hours
- **Purpose**: Searches for trades within X hours before and after the transaction datetime

**When to adjust:**
- Increase if trades are recorded with delays
- Decrease for more precise matching
- Consider market hours and trading patterns

## Example Workflow

### Complete Example

**1. Input Conversation JSON:**
```json
[
  {
    "conversation_number": 1,
    "filename": "call_20251020.wav",
    "metadata": {
      "client_id": "P77197",
      "broker_id": "0489",
      "hkt_datetime": "2025-10-20T10:15:30"
    },
    "transcriptions": {
      "sensevoice": "客戶說要買入18138騰訊認購證，兩萬股，三毫八"
    }
  }
]
```

**2. Transaction Analysis Output:**
```json
{
  "transactions": [
    {
      "transaction_type": "buy",
      "confidence_score": 1.0,
      "conversation_number": 1,
      "hkt_datetime": "2025-10-20T10:15:30",
      "stock_code": "18138",
      "stock_name": "騰訊認購證",
      "quantity": "20000",
      "price": "0.38",
      "explanation": "客戶明確要求買入18138騰訊認購證..."
    }
  ]
}
```

**3. Trade Verification:**
- Searches `trades.csv` for client P77197
- Looks for records between 09:15:30 and 11:15:30 (±1 hour)
- Matches stock code 18138, order side B (buy), quantity 20000, price 0.38
- Returns confidence score based on matching criteria

## Files Modified

1. **`sample_transaction.json`**: Updated with `hkt_datetime` examples
2. **`tabs/tab_transaction_analysis_json.py`**: 
   - Added datetime extraction from metadata
   - Updated system message to include datetime information
   - Programmatically adds datetime to each transaction
3. **`tabs/tab_trade_verification.py`**:
   - Priority-based datetime selection (transaction first, metadata fallback)
   - Enhanced verification output with datetime information
   - Updated documentation

## Best Practices

1. **Always Include DateTime in Metadata**: Ensure conversation JSON has `hkt_datetime` in metadata
2. **Use ISO Format**: Prefer `YYYY-MM-DDTHH:MM:SS` format for consistency
3. **Adjust Time Window**: Based on your specific use case and data characteristics
4. **Review Matches**: Check confidence scores and matching criteria for each verified transaction
5. **Handle Multiple Transactions**: When a call has multiple transactions at different times, each will use its own datetime for matching

## Troubleshooting

### No Matches Found
- Check if `hkt_datetime` is present in the transaction or metadata
- Verify the datetime format is valid
- Increase the time window if trades are recorded with delays
- Ensure client_id and broker_id are correct

### Multiple Matches
- Review the confidence scores for each match
- Check if the time window is too wide
- Verify transaction details (stock code, quantity, price) are accurate

### DateTime Not Available
- Error: "No HKT datetime provided in transaction or metadata"
- Solution: Add `hkt_datetime` to the conversation metadata before analysis

## Related Documentation

- Transaction Analysis: See `tabs/tab_transaction_analysis_json.py`
- Trade Verification: See `tabs/tab_trade_verification.py`
- CSV Format: See `trades.csv` structure (OrderTime column must be datetime)

---

**Last Updated**: 2025-11-06
**Version**: 1.0

