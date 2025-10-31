# Transaction Analysis - Multiple Transactions Support

## Overview

The Transaction Analysis tab now supports analyzing conversations with:
- **Multiple transactions** (e.g., buying one stock and selling another)
- **Single transaction** (e.g., only buying one stock)
- **No transactions** (e.g., just inquiry calls)

## Pydantic Model Structure

### Transaction (Single Transaction)
```python
class Transaction(BaseModel):
    transaction_type: Literal["buy", "sell", "queue"]
    confidence_score: float  # 0.0 to 2.0
    stock_code: Optional[str]
    stock_name: Optional[str]
    quantity: Optional[str]
    price: Optional[str]
    explanation: str
```

### TransactionAnalysisResult (Complete Result)
```python
class TransactionAnalysisResult(BaseModel):
    transactions: list[Transaction]  # Can be empty list
    transcription_comparison: str
    overall_summary: str
```

## Example JSON Outputs

### Example 1: Multiple Transactions
```json
{
  "transactions": [
    {
      "transaction_type": "buy",
      "confidence_score": 2.0,
      "stock_code": "0700",
      "stock_name": "騰訊",
      "quantity": "1000股",
      "price": "350元",
      "explanation": "客戶明確要求買入騰訊，券商重複確認了股票代號、數量和價格"
    },
    {
      "transaction_type": "sell",
      "confidence_score": 1.8,
      "stock_code": "9988",
      "stock_name": "阿里巴巴",
      "quantity": "500股",
      "price": "市價",
      "explanation": "客戶要求賣出阿里巴巴，使用市價單"
    }
  ],
  "transcription_comparison": "兩個轉錄文字在股票代號上一致，但在數量的表達上略有差異...",
  "overall_summary": "這次通話涉及兩筆交易：買入騰訊和賣出阿里巴巴，客戶都有明確確認..."
}
```

### Example 2: Single Transaction
```json
{
  "transactions": [
    {
      "transaction_type": "queue",
      "confidence_score": 1.5,
      "stock_code": "0005",
      "stock_name": "滙豐",
      "quantity": "2000股",
      "price": "65元",
      "explanation": "客戶要求排隊買入滙豐，但尚未確認是否已成交"
    }
  ],
  "transcription_comparison": "兩個轉錄在價格上略有差異...",
  "overall_summary": "客戶排隊買入滙豐控股2000股..."
}
```

### Example 3: No Transactions
```json
{
  "transactions": [],
  "transcription_comparison": "兩個轉錄文字基本一致，都只是查詢股價...",
  "overall_summary": "這次通話只是查詢滙豐的當前股價，沒有進行任何交易"
}
```

## UI Display

### Summary Result Box
Shows a formatted summary of all transactions:
```
📊 交易分析結果
==================================================

📋 總共識別到 2 個交易

──────────────────────────────────────────────────
交易 #1
──────────────────────────────────────────────────
🔖 交易類型: buy
⭐ 置信度分數: 2.0 / 2.0
📈 股票代號: 0700
🏢 股票名稱: 騰訊
🔢 數量: 1000股
💰 價格: 350元

📝 分析說明:
客戶明確要求買入騰訊...

──────────────────────────────────────────────────
交易 #2
──────────────────────────────────────────────────
🔖 交易類型: sell
⭐ 置信度分數: 1.8 / 2.0
...
```

### JSON Result Box
Shows the complete JSON structure with proper formatting (2-space indentation, Unicode support)

### Individual Fields
For backwards compatibility, individual fields show the **first transaction** (or empty if no transactions found)

## System Message

The system message has been updated to instruct the LLM to:
1. Look for **all** transactions in the conversation
2. Return a list of transactions (can be empty)
3. Provide overall summary and transcription comparison

## Benefits

✅ **Comprehensive Analysis**: Captures all transactions in a single conversation
✅ **Flexible**: Handles 0, 1, or multiple transactions
✅ **Structured**: Clear JSON format for programmatic processing
✅ **Backwards Compatible**: Individual fields still work for simple cases
✅ **Better Summary**: Overall summary provides context for all transactions

