# 🚀 Stock Extractor - Quick Start Guide

## What I Created For You

I've created a **Stock Information Extractor** that uses:
- ✅ **Pydantic** for structured data validation
- ✅ **Multiple LLM models** (you can choose which one to use)
- ✅ **Gradio** for an easy-to-use web interface

## Files Created

1. **`stock_extractor_gui.py`** - Main application with Gradio UI
2. **`test_stock_extractor.py`** - Test script to try it without the GUI
3. **`STOCK_EXTRACTOR_README.md`** - Detailed documentation
4. **`STOCK_EXTRACTOR_QUICKSTART.md`** - This file

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install pydantic langchain-core langchain-ollama gradio
```

Or if you want to update all requirements:
```bash
pip install -r requirements.txt
```

### Step 2: Make Sure Ollama is Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Make sure you have at least one model installed
ollama pull qwen3:32b
```

### Step 3: Run the Application

**Option A: Launch the Gradio GUI**
```bash
python stock_extractor_gui.py
```
Then open your browser to: http://localhost:7860

**Option B: Test from Command Line**
```bash
python test_stock_extractor.py simple
```

## How It Works

### Input (Conversation):
```
券商：你好，請問需要什麼幫助？
客戶：我想買騰訊
券商：好的，七百號騰訊，買多少？
客戶：一千股，市價買入
```

### Output (Structured JSON via Pydantic):
```json
{
  "stocks": [
    {
      "stock_number": "00700",
      "stock_name": "騰訊",
      "confidence": "high",
      "reasoning": "對話中提到「七百號」，對應香港股票代碼00700"
    }
  ],
  "summary": "客戶詢問購買騰訊股票"
}
```

## Key Features

### 1. Pydantic Models
The script uses two Pydantic models to ensure structured output:

```python
class StockInfo(BaseModel):
    stock_number: str      # e.g., "00700"
    stock_name: str        # e.g., "騰訊"
    confidence: str        # "high", "medium", "low"
    reasoning: Optional[str]  # Explanation

class ConversationStockExtraction(BaseModel):
    stocks: List[StockInfo]
    summary: str
```

This guarantees:
- ✅ Consistent data structure
- ✅ Type validation
- ✅ Easy to parse and use programmatically
- ✅ Can be directly saved to database or exported

### 2. Multiple LLM Model Selection
Choose from these models (or add your own):
- `qwen3:32b` ⭐ (Recommended for Chinese)
- `deepseek-r1:32b` (Good reasoning)
- `deepseek-r1:70b` (Better quality)
- `gpt-oss:20b`
- `gemma3-27b`
- `qwen2.5:72b`
- `llama3.3:70b`

### 3. Gradio Interface
- 🎨 Beautiful, modern UI
- 📝 Text input with examples
- ⚙️ Configurable settings (model, temperature, system message)
- 📊 Formatted output + raw JSON
- 💡 Pre-loaded example conversations

## Testing Examples

### Test Simple Conversation
```bash
python test_stock_extractor.py simple
```

### Test with Multiple Stocks
```bash
python test_stock_extractor.py multiple_stocks
```

### Test with STT Errors (e.g., "一百一三八" → should be "18138")
```bash
python test_stock_extractor.py with_errors
```

### Test All Conversations
```bash
python test_stock_extractor.py all
```

### Compare Different Models
```bash
python test_stock_extractor.py compare
```

### Test with Specific Model
```bash
python test_stock_extractor.py simple deepseek-r1:32b
```

## Using in Your Code

You can import and use the function directly:

```python
from stock_extractor_gui import (
    extract_stocks_from_conversation,
    DEFAULT_SYSTEM_MESSAGE,
    ConversationStockExtraction
)
import json

conversation = """
券商：你好
客戶：我想買騰訊和小米
"""

# Get the extraction
result_text, json_str = extract_stocks_from_conversation(
    conversation_text=conversation,
    model="qwen3:32b",
    ollama_url="http://localhost:11434",
    system_message=DEFAULT_SYSTEM_MESSAGE,
    temperature=0.1,
)

# Parse JSON to Python dict
data = json.loads(json_str)

# Access structured data
for stock in data["stocks"]:
    print(f"Stock: {stock['stock_name']} ({stock['stock_number']})")
    print(f"Confidence: {stock['confidence']}")
```

## Gradio Interface Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   Stock Information Extractor                    │
├──────────────────────────────┬──────────────────────────────────┤
│  📝 Input                    │  📊 Output                        │
│                              │                                   │
│  ┌────────────────────────┐ │  ┌──────────────────────────────┐│
│  │ Conversation Text      │ │  │ Formatted Results            ││
│  │                        │ │  │                              ││
│  │                        │ │  │ • Stock #1: 騰訊 (00700)     ││
│  └────────────────────────┘ │  │   Confidence: HIGH           ││
│                              │  │                              ││
│  ⚙️ Model: qwen3:32b       │  └──────────────────────────────┘│
│  🌡️ Temperature: 0.1       │                                   │
│  🔗 Ollama: localhost:11434 │  ┌──────────────────────────────┐│
│                              │  │ Raw JSON Output              ││
│  [🚀 開始提取]              │  │ { "stocks": [...] }          ││
│                              │  └──────────────────────────────┘│
└──────────────────────────────┴──────────────────────────────────┘
```

## Configuration Options

### Temperature Setting
- **0.0 - 0.3**: More deterministic, factual (recommended for extraction)
- **0.4 - 0.7**: Balanced
- **0.8 - 2.0**: More creative, varied output

### System Message
You can customize the system message to:
- Add more examples of common errors
- Include specific terminology for your use case
- Add rules for edge cases

### Model Selection
- **For accuracy**: Use larger models (qwen2.5:72b, deepseek-r1:70b)
- **For speed**: Use smaller models (qwen3:32b, deepseek-r1:32b)
- **For Cantonese**: qwen models work best

## Common Use Cases

### 1. Post-STT Processing
```
Audio → Speech-to-Text → Stock Extractor → Database
```

### 2. Trade Verification
```
Conversation → Extract Stocks → Cross-check with trades.csv
```

### 3. Compliance Checking
```
Call Recording → Extract → Verify against order records
```

### 4. Data Analysis
```
Multiple calls → Extract all stocks → Analyze trading patterns
```

## Troubleshooting Quick Tips

| Issue | Solution |
|-------|----------|
| Connection refused | Run `ollama serve` |
| Model not found | Run `ollama pull qwen3:32b` |
| Parsing errors | Lower temperature to 0.1 |
| Slow responses | Use smaller model (qwen3:32b) |
| Wrong extractions | Add more context to system message |

## Next Steps

1. ✅ **Try the examples** - Run the test script
2. ✅ **Launch the GUI** - See the Gradio interface
3. ✅ **Customize** - Adjust system message for your needs
4. ✅ **Integrate** - Use in your existing pipeline
5. ✅ **Experiment** - Try different models and compare results

## Questions?

- 📖 Read the full documentation: `STOCK_EXTRACTOR_README.md`
- 🧪 Run tests: `python test_stock_extractor.py all`
- 🎨 Launch GUI: `python stock_extractor_gui.py`

---

**Happy Extracting! 🎉**

