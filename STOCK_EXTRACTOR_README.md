# 📈 Stock Extractor GUI - User Guide

## Overview

`stock_extractor_gui.py` is a specialized tool that uses Large Language Models (LLMs) with Pydantic structured output to extract stock information from conversation transcripts. It's designed to handle Cantonese conversations about Hong Kong stock trading.

## Features

- **🔍 Intelligent Stock Extraction**: Automatically identifies stock codes and names from conversations
- **✅ Structured Output**: Uses Pydantic models to ensure consistent, validated JSON output
- **🛠️ Error Correction**: Can detect and correct Speech-to-Text errors (e.g., "一百一三八" → "18138")
- **📊 Confidence Scoring**: Each extracted stock includes a confidence level (high/medium/low)
- **🔄 Multiple LLM Models**: Choose from various models to compare results
- **🎨 User-Friendly Interface**: Built with Gradio for easy interaction

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

The key dependencies are:
- `gradio>=4.0.0` - For the web interface
- `langchain-ollama>=0.1.0` - For LLM integration
- `langchain-core>=0.1.0` - For output parsing
- `pydantic>=2.0.0` - For data validation

2. **Install Ollama** (if not already installed):
   - Download from: https://ollama.ai
   - Pull the models you want to use:
   ```bash
   ollama pull qwen3:32b
   ollama pull deepseek-r1:32b
   ```

## Usage

### Starting the Application

Run the script:
```bash
python stock_extractor_gui.py
```

The Gradio interface will launch at: `http://localhost:7860`

### Using the Interface

1. **Input Conversation**:
   - Paste or type the conversation transcript in the left text box
   - You can use the example buttons to try pre-loaded conversations

2. **Select Model**:
   - Choose from the dropdown menu (default: `qwen3:32b`)
   - You can also type a custom model name

3. **Adjust Settings**:
   - **Ollama URL**: Default is `http://localhost:11434`
   - **Temperature**: Lower values (0.1-0.3) give more consistent results
   - **System Message**: Customize the instructions to the LLM (optional)

4. **Extract**:
   - Click the "🚀 開始提取股票資訊" button
   - Wait for the LLM to process (may take 5-30 seconds depending on model)

5. **View Results**:
   - **Formatted Results**: Human-readable output with emojis and formatting
   - **Raw JSON Output**: Structured data that can be used programmatically

## Example Output

### Input Conversation:
```
券商：你好，請問需要什麼幫助？
客戶：我想買騰訊
券商：好的，七百號騰訊，買多少？
客戶：一千股，市價買入
```

### Formatted Output:
```
================================================================================
📊 股票提取結果 (Stock Extraction Results)
================================================================================

📝 對話摘要: 客戶詢問購買騰訊股票，券商確認為00700號

🔍 找到 1 個股票:

   1. ✅ 股票 #1
      • 股票代號: 00700
      • 股票名稱: 騰訊
      • 置信度: HIGH
      • 推理: 對話中提到「七百號」，對應香港股票代碼00700

================================================================================
```

### Raw JSON Output:
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
  "summary": "客戶詢問購買騰訊股票，券商確認為00700號"
}
```

## Pydantic Models

The script uses two Pydantic models for structured output:

### StockInfo
```python
class StockInfo(BaseModel):
    stock_number: str      # Stock code (e.g., "00700")
    stock_name: str        # Stock name in Traditional Chinese
    confidence: str        # "high", "medium", or "low"
    reasoning: Optional[str]  # Explanation of identification
```

### ConversationStockExtraction
```python
class ConversationStockExtraction(BaseModel):
    stocks: List[StockInfo]  # All stocks found
    summary: str             # Conversation summary
```

## Supported Models

The script comes pre-configured with these models (you need to have them installed via Ollama):

- `qwen3:32b` - ⭐ Recommended for Cantonese/Chinese
- `deepseek-r1:32b` - Good reasoning capabilities
- `deepseek-r1:70b` - Better quality, slower
- `gpt-oss:20b` - Alternative option
- `gemma3-27b` - Google's model
- `qwen2.5:72b` - Larger Qwen model
- `llama3.3:70b` - Meta's Llama 3.3

You can also enter custom model names if you have other models installed.

## Tips for Best Results

1. **Model Selection**:
   - For Cantonese conversations: Use `qwen3:32b` or `qwen2.5:72b`
   - For reasoning about corrections: Try `deepseek-r1:32b` or `deepseek-r1:70b`

2. **Temperature Settings**:
   - Use low temperature (0.1-0.3) for more consistent, factual extraction
   - Higher temperature (0.5-0.7) for more creative interpretation of ambiguous cases

3. **System Message**:
   - The default system message is optimized for Hong Kong stock trading conversations
   - You can customize it for different contexts or add more examples

4. **Input Quality**:
   - Clearer conversations yield better results
   - Include context (e.g., broker confirmations) to help the LLM understand

## Common Speech-to-Text Errors

The system is designed to correct these common Cantonese STT errors:

| Incorrect | Correct | Example |
|-----------|---------|---------|
| 百 | 八 | 一百一三八 → 18138 |
| 孤 | 沽 (賣出) | 孤出 → 沽出 (sell) |
| 轮 | 窩輪 | 买个轮 → 買個窩輪 (warrant) |

## Programmatic Usage

You can also use the extraction function directly in your code:

```python
from stock_extractor_gui import extract_stocks_from_conversation

conversation = """
券商：你好
客戶：我想買騰訊和小米
"""

result, json_output = extract_stocks_from_conversation(
    conversation_text=conversation,
    model="qwen3:32b",
    ollama_url="http://localhost:11434",
    system_message=DEFAULT_SYSTEM_MESSAGE,
    temperature=0.1,
)

print(result)  # Formatted output
print(json_output)  # JSON string
```

## Troubleshooting

### Issue: "Connection refused" or "Error calling model"
**Solution**: Make sure Ollama is running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Or restart Ollama
ollama serve
```

### Issue: "Model not found"
**Solution**: Pull the model first:
```bash
ollama pull qwen3:32b
```

### Issue: Parsing errors in output
**Solution**: 
- Try lowering the temperature (0.1)
- Some models may not follow JSON format perfectly; the script will show the raw output if parsing fails
- Try a different model (qwen3:32b usually gives good structured output)

### Issue: Wrong stock identification
**Solution**:
- Check if the conversation has enough context
- Try adding more context to the system message
- Use a larger model (e.g., qwen2.5:72b or deepseek-r1:70b)

## Integration with Other Scripts

This script can be integrated with:

1. **Speech-to-Text Pipeline**: Process audio → STT → Stock Extraction
2. **Database Storage**: Save extracted stocks to MongoDB
3. **Trade Verification**: Cross-reference with `search_client_trades.py`
4. **Multi-LLM Comparison**: Use with `tab_multi_llm.py` for consensus

## Performance

- **qwen3:32b**: ~5-10 seconds per request (4-8GB VRAM)
- **deepseek-r1:32b**: ~8-15 seconds per request (4-8GB VRAM)
- **deepseek-r1:70b**: ~15-30 seconds per request (8-16GB VRAM)

Times vary based on your hardware (GPU/CPU), conversation length, and concurrent usage.

## License

This script is part of the test-diarization project. Refer to the main project LICENSE for details.

## Support

For issues or questions:
1. Check the console output for detailed error messages
2. Verify Ollama is running and models are installed
3. Try the example conversations first to ensure setup is correct

---

**Created**: 2025-10-31  
**Version**: 1.0.0

