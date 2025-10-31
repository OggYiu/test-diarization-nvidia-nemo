# Transaction Analysis Tab

## Overview
The **Transaction Analysis** tab is a new feature in the Phone Call Analysis Suite that compares two different STT (Speech-to-Text) transcriptions and identifies stock transactions using structured Pydantic models.

## Features

### Input Fields
1. **Transcription 1 (STT Model 1)**: First transcription text from an STT model
2. **Transcription 2 (STT Model 2)**: Second transcription text from another STT model
3. **Stock References**: Optional field to provide possible stock names and codes mentioned in the conversation

### Output Fields
The analysis provides structured output with the following fields:

1. **Transaction Type**: One of:
   - `buy` - Buy transaction identified
   - `sell` - Sell transaction identified  
   - `queue` - Queue/pending transaction identified
   - `none` - No transaction found

2. **Confidence Score**: A numerical score from 0.0 to 2.0:
   - `0.0` - Not sure at all / No transaction
   - `1.0` - Moderately confident
   - `2.0` - Very confident

3. **Stock Code**: The stock code/number identified (e.g., "0700")

4. **Stock Name**: The stock name identified (e.g., "騰訊")

5. **Quantity**: The quantity/amount of stocks in the transaction

6. **Price**: The price mentioned in the transaction

7. **Explanation**: Detailed explanation of why this transaction type and confidence score were assigned

8. **Transcription Comparison**: Analysis of how the two transcriptions differ and which one is more reliable

## How It Works

### Pydantic-Based Structured Output
The tab uses Pydantic models to ensure structured, type-safe output from the LLM. The `TransactionAnalysis` model defines the exact schema expected from the analysis.

### LLM Integration
- Uses Ollama-compatible models (default: qwen3:32b)
- Requests JSON-formatted responses for structured output
- Low temperature (0.3) by default for more deterministic results

### Analysis Process
1. Compares both transcriptions to identify discrepancies
2. Looks for confirmation phrases that indicate actual transactions
3. Extracts transaction details (type, stock info, quantity, price)
4. Assigns a confidence score based on evidence strength
5. Provides detailed explanations and comparisons

## Usage Example

### Input
**Transcription 1 (STT Model 1):**
```
券商：你好，請問要買咩？
客戶：我想買騰訊
券商：好，騰訊零七零零，買幾多？
客戶：買一千股
券商：好，確認一千股騰訊零七零零，市價買入，對嗎？
客戶：對
```

**Transcription 2 (STT Model 2):**
```
券商：你好，請問要買咩？
客戶：我想買騰訊
券商：好，騰訊零百零零，買幾多？
客戶：買一千股
券商：好，確認一千股騰訊零百零零，市價買入，對嗎？
客戶：對
```

**Stock References:**
```
騰訊 0700
阿里巴巴 9988
```

### Output
- **Transaction Type**: `buy`
- **Confidence Score**: `2.0`
- **Stock Code**: `0700`
- **Stock Name**: `騰訊`
- **Quantity**: `1000股`
- **Price**: `市價`
- **Explanation**: "對話中有明確的買入確認流程，券商重複了下單資料讓客戶確認，客戶也明確回答'對'，因此置信度為2.0（非常確定）"
- **Comparison**: "兩個轉錄文字非常相似，但轉錄2將'零七零零'誤認為'零百零零'，這是常見的STT誤差（'八'被誤認為'百'）"

## System Prompt
The tab includes a carefully crafted system prompt that:
- Understands Cantonese stock trading terminology
- Recognizes common STT errors (e.g., "百" mistaken for "八")
- Identifies transaction confirmation patterns
- Provides structured analysis with confidence scoring

## Configuration
- **Default Model**: qwen3:32b
- **Default Temperature**: 0.3 (lower for more consistent results)
- **Default Ollama URL**: http://localhost:11434
- **Output Format**: JSON (structured via Pydantic)

## Technical Details

### Dependencies
- `gradio` - UI framework
- `pydantic` - Data validation and structured output
- `langchain_ollama` - LLM integration

### File Location
- Implementation: `tabs/tab_transaction_analysis.py`
- Imported in: `tabs/__init__.py`
- Used in: `unified_gui.py`

## Common Use Cases

1. **Comparing STT Models**: Identify which STT model produces more accurate transcriptions for Cantonese financial conversations

2. **Transaction Verification**: Verify if a phone conversation actually resulted in a transaction or was just a discussion

3. **Confidence Assessment**: Understand how certain the LLM is about the identified transaction

4. **Handling STT Errors**: Automatically correct common Cantonese STT errors (e.g., homophone confusion)

## Tips for Best Results

1. **Provide Stock References**: Adding known stock names/codes helps the LLM identify them more accurately

2. **Use Both Transcriptions**: Even if one transcription seems good, having two helps identify errors

3. **Adjust Temperature**: Lower temperature (0.1-0.3) for more consistent results, higher (0.5-0.7) for more creative interpretation

4. **Review Explanations**: Always read the explanation to understand why the LLM made its decision

5. **Model Selection**: Larger models (70b) may provide more nuanced analysis but are slower

## Limitations

- Requires a running Ollama instance
- Model must support JSON output format
- Analysis quality depends on the selected LLM model
- May struggle with very noisy or incomplete transcriptions

## Future Enhancements

Potential improvements:
- Support for multiple transactions in one conversation
- Historical transaction pattern learning
- Integration with stock database for validation
- Multi-language support beyond Cantonese
- Confidence score calibration based on feedback

