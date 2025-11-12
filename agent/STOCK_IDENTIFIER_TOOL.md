# Stock Identifier Tool

## Overview
The Stock Identifier Tool uses LLM analysis to identify stocks mentioned in conversation transcripts. It can extract stock names, symbols, quantities, prices, and the context in which they were discussed.

## Tool Function
- **Name**: `identify_stocks_in_conversation`
- **Input**: Conversation text (string)
- **Output**: Formatted analysis with identified stocks

## Features
The tool identifies:
- **Stock names** (e.g., "Tencent", "HSBC")
- **Stock symbols/codes** (e.g., "0700", "0005")
- **Context** (buy, sell, price discussion, etc.)
- **Quantities** (number of shares mentioned)
- **Prices** (price per share mentioned)

## Usage

### As a Standalone Tool
```python
from tools.stock_identifier_tool import identify_stocks_in_conversation

conversation = """
Broker: What would you like to trade today?
Client: I want to buy 1000 shares of Tencent at HK$350.
Broker: Okay, placing order for 1000 shares of 0700 at HK$350.
"""

result = identify_stocks_in_conversation(conversation)
print(result)
```

### With the Agent
The tool is automatically available to the agent when it's included in the tools list. You can ask the agent to analyze conversation text:

```python
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Analyze this conversation and identify the stocks: [conversation text here]"
        }
    ]
})
```

### In the Full Pipeline
The tool can be used after transcription and correction:

```python
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Process audio file 'conversation.wav': extract metadata, diarize with 2 speakers, chop with 50ms padding, transcribe with SenseVoiceSmall, apply Cantonese corrections, and then identify all stocks discussed."
        }
    ]
})
```

## Example Output

```
Identified 2 stock(s) in the conversation:

Stock 1:
  Name: Tencent Holdings
  Symbol/Code: 0700
  Context: Client wants to buy shares
  Quantity: 1000 shares
  Price: HK$350 per share

Stock 2:
  Name: HSBC
  Symbol/Code: 0005
  Context: Price inquiry
  Quantity: N/A
  Price: HK$62

Summary: The conversation involves buying Tencent shares and inquiring about HSBC price.
```

## Technical Details

### LLM Configuration
The tool uses the same LLM configuration as the main agent:
- **Model**: qwen3:8b (via Ollama)
- **Temperature**: 0.0 (for consistent results)
- **Base URL**: http://localhost:11434/v1

### Response Format
The tool returns structured analysis in the following format:
- Human-readable formatted text (for agent display)
- Internal JSON structure with stocks array and summary

### Error Handling
The tool handles:
- JSON parsing errors from LLM responses
- Missing or incomplete stock information
- Conversations with no stocks mentioned

## Testing
Run the test script to see examples:
```bash
cd agent
python test_stock_identifier.py
```

## Integration with Existing Tools
The stock identifier tool complements the existing audio processing pipeline:

1. **extract_metadata_from_filename** → Extract file metadata
2. **diarize_audio** → Identify speakers
3. **chop_audio_by_rttm** → Split audio by speaker
4. **transcribe_audio_segments** → Convert speech to text
5. **correct_transcriptions** → Apply Cantonese corrections
6. **identify_stocks_in_conversation** → Analyze and identify stocks ✨ NEW

## Notes
- The tool requires Ollama to be running with the qwen3:8b model loaded
- Analysis quality depends on the clarity and completeness of the conversation text
- Works best with properly transcribed and corrected text
- Can handle both English and Chinese (Cantonese/Mandarin) conversations

