# STT Transcription Merger Tool

## Overview
This tool uses Large Language Models (LLM) to analyze and merge transcriptions from two different Speech-to-Text (STT) models, creating an improved and more accurate transcription by combining the strengths of each model.

## Features

✨ **Dual Transcription Input**: Input transcriptions from two different STT models
🤖 **LLM-Powered Analysis**: Uses advanced language models to intelligently merge transcriptions
📊 **Detailed Analysis**: Provides explanation of improvements and decision-making process
🎯 **Multiple Templates**: Pre-built system prompts for different use cases
🔧 **Customizable**: Adjust LLM parameters and system prompts

## Why Merge Transcriptions?

Different STT models have different strengths and weaknesses:
- One model might be better at recognizing numbers
- Another might handle accents or dialects better
- Some excel at domain-specific terminology
- Others are better with general conversation

By combining two transcriptions, you can leverage the best of both models to create a more accurate final result.

## Installation

### Prerequisites
1. Python 3.8+
2. Ollama installed and running (https://ollama.ai)
3. Required Python packages (see requirements.txt)

### Setup
```bash
# Install dependencies (if not already installed)
pip install gradio langchain-ollama

# Make sure Ollama is running
# Download a model if you haven't already
ollama pull qwen3:32b
```

## Usage

### Starting the Tool

```bash
python transcription_merger.py
```

The tool will start a web interface at `http://localhost:7861`

### Step-by-Step Guide

1. **Prepare Your Transcriptions**
   - Get transcriptions of the same audio from two different STT models
   - For example: Whisper + SenseVoice, or WSYue + FunASR

2. **Input Transcriptions**
   - Label each transcription (e.g., "Whisper v3", "SenseVoice")
   - Paste the first transcription in the first text box
   - Paste the second transcription in the second text box

3. **Configure LLM Settings**
   - Select an LLM model (recommended: qwen3:32b or larger)
   - Set the Ollama URL (default: http://localhost:11434)
   - Adjust temperature:
     - **0.1-0.3**: Conservative, stays close to original
     - **0.4-0.7**: Balanced (recommended)
     - **0.8-1.0**: More creative, may deviate more

4. **Choose System Prompt Template**
   - **通用粵語轉錄 (General Cantonese)**: For general conversations
   - **香港股市交易對話 (HK Stock Trading)**: For stock trading phone calls
   - **自訂 (Custom)**: Write your own prompt

5. **Start Analysis**
   - Click the "🚀 開始分析並合併" button
   - Wait for the LLM to analyze (usually 5-30 seconds depending on length and model)
   - Review the merged transcription and analysis

## System Prompt Templates

### General Cantonese Transcription
Best for everyday Cantonese conversations. The LLM will:
- Compare both versions
- Identify differences and strengths
- Merge the best parts from each
- Fix common STT errors (homophones, numbers, proper nouns)

### Hong Kong Stock Trading
Specialized for phone recordings of stock trading conversations. The LLM will:
- Focus on stock codes, prices, quantities
- Recognize trading terminology (窩輪, 沽, etc.)
- Fix common financial number errors
- Identify broker-client confirmation patterns

## Example Use Cases

### Use Case 1: Cantonese Conversation
```
Model 1 (Whisper): "我想買一百股騰訊"
Model 2 (WSYue): "我想買一八股騰訊"

Merged Output: "我想買一八股騰訊"
Analysis: Model 2 correctly identified "一八" (18) instead of "一百" (100),
which is a common STT error in Cantonese where "八" and "百" sound similar.
```

### Use Case 2: Stock Trading
```
Model 1: "買入建行代號九三九"
Model 2: "買入建行代號九三零"

Merged Output: "買入建行代號939" (0939)
Analysis: Combined context - 建行 (CCB) has stock code 0939. Model 1 was
closer with "九三九" but the correct code is 0939 (九三零九 in full form).
```

## Tips for Best Results

1. **Use Larger Models**: 32B or 70B models provide better analysis than smaller ones
2. **Lower Temperature for Accuracy**: Use 0.2-0.4 for factual content like trading conversations
3. **Clear Labels**: Give descriptive labels to help the LLM understand model characteristics
4. **Custom Prompts**: For specialized domains, create custom system prompts with relevant context
5. **Review Output**: Always review the merged result, especially for critical information

## Troubleshooting

### "Connection refused" error
- Make sure Ollama is running: `ollama serve`
- Check the Ollama URL is correct

### "Model not found" error
- Pull the model first: `ollama pull qwen3:32b`
- Or use a model you already have: `ollama list`

### Slow response
- Larger models take longer
- Longer transcriptions take more time
- Consider using a smaller model like `qwen3:8b` for testing

### Poor merge quality
- Try a larger model
- Adjust the system prompt to be more specific
- Lower the temperature for more conservative merging
- Make sure both transcriptions are of reasonable quality

## Technical Details

### LLM Integration
- Uses LangChain with Ollama for flexible model selection
- Supports any Ollama-compatible model
- Temperature control for generation behavior

### Architecture
- **Frontend**: Gradio web interface
- **Backend**: Python with LangChain
- **LLM**: Ollama (local inference)

### Port Configuration
- Default port: 7861
- Change in code if needed (to avoid conflicts with unified_gui.py which uses 7860)

## Related Tools

This tool is part of a larger phone call analysis suite:
- `unified_gui.py`: Main interface with all tools
- `tabs/tab_stt.py`: Speech-to-text transcription
- `tabs/tab_llm_analysis.py`: Single transcription analysis
- `tabs/tab_llm_comparison.py`: Compare multiple LLM analyses

## License

Same as the parent project.

## Support

If you encounter issues:
1. Check that Ollama is running
2. Verify the model is downloaded
3. Review the console output for error messages
4. Check the Gradio interface for error details

