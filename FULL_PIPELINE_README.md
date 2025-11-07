# Full Pipeline Tab - Complete Documentation

## Overview

The **Full Pipeline** tab is a new feature that chains all existing analysis tabs into one seamless workflow. Instead of manually running each tab separately and copying JSON outputs between tabs, this tab automates the entire process from audio input to final compliance report.

## Pipeline Flow

The Full Pipeline executes 6 steps sequentially:

```
Audio File
    ↓
1. Speech-to-Text (STT)
    ↓ (Conversation JSON)
2. JSON Batch Analysis (Stock Extraction)
    ↓ (Merged Stocks JSON)
3. Transaction Analysis
    ↓ (Transaction JSON)
4. Trade Verification
    ↓ (Verification JSON)
5. Conversation Record Analysis
    ↓ (Analysis JSON)
6. Compliance Analysis
    ↓
Final Compliance Report
```

## How to Use

### Step 1: Prepare Your Files

Before running the pipeline, make sure you have:

1. **Audio file(s)**: WAV, MP3, FLAC, M4A, OGG, or OPUS format
2. **trades.csv**: Trade records file in the correct format (required for verification)
3. **Ollama server**: Running with the required models

### Step 2: Launch the Interface

```bash
python unified_gui.py
```

The application will start on `http://localhost:7860`

### Step 3: Navigate to Full Pipeline Tab

The **🔗 Full Pipeline** tab is the first tab in the interface.

### Step 4: Configure Settings

#### 📝 Step 1: Speech-to-Text Settings

- **Audio File(s) or Folder**: Upload your audio file(s)
- **Use SenseVoice**: Recommended for Cantonese (checked by default)
- **Use Whisper v3 Cantonese**: Optional additional model
- **Language**: Auto-detect or select specific language (auto/zh/yue/en)
- **Advanced STT Settings** (optional):
  - Overwrite Diarization Cache
  - Padding (ms): Default 100ms
  - Use Enhanced Format: Recommended (checked)
  - Apply Text Corrections: Optional
  - VAD Model: fsmn-vad (recommended)
  - Max Single Segment Time: 30000ms

#### 🔍 Step 2: Stock Extraction Settings

- **Select LLMs for Stock Extraction**: Choose one or more models
  - Default: qwen2.5:32b (or first available)
  - Can select multiple for consensus
- **Advanced Stock Extraction Settings** (optional):
  - System Message: Customizable extraction prompt
  - Use Vector Correction: Recommended (checked)
  - Use Contextual Analysis: Recommended (checked)
  - Enable Stock Verification: Optional
  - Verification LLM: Model for verification

#### 💰 Step 3: Transaction Analysis Settings

- **Transaction Analysis Model**: LLM for transaction identification
  - Default: qwen2.5:32b (or first available)
- **Advanced Transaction Settings** (optional):
  - System Message: Customizable analysis prompt
  - Temperature: 0.3 (default, lower = more deterministic)

#### 🔎 Step 4: Trade Verification Settings

- **Trades CSV File Path**: Path to your trades.csv file
  - Default: `trades.csv`
- **Time Window (hours)**: Search window for matching trades
  - Default: 1.0 hour (±1 hour from conversation time)
  - Range: 0.1 to 24.0 hours

#### 🎯 Step 5: Record Analysis Settings

- **Client ID Filter**: Optional filter for specific client
  - Leave empty to use client ID from conversation metadata
- **Record Analysis Model**: LLM for analyzing records
  - Default: qwen2.5:32b (or first available)
- **Advanced Record Analysis Settings** (optional):
  - Temperature: 0.3 (default)
  - Use Combined Analysis: Recommended for multiple conversations (checked)

#### 🛡️ Step 6: Compliance Settings

- **Use LLM for Compliance Analysis**: Enable AI-powered analysis (recommended)
- **Compliance Analysis Model**: LLM for compliance assessment
  - Default: qwen2.5:32b (or first available)

#### ⚙️ Global Settings

- **Ollama URL**: URL of your Ollama server
  - Default: `http://localhost:11434` (configured in model_config.py)
- **Default Temperature**: Default temperature for LLM calls
  - Default: 0.3

### Step 5: Run the Pipeline

Click the **🚀 Run Full Pipeline** button to start the automated analysis.

### Step 6: Monitor Progress

The **Pipeline Status Log** will show real-time progress:

```
================================================================================
🚀 STARTING FULL PIPELINE
================================================================================

📝 STEP 1/6: Speech-to-Text Processing
--------------------------------------------------------------------------------
✅ STT completed successfully
   Generated conversation JSON (12345 chars)

🔍 STEP 2/6: JSON Batch Analysis (Stock Extraction)
--------------------------------------------------------------------------------
✅ JSON Batch Analysis completed successfully
   Generated merged stocks JSON (2345 chars)

💰 STEP 3/6: Transaction Analysis
--------------------------------------------------------------------------------
✅ Transaction Analysis completed successfully
   Generated transaction JSON (3456 chars)

🔎 STEP 4/6: Trade Verification
--------------------------------------------------------------------------------
✅ Trade Verification completed successfully
   Generated verification JSON (4567 chars)

🎯 STEP 5/6: Conversation Record Analysis
--------------------------------------------------------------------------------
✅ Conversation Record Analysis completed successfully
   Generated analysis JSON (5678 chars)

🛡️ STEP 6/6: Compliance Analysis (Final)
--------------------------------------------------------------------------------
✅ Compliance Analysis completed successfully
   Generated compliance report (6789 chars)

================================================================================
✅ FULL PIPELINE COMPLETED SUCCESSFULLY
================================================================================

📊 All 6 steps completed:
   1. ✅ Speech-to-Text
   2. ✅ JSON Batch Analysis (Stock Extraction)
   3. ✅ Transaction Analysis
   4. ✅ Trade Verification
   5. ✅ Conversation Record Analysis
   6. ✅ Compliance Analysis

🎉 Final compliance report generated successfully!
```

## Results

After the pipeline completes, you can view results in multiple tabs:

### 🎉 Final Results Tabs

1. **Compliance Report**: The final compliance analysis report
   - Overall confidence score
   - Human review recommendations
   - Compliance assessment
   - Risk factors and recommendations

2. **Conversation JSON** (Step 1 output)
   - Full conversation transcription
   - Speaker identification
   - Timestamps
   - Metadata

3. **Stocks JSON** (Step 2 output)
   - Extracted stock information
   - Stock codes and names
   - Confidence scores
   - Detection metadata

4. **Transactions JSON** (Step 3 output)
   - Identified transactions
   - Transaction types (buy/sell/queue)
   - Stock details
   - Confidence scores

5. **Verification JSON** (Step 4 output)
   - Trade record matches
   - Match confidence
   - Detailed comparisons
   - Discrepancies

6. **Analysis JSON** (Step 5 output)
   - Record-by-record analysis
   - Confidence that trades were discussed
   - Evidence from conversation
   - Overall assessment

## Error Handling

If any step fails:
- The pipeline will stop at that step
- The error will be displayed in the Pipeline Status Log
- Partial results up to that step will still be available
- You can review the error and adjust settings as needed

Common errors:
- **No audio file provided**: Upload an audio file
- **No STT model selected**: Enable at least one STT model
- **Trades file not found**: Verify trades.csv path
- **Ollama connection error**: Ensure Ollama server is running
- **Model not found**: Pull required models in Ollama

## Tips for Best Results

1. **Audio Quality**: Use clear, high-quality audio files for best transcription
2. **Multiple Models**: Use multiple LLMs for stock extraction to get consensus
3. **Time Window**: Adjust based on when calls typically happen relative to trades
4. **Combined Analysis**: Enable for related conversations from same client/day
5. **LLM Selection**: Use larger models (32B+) for better analysis quality

## Comparison with Manual Workflow

### Before (Manual Workflow)
1. Run STT tab → Copy conversation JSON
2. Paste into JSON Batch Analysis → Copy merged stocks JSON
3. Paste conversation + stocks into Transaction Analysis → Copy transaction JSON
4. Paste transaction into Trade Verification → Copy verification JSON
5. Paste conversation into Conversation Record Analysis → Copy analysis JSON
6. Paste verification + analysis into Compliance Analysis → Get final report

**Time**: 5-10 minutes of manual copying/pasting

### Now (Automated Pipeline)
1. Upload audio file
2. Configure settings (one time)
3. Click "Run Full Pipeline"
4. Get final compliance report

**Time**: 1 click, automatic execution

## CSV Output Files

The pipeline automatically saves results to CSV files:

- **report.csv**: Trade verification results
- **verify.csv**: Conversation record analysis results
- **compliance.csv**: Final compliance analysis

These files are saved in the project root directory.

## Integration with Existing Tabs

The Full Pipeline tab doesn't replace the individual tabs - it complements them:

- Use **individual tabs** when you need fine-grained control or want to analyze specific steps
- Use **Full Pipeline** when you want end-to-end automated analysis

You can still access all individual tabs to:
- Inspect intermediate results in detail
- Re-run specific steps with different parameters
- Use manual chaining with custom inputs

## Technical Details

### Dependencies

The Full Pipeline imports and uses processing functions from:
- `tabs.tab_stt.process_audio_or_folder`
- `tabs.tab_json_batch_analysis.process_json_batch`
- `tabs.tab_transaction_analysis_json.analyze_transactions_with_json`
- `tabs.tab_trade_verification.verify_transactions`
- `tabs.tab_conversation_record_analysis.analyze_conversation_records`
- `tabs.tab_compliance_analysis.analyze_compliance`

### State Management

Unlike the individual tabs that use Gradio state components for chaining, the Full Pipeline:
- Passes outputs directly between functions
- Stores intermediate results in variables
- Returns all intermediate JSONs for inspection

This makes it more transparent and easier to debug.

### Progress Tracking

The pipeline uses Gradio's `progress` parameter to show:
- Current step (1/6, 2/6, etc.)
- Step description
- Progress percentage (0%, 17%, 34%, 51%, 68%, 85%, 100%)

## Troubleshooting

### Pipeline stops at Step 1
- Check audio file format and quality
- Verify STT models are loaded correctly
- Check model_cache directory

### Pipeline stops at Step 2, 3, 5, or 6
- Verify Ollama server is running: `http://localhost:11434`
- Check if required models are pulled
- Review Ollama logs for errors

### Pipeline stops at Step 4
- Verify trades.csv exists and is readable
- Check CSV format matches expected schema
- Ensure client ID in conversation matches ACCode in trades.csv

### Out of Memory
- Process fewer audio files at once
- Use smaller models
- Increase system RAM or use GPU

### Slow Performance
- Use GPU acceleration if available
- Use smaller/faster models
- Reduce number of LLMs for stock extraction
- Increase max_single_segment_time to create fewer segments

## Future Enhancements

Potential improvements for future versions:
- Batch processing of multiple audio files
- Parallel processing where possible
- Resume from failed step
- Save/load pipeline configurations
- Export all results as a single report
- Email/notification when pipeline completes
- Scheduled/automated pipeline runs

## Support

For issues or questions:
1. Check this README
2. Review individual tab documentation
3. Check logs in `error.log` and `onelogger.log`
4. Verify all dependencies are installed

## Files Created by This Feature

- `tabs/tab_full_pipeline.py`: Main implementation
- `FULL_PIPELINE_README.md`: This documentation
- Updated `tabs/__init__.py`: Export new function
- Updated `unified_gui.py`: Include new tab

---

**Version**: 1.0  
**Last Updated**: 2025-11-07  
**Author**: AI Assistant

