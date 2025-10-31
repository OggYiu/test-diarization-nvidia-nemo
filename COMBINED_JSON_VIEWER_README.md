# Combined JSON Viewer Tab 🔗

## Overview

The Combined JSON Viewer tab is a new feature in the Phone Call Analysis Suite that allows you to combine and view JSON results from multiple analysis tabs in a single unified JSON structure.

## Purpose

When you run multiple analyses (e.g., STT Stock Comparison and Transaction Analysis) on the same audio file or transcription, you may want to:

1. **Compare results side-by-side** - View all JSON outputs in one place
2. **Create unified reports** - Combine multiple analysis outputs into a single JSON file
3. **Export consolidated data** - Save all analysis results together for further processing or archival

## How to Use

### Step 1: Run Your Analyses

First, run your analyses in other tabs and copy their JSON outputs:

- **STT Stock Comparison Tab**: Copy the JSON from the "Structured Data (JSON)" textbox
- **Transaction Analysis Tab**: Copy the JSON from the "Pydantic JSON 輸出" textbox
- Any other tab that produces JSON output

### Step 2: Navigate to Combined JSON Viewer Tab

Go to the "🔗 Combined JSON Viewer" tab in the main interface.

### Step 3: Paste JSON Results

Paste each JSON result into the corresponding input boxes:
- **JSON Input 1**: First analysis result
- **JSON Input 2**: Second analysis result
- **JSON Input 3 (Optional)**: Third analysis result
- **JSON Input 4 (Optional)**: Fourth analysis result

### Step 4: Combine Results

Click the **"🔗 Combine All JSON Results"** button.

The tool will:
- Validate each JSON input
- Automatically identify the type of analysis (Stock Extraction, Transaction Analysis, etc.)
- Combine all valid JSON into a unified structure
- Display a summary with metadata and source information

### Step 5: View and Save

- **Summary & Status**: Shows a high-level overview of combined results
- **Combined JSON Output**: Shows the complete unified JSON structure
- **Save to File**: Optionally save the combined JSON to a file

## Output Structure

The combined JSON follows this structure:

```json
{
  "metadata": {
    "timestamp": "2025-10-31T12:34:56.789012",
    "total_sources": 2,
    "sources": ["Input 1", "Input 2"]
  },
  "results": [
    {
      "source_id": "source_1",
      "source_index": 1,
      "original_index": 1,
      "data": {
        // Original JSON from first analysis
      }
    },
    {
      "source_id": "source_2",
      "source_index": 2,
      "original_index": 2,
      "data": {
        // Original JSON from second analysis
      }
    }
  ]
}
```

## Features

### 📦 Unified View
- Combines multiple JSON results into a single structured document
- Maintains original data integrity
- Adds metadata for tracking and organization

### 🔍 Automatic Type Detection
- Automatically identifies JSON types:
  - Stock Extraction (detects "stocks" field)
  - Transaction Analysis (detects "transactions" field)
  - General Analysis (detects "summary" field)
  - Custom Data (any other JSON structure)

### ⚠️ Error Handling
- Validates each JSON input
- Provides clear error messages for invalid JSON
- Continues processing valid inputs even if some are invalid

### 💾 File Export
- Save combined results as a JSON file
- Automatic filename generation with timestamp
- Customizable filename

### 🕐 Metadata
- Automatic timestamp generation
- Source tracking
- Total source count

### 📊 Summary Display
- Clear overview of all combined sources
- Type identification for each source
- Quick statistics (number of stocks, transactions, etc.)

## Example Use Cases

### Use Case 1: Stock Extraction + Transaction Analysis

You transcribe an audio file using two different STT models and want to:
1. Extract stock information from both transcriptions (STT Stock Comparison)
2. Analyze transactions in both transcriptions (Transaction Analysis)
3. Combine both JSON results to see the complete picture

**Workflow:**
1. Run STT Stock Comparison → Copy JSON output
2. Run Transaction Analysis → Copy JSON output
3. Paste both in Combined JSON Viewer
4. Click "Combine All JSON Results"
5. View unified report showing both stock extraction and transaction analysis

### Use Case 2: Multiple Transaction Analyses

You have multiple audio files from different phone calls and want to:
1. Analyze each call separately (Transaction Analysis for each)
2. Combine all transaction analyses into one JSON file
3. Export for batch processing or reporting

**Workflow:**
1. Analyze Call 1 → Copy JSON
2. Analyze Call 2 → Copy JSON
3. Analyze Call 3 → Copy JSON
4. Paste all three in Combined JSON Viewer
5. Save as single consolidated JSON file

### Use Case 3: Cross-Validation

You want to cross-validate results from multiple LLM models:
1. Run the same analysis with different LLM models
2. Combine all JSON results
3. Compare how different models identified stocks or transactions

## Examples

The tab includes two built-in examples:

### Example 1: Stock Extraction + Transaction
Demonstrates combining:
- Stock extraction result (stocks found)
- Transaction analysis result (buy/sell/queue transactions)

### Example 2: Multiple Transaction Analyses
Demonstrates combining:
- Three separate transaction analyses
- Different transaction types (buy, sell, queue)
- Various confidence scores

## Tips

1. **Always validate JSON first**: Make sure the JSON is valid before pasting (most analysis tabs provide valid JSON automatically)

2. **Use descriptive filenames**: When saving, use meaningful filenames like `customer_123_combined_analysis.json`

3. **Check the summary**: Review the summary to ensure all sources were processed correctly

4. **Empty inputs are OK**: You can leave input boxes 3 and 4 empty if you only have 2 results to combine

5. **Copy button available**: All textboxes have a copy button for easy copying of results

## Limitations

- Maximum 4 JSON inputs per combination
- Each input must be valid JSON or will be marked as error
- Large JSON files may take longer to process

## Future Enhancements

Potential future features:
- Support for more than 4 inputs
- JSON comparison and diff view
- Filtering and searching within combined results
- Export to other formats (CSV, Excel)
- Visualization of combined data

## Technical Details

- **Technology**: Python 3.x, Gradio
- **JSON Processing**: Standard `json` module
- **Error Handling**: Comprehensive try-catch blocks
- **File Encoding**: UTF-8 for proper Chinese character support

## Questions or Issues?

If you encounter any issues with the Combined JSON Viewer tab, please check:

1. Is your JSON valid? (Use a JSON validator online)
2. Are you pasting into the correct input boxes?
3. Check the Summary & Status for error messages
4. Ensure file write permissions if saving fails

---

**Created**: October 2025  
**Version**: 1.0  
**Part of**: Phone Call Analysis Suite

