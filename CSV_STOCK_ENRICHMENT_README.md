# CSV Stock Name Enrichment Tab

## Overview

A new tab in the Unified GUI that enriches CSV files containing stock codes with their corresponding stock names from the vector store.

## Features

- 📁 **File Upload**: Drag and drop or select CSV files
- 🔍 **Automatic Detection**: Finds the `SCTYCode` column in your CSV
- ➕ **Column Insertion**: Adds a new `stock_name` column right next to `SCTYCode`
- 🔎 **Vector Store Lookup**: Queries the Milvus vector store for accurate stock names
- 💾 **Easy Download**: Generates an enriched CSV file with prefix `enriched_`
- 📊 **Statistics**: Shows processing summary and error logs

## How to Use

1. **Launch the Unified GUI**:
   ```bash
   python unified_gui.py
   ```

2. **Navigate to "CSV Stock Enrichment" Tab**

3. **Upload Your CSV File**:
   - The CSV must contain a column named `SCTYCode`
   - Stock codes can be in various formats (e.g., "700", "00700", "1810")

4. **Click "Enrich CSV with Stock Names"**:
   - The system will process each stock code
   - Progress will be shown during processing

5. **Download the Enriched CSV**:
   - The output file will be named `enriched_<original_filename>.csv`
   - A new column `stock_name` will appear next to `SCTYCode`

## CSV Format Requirements

### Input CSV Example:

```csv
MsgRef,OrderGroup,OrderNo,ACCode,SCTYCode,OrderSide,OrderQty,OrderPrice
10575,INPUT,78246476,P77197,3320,A,10000,4.94
10639,INPUT,78246540,P77197,9698,B,1000,36.52
11284,INPUT,78247195,M57509,981,B,500,91.9
```

### Output CSV Example:

```csv
MsgRef,OrderGroup,OrderNo,ACCode,SCTYCode,stock_name,OrderSide,OrderQty,OrderPrice
10575,INPUT,78246476,P77197,3320,華潤電力,A,10000,4.94
10639,INPUT,78246540,P77197,9698,萬洲國際,B,1000,36.52
11284,INPUT,78247195,M57509,981,中芯國際,B,500,91.9
```

## Features in Detail

### Smart Caching
- Stock codes are cached during processing
- Duplicate stock codes are only looked up once
- Significantly speeds up large CSV files

### Error Handling
- Stock codes not found in vector store are marked as "N/A"
- Processing errors are logged and shown in the Error Log
- Processing continues even if individual lookups fail

### Stock Code Normalization
- Automatically normalizes stock codes (e.g., "700" → "00700")
- Handles various stock code formats
- Compatible with Hong Kong stock code conventions

## Processing Statistics

After processing, you'll see:
- **Total rows processed**: Number of rows in your CSV
- **Unique stock codes found**: Number of distinct stock codes
- **Stock names added**: Successfully enriched entries
- **Errors/Not found**: Codes that couldn't be matched

## Technical Details

### Dependencies
- `stock_verifier_module.stock_verifier_improved`: Stock verification and vector store access
- `gradio`: Web interface
- `csv`: CSV file processing

### Vector Store
- Uses Milvus vector database
- Queries with `verify_and_correct_stock()` function
- Confidence threshold: 0.5
- Returns top 1 match per stock code

### Performance
- Large files (1000+ rows) may take a few minutes
- Processing time depends on:
  - Number of unique stock codes
  - Vector store response time
  - Network latency to Milvus

## Error Handling

Common errors and solutions:

1. **"SCTYCode column not found"**
   - Ensure your CSV has a column named exactly `SCTYCode`
   - Check for spelling and case sensitivity

2. **"Vector store not available"**
   - Check Milvus connection settings
   - Verify vector store is running and accessible

3. **"Could not find stock name for code 'XXXX'"**
   - Stock code doesn't exist in vector store
   - Consider updating your vector store database

## Files Modified

- `tabs/tab_csv_stock_enrichment.py` (new)
- `tabs/__init__.py` (updated)
- `unified_gui.py` (updated)

## Example Usage Scenarios

### Scenario 1: Trade Records
Enrich trade records from `trades.csv` with stock names for easier analysis.

### Scenario 2: Transaction Reports
Add stock names to transaction reports for client-facing documents.

### Scenario 3: Data Analysis
Prepare data for analysis tools that need both stock codes and names.

## Limitations

- Currently only supports CSV files with `SCTYCode` column
- Stock names are retrieved from vector store only (not external APIs)
- Single-threaded processing (sequential lookups)

## Future Enhancements

Potential improvements:
- Support for custom column name mapping
- Batch processing for multiple files
- Parallel vector store queries
- Additional stock information columns (sector, market cap, etc.)
- Support for multiple stock exchanges

## Support

For issues or questions, check the error log in the interface or review the traceback in the console.

