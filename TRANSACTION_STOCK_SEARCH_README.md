# Transaction Stock Search

A tool for searching stocks from transaction JSON in Milvus vector store. This tool extracts stock codes and names from transaction JSON files and performs semantic searches in the vector store.

## Features

- **JSON Parsing**: Extracts `stock_code` and `stock_name` from transaction JSON
- **Dual Search**: Searches the vector store by both stock code AND stock name
- **Semantic Search**: Uses Milvus and Ollama embeddings for intelligent similarity search
- **JSON Output**: Returns structured JSON results with all search findings

## Usage

### 1. Standalone Script

Run the standalone version:

```bash
python transaction_stock_search.py
```

This will launch a Gradio interface on `http://127.0.0.1:7861`

### 2. Unified GUI Tab

The feature is also available as a tab in the unified GUI:

```bash
python unified_gui.py
```

Look for the "📊 Transaction Stock Search" tab.

## Input Format

The tool expects a JSON file with the following structure:

```json
{
  "transactions": [
    {
      "transaction_type": "buy",
      "confidence_score": 2.0,
      "stock_code": "18138",
      "stock_name": "騰訊認購證",
      "quantity": "20000",
      "price": "0.38",
      "explanation": "..."
    },
    {
      "transaction_type": "queue",
      "confidence_score": 1.0,
      "stock_code": "00020",
      "stock_name": "金碟科技",
      "quantity": "15",
      "price": "0.72",
      "explanation": "..."
    }
  ]
}
```

**Required fields per transaction:**
- `stock_code`: The stock code/ticker symbol
- `stock_name`: The stock name

**Optional fields** (not used by this tool):
- `transaction_type`
- `confidence_score`
- `quantity`
- `price`
- `explanation`
- Any other custom fields

## Output Format

The tool returns a JSON response with the following structure:

```json
{
  "status": "success",
  "stocks_processed": 3,
  "search_results": [
    {
      "stock_code": "18138",
      "stock_name": "騰訊認購證",
      "search_by_code": {
        "query": "18138",
        "results_count": 3,
        "results": [
          {
            "content": "...",
            "metadata": {...}
          }
        ]
      },
      "search_by_name": {
        "query": "騰訊認購證",
        "results_count": 3,
        "results": [
          {
            "content": "...",
            "metadata": {...}
          }
        ]
      }
    }
  ]
}
```

### Output Fields

- **status**: Either "success" or "error"
- **stocks_processed**: Number of stocks extracted from the JSON
- **search_results**: Array of results for each stock
  - **stock_code**: The stock code searched
  - **stock_name**: The stock name searched
  - **search_by_code**: Results when searching by stock code
    - **query**: The actual search query used
    - **results_count**: Number of results found
    - **results**: Array of result objects with `content` and `metadata`
  - **search_by_name**: Results when searching by stock name
    - Same structure as `search_by_code`

## How It Works

1. **Parse JSON**: Reads the transaction JSON and extracts stock information
2. **Extract Stocks**: Creates an array of `{stock_code, stock_name}` objects
3. **Search by Code**: For each stock, searches the Milvus vector store using the stock code
4. **Search by Name**: For each stock, searches the Milvus vector store using the stock name
5. **Return Results**: Combines all results into a structured JSON response

## Configuration

The tool connects to Milvus using the following configuration:

- **Endpoint**: Serverless AWS EU-Central-1 cluster
- **Collection**: "phone_calls"
- **Database**: "stocks"
- **Embeddings**: Ollama `qwen3-embedding:8b` model
- **Consistency**: Strong

## Example

### Input

Use the provided `sample_transaction.json`:

```bash
cat sample_transaction.json
```

### Process

1. Paste the JSON into the "Transaction JSON" textbox
2. Adjust "Results per Query" slider (default: 3)
3. Click "🔎 Search Stocks"
4. View the JSON results in the output box

### Output

You'll receive JSON results showing:
- Matches for each stock code
- Matches for each stock name
- Full content and metadata from the vector store
- Separate results for code vs name searches

## Tips

- **Semantic Search**: The tool uses semantic similarity, so it finds conceptually similar results, not just exact matches
- **Adjustable Results**: Use the slider to get more or fewer results per query
- **Copy Results**: Use the copy button on the output box to easily copy the JSON results
- **Both Searches**: Each stock is searched twice (by code and by name) to ensure comprehensive results

## Files

- **transaction_stock_search.py**: Standalone script
- **tabs/tab_transaction_stock_search.py**: Tab module for unified GUI
- **sample_transaction.json**: Sample input file for testing
- **TRANSACTION_STOCK_SEARCH_README.md**: This documentation

## Dependencies

- `gradio`: Web interface
- `langchain-ollama`: Ollama embeddings integration
- `langchain-milvus`: Milvus vector store integration
- `json`: JSON parsing (built-in)

## Error Handling

The tool handles various error conditions:

- **Invalid JSON**: Returns error message if JSON is malformed
- **Missing Transactions Key**: Validates that the JSON has a "transactions" array
- **No Stocks Found**: Returns error if no valid stock data is extracted
- **Empty Input**: Validates that input is not empty
- **Search Errors**: Catches and reports any errors during vector store searches

## Integration

The tool is fully integrated into the Phone Call Analysis Suite:

1. Added to `tabs/__init__.py`
2. Added to `unified_gui.py`
3. Available as Tab 13 in the unified interface
4. Can be run standalone or as part of the suite

