"""
Transaction Stock Search
Standalone script to search stocks from transaction JSON in Milvus vector store.
Extracts stock codes and names from transaction JSON and searches the vector store.
"""

import gradio as gr
import json
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus

# Milvus configuration
CLUSTER_ENDPOINT = "https://in03-5fb5696b79d8b56.serverless.aws-eu-central-1.cloud.zilliz.com"
TOKEN = "9052d0067c0dd76fc12de51d2cc7a456dcd6caf58e72e344a2c372c85d6f7b486f39d1f2fd15916a7a9234127760e622c3145c36"

# Initialize embeddings
print("Loading embeddings model...")
embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")

# Initialize vector store
print("Connecting to Milvus...")
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="phone_calls",
    connection_args={
        "uri": CLUSTER_ENDPOINT,
        "token": TOKEN,
        "db_name": "stocks"
    },
    consistency_level="Strong",
    drop_old=False,
)
print("Connected successfully!")


def extract_stocks_from_json(json_input):
    """
    Extract stock_code and stock_name from transaction JSON.
    
    Args:
        json_input: JSON string or file content
    
    Returns:
        tuple: (stocks list, error message)
    """
    try:
        data = json.loads(json_input)
        
        if "transactions" not in data:
            return None, "❌ Error: JSON must contain 'transactions' key"
        
        stocks = []
        for transaction in data["transactions"]:
            if "stock_code" in transaction and "stock_name" in transaction:
                stocks.append({
                    "stock_code": transaction["stock_code"],
                    "stock_name": transaction["stock_name"]
                })
        
        return stocks, None
    
    except json.JSONDecodeError as e:
        return None, f"❌ Invalid JSON format: {str(e)}"
    except Exception as e:
        return None, f"❌ Error parsing JSON: {str(e)}"


def search_stock_in_vector_store(stock_query, num_results=3):
    """
    Search the vector store for a specific stock query.
    
    Args:
        stock_query: The search query (stock code or name)
        num_results: Number of results to return
    
    Returns:
        List of search results
    """
    try:
        results = vector_store.similarity_search(stock_query, k=int(num_results))
        
        result_list = []
        for res in results:
            result_list.append({
                "content": res.page_content,
                "metadata": res.metadata if res.metadata else {}
            })
        
        return result_list
    
    except Exception as e:
        return []


def process_transaction_json(json_input, num_results_per_query=3):
    """
    Main function to process transaction JSON and search vector store.
    
    Args:
        json_input: JSON string with transactions
        num_results_per_query: Number of results per stock search
    
    Returns:
        JSON string with search results
    """
    # Validate input
    if not json_input or json_input.strip() == "":
        return json.dumps({
            "status": "error",
            "message": "⚠️ Please provide transaction JSON"
        }, indent=2, ensure_ascii=False)
    
    # Extract stocks from JSON
    stocks, error = extract_stocks_from_json(json_input)
    if error:
        return json.dumps({
            "status": "error",
            "message": error
        }, indent=2, ensure_ascii=False)
    
    if not stocks:
        return json.dumps({
            "status": "error",
            "message": "⚠️ No stocks found in transaction JSON"
        }, indent=2, ensure_ascii=False)
    
    # Search for each stock
    search_results = []
    
    for stock in stocks:
        stock_code = stock["stock_code"]
        stock_name = stock["stock_name"]
        
        # Search by stock code
        code_results = search_stock_in_vector_store(stock_code, num_results_per_query)
        
        # Search by stock name
        name_results = search_stock_in_vector_store(stock_name, num_results_per_query)
        
        search_results.append({
            "stock_code": stock_code,
            "stock_name": stock_name,
            "search_by_code": {
                "query": stock_code,
                "results_count": len(code_results),
                "results": code_results
            },
            "search_by_name": {
                "query": stock_name,
                "results_count": len(name_results),
                "results": name_results
            }
        })
    
    # Build final response
    response = {
        "status": "success",
        "stocks_processed": len(stocks),
        "search_results": search_results
    }
    
    return json.dumps(response, indent=2, ensure_ascii=False)


# Create Gradio interface
with gr.Blocks(title="Transaction Stock Search") as demo:
    gr.Markdown("# 📊 Transaction Stock Search")
    gr.Markdown(
        "Search stocks from transaction JSON in Milvus vector store. "
        "The system extracts stock codes and names, then searches the vector store by both."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Transaction JSON Input")
            
            json_input = gr.Textbox(
                label="Transaction JSON",
                placeholder='''{
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
}''',
                lines=20,
                max_lines=30
            )
            
            num_results = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Results per Query"
            )
            
            search_button = gr.Button("🔎 Search Stocks", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### Search Results (JSON)")
            
            output_box = gr.Textbox(
                label="Results",
                lines=30,
                max_lines=40,
                show_copy_button=True
            )
    
    gr.Markdown("---")
    gr.Markdown(
        """
        ## 📋 How It Works
        
        1. **Extract**: Reads the transaction JSON and extracts `stock_code` and `stock_name` from each transaction
        2. **Search**: For each stock, performs two searches in the Milvus vector store:
           - Search by stock code
           - Search by stock name
        3. **Return**: Returns all results in JSON format with:
           - Status and count of stocks processed
           - For each stock: results from both code and name searches
           - Each result includes content and metadata from the vector store
        
        ## 💡 Tips
        
        - The search uses semantic similarity powered by Milvus and Ollama embeddings
        - Only `stock_code` and `stock_name` fields are required from each transaction
        - Results are returned in JSON format for easy processing
        - Each stock is searched separately to provide detailed results
        """
    )
    
    # Set up the search action
    search_button.click(
        fn=process_transaction_json,
        inputs=[json_input, num_results],
        outputs=output_box
    )

# Launch the app
if __name__ == "__main__":
    print("\nStarting Gradio interface...")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)

