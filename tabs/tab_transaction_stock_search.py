"""
Tab 13: Transaction Stock Search
Search stocks from transaction JSON in Milvus vector store.
Extracts stock codes and names from transaction JSON and searches the vector store.
"""

import gradio as gr
import json
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus

# Milvus configuration
CLUSTER_ENDPOINT = "https://in03-5fb5696b79d8b56.serverless.aws-eu-central-1.cloud.zilliz.com"
TOKEN = "9052d0067c0dd76fc12de51d2cc7a456dcd6caf58e72e344a2c372c85d6f7b486f39d1f2fd15916a7a9234127760e622c3145c36"

# Global variable to store vector store instance
_vector_store = None


def initialize_vector_store():
    """
    Initialize the Milvus vector store with embeddings.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    global _vector_store
    
    try:
        embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
        
        _vector_store = Milvus(
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
        
        return True, "✅ Connected successfully!"
    
    except Exception as e:
        error_msg = f"❌ Failed to initialize vector store: {str(e)}"
        return False, error_msg


def extract_stocks_from_json(json_input):
    """
    Extract stock_code and stock_name from transaction JSON.
    
    Args:
        json_input: JSON string or file content
    
    Returns:
        list of dict with stock_code and stock_name
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
    global _vector_store
    
    if _vector_store is None:
        return []
    
    try:
        results = _vector_store.similarity_search(stock_query, k=int(num_results))
        
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
    global _vector_store
    
    # Check if vector store is initialized
    if _vector_store is None:
        success, message = initialize_vector_store()
        if not success:
            return json.dumps({
                "status": "error",
                "message": message
            }, indent=2, ensure_ascii=False)
    
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


def initialize_on_demand():
    """
    Initialize the vector store when needed.
    Returns status message.
    """
    global _vector_store
    
    if _vector_store is not None:
        return "✅ Vector store already initialized and ready!"
    
    success, message = initialize_vector_store()
    return message


def create_transaction_stock_search_tab():
    """Create and return the Transaction Stock Search tab"""
    with gr.Tab("📊 Transaction Stock Search"):
        gr.Markdown("### Search stocks from transaction JSON in Milvus vector store")
        gr.Markdown(
            "Upload or paste transaction JSON to extract stocks and search the vector store. "
            "The system will search by both stock code and stock name, returning results in JSON format."
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Connection Status")
                
                init_btn = gr.Button("🔌 Initialize Connection", variant="secondary")
                init_status = gr.Textbox(
                    label="Status",
                    value="Click 'Initialize Connection' to connect to Milvus",
                    interactive=False,
                    lines=2
                )
                
                gr.Markdown("---")
                gr.Markdown("#### Transaction JSON Input")
                
                json_input = gr.Textbox(
                    label="Transaction JSON",
                    placeholder='{\n  "transactions": [\n    {\n      "stock_code": "18138",\n      "stock_name": "騰訊認購證",\n      ...\n    }\n  ]\n}',
                    lines=15,
                    max_lines=25
                )
                
                with gr.Row():
                    num_results = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Results per Query"
                    )
                
                search_button = gr.Button("🔎 Search Stocks", variant="primary", size="lg")
                
            with gr.Column():
                gr.Markdown("#### Search Results (JSON)")
                
                output_box = gr.Textbox(
                    label="Results",
                    lines=30,
                    max_lines=40,
                    show_copy_button=True,
                    interactive=False
                )
        
        gr.Markdown("---")
        gr.Markdown(
            """
            ### 📝 Input Format Example
            
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
                }
              ]
            }
            ```
            
            ### 📤 Output Format
            
            The results will be returned in JSON format with:
            - `status`: Success or error status
            - `stocks_processed`: Number of stocks extracted
            - `search_results`: Array of results for each stock
              - `stock_code`: The stock code
              - `stock_name`: The stock name
              - `search_by_code`: Results when searching by stock code
              - `search_by_name`: Results when searching by stock name
            
            ### 💡 Tips
            - The search uses semantic similarity powered by Milvus
            - Each stock is searched twice: once by code and once by name
            - Results include content and metadata from the vector store
            """
        )
        
        # Set up event handlers
        init_btn.click(
            fn=initialize_on_demand,
            inputs=[],
            outputs=init_status
        )
        
        search_button.click(
            fn=process_transaction_json,
            inputs=[json_input, num_results],
            outputs=output_box
        )

