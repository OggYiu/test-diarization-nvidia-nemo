"""
Tab 12: Milvus Vector Search
Search through stock data using semantic similarity search powered by Milvus and Ollama embeddings.
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


def initialize_vector_store(status_callback=None):
    """
    Initialize the Milvus vector store with embeddings.
    
    Args:
        status_callback: Optional callback function to report status
    
    Returns:
        tuple: (success: bool, message: str)
    """
    global _vector_store
    
    try:
        if status_callback:
            status_callback("Loading embeddings model...")
        
        embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
        
        if status_callback:
            status_callback("Connecting to Milvus...")
        
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


def search_vector_store(query, num_results=5):
    """
    Search the vector store and return formatted results.
    
    Args:
        query: The search query - can be a JSON string with transactions or a plain text query
        num_results: Number of results to return per search (default: 5)
    
    Returns:
        Formatted string with search results
    """
    global _vector_store
    
    # Check if vector store is initialized
    if _vector_store is None:
        # Try to initialize
        success, message = initialize_vector_store()
        if not success:
            return message
    
    if not query or query.strip() == "":
        return "⚠️ Please enter a search query."
    
    try:
        # Try to parse as JSON first
        try:
            data = json.loads(query)
            if "transactions" in data and isinstance(data["transactions"], list):
                # JSON format with transactions - search each stock
                return search_transactions(data, int(num_results))
        except json.JSONDecodeError:
            # Not JSON, treat as plain text query
            pass
        
        # Plain text search
        results = _vector_store.similarity_search(query, k=int(num_results))
        
        if not results:
            return f"No results found for: '{query}'"
        
        output = f"🔍 Found {len(results)} results for: '{query}'\n\n"
        output += "=" * 80 + "\n\n"
        
        for idx, res in enumerate(results, 1):
            output += f"📄 Result {idx}:\n"
            output += f"{'-' * 80}\n"
            output += f"Content: {res.page_content}\n"
            if res.metadata:
                output += f"Metadata: {res.metadata}\n"
            output += f"{'-' * 80}\n\n"
        
        return output
    
    except Exception as e:
        return f"❌ Error during search: {str(e)}"


def search_transactions(data, num_results=5):
    """
    Search for each stock in the transactions list by stock code and stock name.
    
    Args:
        data: Dictionary containing 'transactions' list
        num_results: Number of results to return per search
    
    Returns:
        Formatted string with all search results
    """
    global _vector_store
    
    transactions = data.get("transactions", [])
    
    if not transactions:
        return "⚠️ No transactions found in the JSON data."
    
    output = f"🔍 Searching for {len(transactions)} stock(s)...\n\n"
    output += "=" * 80 + "\n\n"
    
    all_results = []
    
    for idx, transaction in enumerate(transactions, 1):
        stock_code = transaction.get("stock_code", "")
        stock_name = transaction.get("stock_name", "")
        transaction_type = transaction.get("transaction_type", "")
        quantity = transaction.get("quantity", "")
        price = transaction.get("price", "")
        confidence = transaction.get("confidence_score", "")
        
        output += f"📊 Stock {idx}: {stock_name} ({stock_code})\n"
        output += f"   Type: {transaction_type} | Quantity: {quantity} | Price: {price} | Confidence: {confidence}\n"
        output += "=" * 80 + "\n\n"
        
        # Search by stock code
        if stock_code:
            output += f"🔎 Searching by Stock Code: {stock_code}\n"
            output += "-" * 80 + "\n"
            
            try:
                code_results = _vector_store.similarity_search(stock_code, k=num_results)
                
                if code_results:
                    for res_idx, res in enumerate(code_results, 1):
                        output += f"   Result {res_idx}:\n"
                        output += f"   Content: {res.page_content[:200]}{'...' if len(res.page_content) > 200 else ''}\n"
                        if res.metadata:
                            output += f"   Metadata: {res.metadata}\n"
                        output += "\n"
                    all_results.extend(code_results)
                else:
                    output += "   No results found.\n\n"
            
            except Exception as e:
                output += f"   ❌ Error: {str(e)}\n\n"
        
        # Search by stock name
        if stock_name:
            output += f"🔎 Searching by Stock Name: {stock_name}\n"
            output += "-" * 80 + "\n"
            
            try:
                name_results = _vector_store.similarity_search(stock_name, k=num_results)
                
                if name_results:
                    for res_idx, res in enumerate(name_results, 1):
                        output += f"   Result {res_idx}:\n"
                        output += f"   Content: {res.page_content[:200]}{'...' if len(res.page_content) > 200 else ''}\n"
                        if res.metadata:
                            output += f"   Metadata: {res.metadata}\n"
                        output += "\n"
                    all_results.extend(name_results)
                else:
                    output += "   No results found.\n\n"
            
            except Exception as e:
                output += f"   ❌ Error: {str(e)}\n\n"
        
        output += "=" * 80 + "\n\n"
    
    # Summary
    output += f"\n📈 Summary:\n"
    output += f"   Total stocks searched: {len(transactions)}\n"
    output += f"   Total results found: {len(all_results)}\n"
    
    # Add overall summary if available
    if "overall_summary" in data:
        output += f"\n📝 Overall Summary:\n"
        output += f"{'-' * 80}\n"
        output += f"{data['overall_summary']}\n"
        output += f"{'-' * 80}\n"
    
    return output


def initialize_on_demand():
    """
    Initialize the vector store when the tab is first accessed.
    Returns status message.
    """
    global _vector_store
    
    if _vector_store is not None:
        return "✅ Vector store already initialized and ready!"
    
    success, message = initialize_vector_store()
    return message


def create_milvus_search_tab():
    """Create and return the Milvus Vector Search tab"""
    with gr.Tab("🔍 Milvus Search"):
        gr.Markdown("### Search through stock data using semantic similarity")
        gr.Markdown(
            "Search your phone call transcriptions and stock data using semantic similarity search. "
            "The search understands meaning, not just exact text matches."
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
                gr.Markdown("#### Search")
                
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter a search query or paste JSON with transactions (see documentation for JSON format)...",
                    lines=10
                )
                
                with gr.Row():
                    num_results = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of Results"
                    )
                
                search_button = gr.Button("🔎 Search", variant="primary", size="lg")
                
            with gr.Column():
                gr.Markdown("#### Search Results")
                
                output_box = gr.Textbox(
                    label="Results",
                    lines=25,
                    max_lines=35,
                    show_copy_button=True,
                    interactive=False
                )
        
        gr.Markdown("---")
        gr.Markdown(
            "💡 **Tips**: \n\n"
            "- The search uses semantic similarity powered by Milvus and Ollama embeddings, "
            "so it will find results that are conceptually similar to your query, not just exact text matches.\n\n"
            "- **JSON Input**: You can paste a JSON object with transaction data. The system will automatically "
            "search for each stock by both stock code and stock name. Each transaction should include `stock_code` "
            "and `stock_name` fields."
        )
        
        # Set up event handlers
        init_btn.click(
            fn=initialize_on_demand,
            inputs=[],
            outputs=init_status
        )
        
        search_button.click(
            fn=search_vector_store,
            inputs=[query_input, num_results],
            outputs=output_box
        )
        
        # Also allow Enter key to trigger search
        query_input.submit(
            fn=search_vector_store,
            inputs=[query_input, num_results],
            outputs=output_box
        )

