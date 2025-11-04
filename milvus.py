from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
import gradio as gr

# Your provided endpoint
CLUSTER_ENDPOINT = "https://in03-5fb5696b79d8b56.serverless.aws-eu-central-1.cloud.zilliz.com"

# Your API key (token)
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


def search_vector_store(query, num_results=5):
    """
    Search the vector store and return formatted results.
    
    Args:
        query: The search query string
        num_results: Number of results to return (default: 5)
    
    Returns:
        Formatted string with search results
    """
    if not query or query.strip() == "":
        return "Please enter a search query."
    
    try:
        results = vector_store.similarity_search(query, k=num_results)
        
        if not results:
            return "No results found."
        
        output = f"Found {len(results)} results for: '{query}'\n\n"
        output += "=" * 80 + "\n\n"
        
        for idx, res in enumerate(results, 1):
            output += f"Result {idx}:\n"
            output += f"{'-' * 80}\n"
            output += f"Content: {res.page_content}\n"
            output += f"Metadata: {res.metadata}\n"
            output += f"{'-' * 80}\n\n"
        
        return output
    
    except Exception as e:
        return f"Error during search: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Milvus Vector Search") as demo:
    gr.Markdown("# 🔍 Milvus Vector Search Interface")
    gr.Markdown("Search through your stock data using semantic similarity search powered by Milvus and Ollama embeddings.")
    
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter your search query here (e.g., '泡泡沬特' or any stock-related term)...",
                lines=2
            )
        with gr.Column(scale=1):
            num_results = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Number of Results"
            )
    
    search_button = gr.Button("🔎 Search", variant="primary", size="lg")
    
    output_box = gr.Textbox(
        label="Search Results",
        lines=20,
        max_lines=30,
        show_copy_button=True
    )
    
    # Set up the search action
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
    
    gr.Markdown("---")
    gr.Markdown("💡 **Tip**: The search uses semantic similarity, so it will find results that are conceptually similar to your query, not just exact matches.")

# Launch the app
if __name__ == "__main__":
    print("\nStarting Gradio interface...")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)