"""
Tab: Multi-LLM Comparison
Query multiple LLMs at once with custom system and user prompts
"""

import traceback
from pathlib import Path
import gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_ollama import ChatOllama


# Common model options
LLM_OPTIONS = [
    "qwen3:32b",
    "gpt-oss:20b",
    "gemma3-27b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
]

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_SYSTEM_PROMPT = ""


def query_single_llm(
    model: str,
    system_prompt: str,
    user_message: str,
    ollama_url: str,
    temperature: float = 0.7
) -> tuple[str, str]:
    """
    Query a single LLM and return the response.
    
    Args:
        model: Model name
        system_prompt: System prompt
        user_message: User message
        ollama_url: Ollama server URL
        temperature: Temperature parameter
        
    Returns:
        tuple: (model_name, response_text)
    """
    try:
        # Initialize the LLM
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=temperature,
        )
        
        # Prepare messages
        messages = [
            ("system", system_prompt),
            ("human", user_message),
        ]
        
        # Get response
        resp = chat_llm.invoke(messages)
        
        # Extract content
        try:
            response_content = getattr(resp, "content", str(resp))
        except Exception:
            response_content = str(resp)
        
        return (model, f"✓ Success\n\n{response_content}")
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return (model, error_msg)


def query_multiple_llms(
    system_prompt: str,
    user_message: str,
    selected_llms: list[str],
    ollama_url: str,
    temperature: float,
) -> str:
    """
    Query multiple LLMs concurrently and return combined results.
    
    Args:
        system_prompt: System prompt for all LLMs
        user_message: User message for all LLMs
        selected_llms: List of selected LLM names
        ollama_url: Ollama server URL
        temperature: Temperature parameter
        
    Returns:
        str: Formatted results from all LLMs
    """
    try:
        # Validate inputs
        if not system_prompt or not system_prompt.strip():
            return "❌ Error: Please provide a system prompt"
        
        if not user_message or not user_message.strip():
            return "❌ Error: Please provide a user message"
        
        if not selected_llms or len(selected_llms) == 0:
            return "❌ Error: Please select at least one LLM"
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL"
        
        # Show progress
        results = {}
        
        # Query LLMs concurrently
        with ThreadPoolExecutor(max_workers=len(selected_llms)) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(
                    query_single_llm,
                    model,
                    system_prompt,
                    user_message,
                    ollama_url,
                    temperature
                ): model for model in selected_llms
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model, response = future.result()
                results[model] = response
        
        # Format output
        output_parts = []
        output_parts.append("=" * 80)
        output_parts.append(f"📊 MULTI-LLM COMPARISON RESULTS")
        output_parts.append(f"Selected LLMs: {len(selected_llms)}")
        output_parts.append("=" * 80)
        output_parts.append("")
        
        for i, model in enumerate(selected_llms, 1):
            output_parts.append(f"\n{'=' * 80}")
            output_parts.append(f"🤖 LLM {i}/{len(selected_llms)}: {model}")
            output_parts.append("=" * 80)
            output_parts.append("")
            output_parts.append(results.get(model, "❌ No response"))
            output_parts.append("")
        
        output_parts.append("=" * 80)
        output_parts.append("✓ All queries completed")
        output_parts.append("=" * 80)
        
        return "\n".join(output_parts)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg


def create_multi_llm_tab():
    """Create and return the Multi-LLM Comparison tab"""
    with gr.Tab("8️⃣ Multi-LLM Query"):
        gr.Markdown("### Query Multiple LLMs Simultaneously")
        gr.Markdown("Send the same prompt to multiple LLMs and compare their responses side-by-side.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Configuration")
                
                # System Prompt
                system_prompt_box = gr.Textbox(
                    label="System Prompt",
                    placeholder="Enter the system prompt that defines the LLM's role and behavior...",
                    lines=8,
                    value=DEFAULT_SYSTEM_PROMPT,
                )
                
                # User Message
                user_message_box = gr.Textbox(
                    label="User Message",
                    placeholder="Enter your question or prompt here...",
                    lines=8,
                )
                
                gr.Markdown("#### Select LLMs")
                gr.Markdown("*Choose one or more LLMs to query*")
                
                # LLM Checkboxes
                llm_checkboxes = gr.CheckboxGroup(
                    choices=LLM_OPTIONS,
                    label="Available LLMs",
                    value=[LLM_OPTIONS[0]],  # Default to first model
                    info="Select multiple LLMs to compare their responses"
                )
                
                gr.Markdown("#### Advanced Settings")
                
                with gr.Row():
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Higher values = more creative, lower = more deterministic"
                    )
                
                ollama_url_box = gr.Textbox(
                    label="Ollama URL",
                    value=DEFAULT_OLLAMA_URL,
                    placeholder="http://localhost:11434",
                )
                
                query_btn = gr.Button("🚀 Query Selected LLMs", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("#### Results")
                
                results_box = gr.Textbox(
                    label="LLM Responses",
                    lines=30,
                    interactive=False,
                    show_copy_button=True,
                )
        
        # Connect the button click event
        query_btn.click(
            fn=query_multiple_llms,
            inputs=[
                system_prompt_box,
                user_message_box,
                llm_checkboxes,
                ollama_url_box,
                temperature_slider,
            ],
            outputs=[results_box],
        )

