"""
Tab: LLM Chat
Simple LLM chat interface with system message and user prompt
"""

import time
import traceback
import gradio as gr
from langchain_ollama import ChatOllama

# Import centralized model configuration
from model_config import MODEL_OPTIONS, DEFAULT_MODEL, DEFAULT_OLLAMA_URL

# Import OpenCC translation utility
from opencc_utils import translate_to_traditional_chinese


def send_to_llm(
    system_message: str,
    user_prompt: str,
    model: str,
    ollama_url: str,
    temperature: float = 0.7,
) -> str:
    """
    Send message to LLM and get response
    
    Args:
        system_message: System message to set context
        user_prompt: User's prompt/question
        model: Model name to use
        ollama_url: Ollama server URL
        temperature: Temperature for response generation
    
    Returns:
        str: LLM response or error message
    """
    try:
        # Validate inputs
        if not user_prompt or not user_prompt.strip():
            return "❌ Error: Please provide a user prompt"
        
        if not model or not model.strip():
            return "❌ Error: Please select a model"
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL"
        
        # Initialize the LLM
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=temperature,
        )
        
        # Prepare messages
        messages = []
        if system_message and system_message.strip():
            messages.append(("system", system_message.strip()))
        messages.append(("human", user_prompt.strip()))
        
        # Get response
        start_time = time.time()
        resp = chat_llm.invoke(messages)
        elapsed_time = time.time() - start_time
        
        # Extract content
        try:
            response_content = getattr(resp, "content", str(resp))
        except Exception:
            response_content = str(resp)
        
        # Translate LLM response to Traditional Chinese
        response_content = translate_to_traditional_chinese(response_content)
        
        # Format response with metadata
        result = f"⏱️ Response time: {elapsed_time:.2f}s\n"
        result += f"🤖 Model: {model}\n"
        result += "=" * 50 + "\n\n"
        result += response_content
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg


def create_llm_chat_tab():
    """Create and return the LLM Chat tab"""
    with gr.Tab("💬 LLM Chat"):
        gr.Markdown("### Simple LLM Chat Interface")
        gr.Markdown("*Send messages to LLM with custom system message and prompt*")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Settings")
                
                # Model selection
                chat_model_dropdown = gr.Dropdown(
                    choices=MODEL_OPTIONS,
                    value=DEFAULT_MODEL,
                    label="Select Model",
                    info="Choose the LLM model to use",
                )
                
                # Ollama URL
                chat_ollama_url = gr.Textbox(
                    label="Ollama URL",
                    value=DEFAULT_OLLAMA_URL,
                    placeholder="http://localhost:11434",
                )
                
                # Temperature
                chat_temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher values = more creative, lower = more focused",
                )
                
                gr.Markdown("---")
                
                # System message
                chat_system_message = gr.Textbox(
                    label="System Message",
                    placeholder="Enter system message to set context (optional)...",
                    lines=5,
                    info="Sets the behavior and context for the LLM",
                )
                
                # User prompt
                chat_user_prompt = gr.Textbox(
                    label="User Prompt",
                    placeholder="Enter your prompt or question...",
                    lines=8,
                    info="Your message to the LLM",
                )
                
                # Send button
                chat_send_btn = gr.Button("🚀 Send Message", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("#### Response")
                
                # Response output
                chat_response_output = gr.Textbox(
                    label="LLM Response",
                    lines=25,
                    interactive=False,
                    show_copy_button=True,
                    placeholder="Response will appear here...",
                )
        
        # Connect the send button
        chat_send_btn.click(
            fn=send_to_llm,
            inputs=[
                chat_system_message,
                chat_user_prompt,
                chat_model_dropdown,
                chat_ollama_url,
                chat_temperature_slider,
            ],
            outputs=chat_response_output,
        )

