"""
Tab: JSON Batch Analysis
Process multiple conversations from a JSON string and extract stock information using LLMs.
Each conversation is analyzed sequentially with full stock extraction.
"""

import json
import traceback
import logging
import time
from typing import List, Dict, Any
from datetime import datetime
import gradio as gr

# Import from the stock comparison tab
from tabs.tab_stt_stock_comparison import (
    extract_stocks_with_single_llm,
    format_extraction_result,
    DEFAULT_SYSTEM_MESSAGE,
    LLM_OPTIONS,
)

# Import centralized model configuration
from model_config import DEFAULT_OLLAMA_URL


# ============================================================================
# JSON Batch Processing Functions
# ============================================================================

def process_json_batch(
    json_input: str,
    selected_llms: list[str],
    system_message: str,
    ollama_url: str,
    temperature: float,
    use_vector_correction: bool = True,
) -> tuple[str, str]:
    """
    Process a JSON batch of conversations and extract stock information.
    
    Args:
        json_input: JSON string containing array of conversation objects
        selected_llms: List of selected LLM names
        system_message: System message for the LLMs
        ollama_url: Ollama server URL
        temperature: Temperature parameter
        use_vector_correction: Whether to use vector store for stock name correction
        
    Returns:
        tuple[str, str]: (formatted_results, combined_json)
    """
    try:
        # Validate inputs
        if not json_input or not json_input.strip():
            return "❌ Error: Please provide a JSON input", ""
        
        if not selected_llms or len(selected_llms) == 0:
            return "❌ Error: Please select at least one LLM", ""
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL", ""
        
        # Parse JSON
        try:
            conversations = json.loads(json_input)
        except json.JSONDecodeError as e:
            return f"❌ Error: Invalid JSON format\n\n{str(e)}", ""
        
        # Validate that it's an array
        if not isinstance(conversations, list):
            return "❌ Error: JSON must be an array of conversation objects", ""
        
        if len(conversations) == 0:
            return "❌ Error: JSON array is empty", ""
        
        # Process each conversation
        all_results = []
        combined_results = []
        
        total_conversations = len(conversations)
        total_llms = len(selected_llms)
        
        # Header
        output_parts = []
        output_parts.append("=" * 100)
        output_parts.append("🔬 JSON BATCH ANALYSIS - STOCK EXTRACTION")
        output_parts.append(f"Total Conversations: {total_conversations}")
        output_parts.append(f"Selected LLMs: {total_llms} ({', '.join(selected_llms)})")
        output_parts.append(f"Vector Store Correction: {'✅ Enabled' if use_vector_correction else '❌ Disabled'}")
        output_parts.append("=" * 100)
        output_parts.append("")
        
        # Process each conversation
        for conv_idx, conversation in enumerate(conversations, 1):
            try:
                # Extract conversation details
                conv_number = conversation.get("conversation_number", conv_idx)
                filename = conversation.get("filename", "N/A")
                metadata = conversation.get("metadata", {})
                transcriptions = conversation.get("transcriptions", {})
                
                # Get the transcription text (use the first available transcription)
                transcription_text = None
                transcription_source = None
                
                # Try to get transcription from different sources
                if isinstance(transcriptions, dict):
                    for source_name, text in transcriptions.items():
                        if text and text.strip():
                            transcription_text = text
                            transcription_source = source_name
                            break
                elif isinstance(transcriptions, str):
                    transcription_text = transcriptions
                    transcription_source = "default"
                
                if not transcription_text or not transcription_text.strip():
                    warning_msg = f"⚠️ Skipping Conversation #{conv_number} - No transcription text found"
                    output_parts.append(warning_msg)
                    output_parts.append("")
                    logging.warning(warning_msg)
                    continue
                
                # Display conversation header
                output_parts.append("\n" + "=" * 100)
                output_parts.append(f"📞 CONVERSATION #{conv_number} / {total_conversations}")
                output_parts.append("=" * 100)
                output_parts.append(f"📁 Filename: {filename}")
                output_parts.append(f"🎤 Transcription Source: {transcription_source}")
                
                # Display metadata if available
                if metadata:
                    output_parts.append(f"👤 Broker: {metadata.get('broker_name', 'N/A')} (ID: {metadata.get('broker_id', 'N/A')})")
                    output_parts.append(f"👥 Client: {metadata.get('client_name', 'N/A')} (ID: {metadata.get('client_id', 'N/A')})")
                    output_parts.append(f"📅 HKT DateTime: {metadata.get('hkt_datetime', 'N/A')}")
                
                output_parts.append("")
                output_parts.append(f"📝 Transcription:")
                output_parts.append("-" * 100)
                # Show truncated transcription
                trans_preview = transcription_text[:500] + "..." if len(transcription_text) > 500 else transcription_text
                for line in trans_preview.split("\n"):
                    output_parts.append(f"   {line}")
                output_parts.append("-" * 100)
                output_parts.append("")
                
                # Store results for this conversation
                conv_stocks = []
                
                # Process with each LLM
                for llm_idx, model in enumerate(selected_llms, 1):
                    msg = f"[Conversation {conv_number}/{total_conversations}] [LLM {llm_idx}/{total_llms}] Starting analysis with: {model}"
                    logging.info(msg)
                    print(msg)
                    
                    output_parts.append(f"🤖 Analyzing with LLM {llm_idx}/{total_llms}: {model}")
                    output_parts.append("")
                    
                    start_time = time.time()
                    
                    # Extract stocks
                    result_model, formatted_result, raw_json = extract_stocks_with_single_llm(
                        model=model,
                        conversation_text=transcription_text,
                        system_message=system_message,
                        ollama_url=ollama_url,
                        temperature=temperature,
                        stt_source=transcription_source,
                        use_vector_correction=use_vector_correction
                    )
                    
                    elapsed_time = time.time() - start_time
                    msg = f"[Conversation {conv_number}/{total_conversations}] [LLM {llm_idx}/{total_llms}] Completed: {model} - Time: {elapsed_time:.2f}s"
                    logging.info(msg)
                    print(msg)
                    
                    # Display results
                    output_parts.append("┌─ RESULTS")
                    for line in formatted_result.split("\n"):
                        output_parts.append(f"│  {line}")
                    output_parts.append("└" + "─" * 99)
                    output_parts.append("")
                    
                    # Parse and store stocks for combined output
                    if raw_json and raw_json.strip():
                        try:
                            parsed = json.loads(raw_json)
                            stocks = parsed.get("stocks", [])
                            for stock in stocks:
                                stock["llm_model"] = model
                            conv_stocks.extend(stocks)
                        except json.JSONDecodeError:
                            pass
                
                # Create combined result for this conversation
                conversation_result = {
                    "conversation_number": conv_number,
                    "filename": filename,
                    "metadata": metadata,
                    "transcription_source": transcription_source,
                    "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "llms_used": selected_llms,
                    "stocks": conv_stocks
                }
                
                combined_results.append(conversation_result)
                
                output_parts.append(f"✅ Completed Conversation #{conv_number}")
                output_parts.append("")
                
            except Exception as conv_error:
                error_msg = f"❌ Error processing Conversation #{conv_idx}: {str(conv_error)}"
                output_parts.append(error_msg)
                output_parts.append(f"Traceback: {traceback.format_exc()}")
                output_parts.append("")
                logging.error(error_msg)
                logging.error(traceback.format_exc())
        
        # Final summary
        output_parts.append("\n" + "=" * 100)
        output_parts.append("✓ BATCH ANALYSIS COMPLETED")
        output_parts.append(f"Total Conversations Processed: {len(combined_results)} / {total_conversations}")
        output_parts.append("=" * 100)
        
        # Format combined JSON
        combined_json = json.dumps(combined_results, indent=2, ensure_ascii=False)
        
        return "\n".join(output_parts), combined_json
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logging.error(error_msg)
        return error_msg, ""


# ============================================================================
# Gradio Tab Creation
# ============================================================================

def create_json_batch_analysis_tab():
    """Create and return the JSON Batch Analysis tab"""
    with gr.Tab("🔟 JSON Batch Analysis"):
        gr.Markdown("### Batch Process Multiple Conversations from JSON")
        gr.Markdown("""
        **Process multiple conversations at once!** Paste a JSON array with conversation objects, 
        and this tool will analyze each conversation sequentially to extract stock information.
        
        **JSON Format:**
        - Array of conversation objects
        - Each object should have: `conversation_number`, `filename`, `metadata`, `transcriptions`
        - The `transcriptions` field can be a dictionary (with source names as keys) or a string
        
        **Features:**
        - Sequential processing of conversations
        - Multi-LLM support (analyze with multiple models)
        - Vector Store Correction for STT errors
        - Comprehensive metadata tracking
        - Combined JSON output with all results
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📋 JSON Input")
                
                json_input_box = gr.Textbox(
                    label="JSON Conversations",
                    placeholder="""Paste your JSON here, e.g.:
[
  {
    "conversation_number": 1,
    "filename": "example.wav",
    "metadata": {
      "broker_name": "Dickson Lau",
      "client_name": "CHENG SUK HING"
    },
    "transcriptions": {
      "sensevoice": "經紀: 你好\\n客戶: 我想買騰訊"
    }
  }
]""",
                    lines=20,
                    show_copy_button=True,
                )
                
                gr.Markdown("#### 🤖 Select LLMs for Analysis")
                
                llm_checkboxes = gr.CheckboxGroup(
                    choices=LLM_OPTIONS,
                    label="Available LLMs",
                    value=[LLM_OPTIONS[0]],  # Default to first model
                    info="Select one or more LLMs to analyze each conversation"
                )
                
                gr.Markdown("#### ⚙️ Advanced Settings")
                
                use_vector_correction_checkbox = gr.Checkbox(
                    label="🔧 Enable Vector Store Correction",
                    value=True,
                    info="Use Milvus vector store to correct stock names that may have STT errors"
                )
                
                system_message_box = gr.Textbox(
                    label="系統訊息 (System Message)",
                    value=DEFAULT_SYSTEM_MESSAGE,
                    lines=10,
                    max_lines=20,
                    show_copy_button=True,
                )
                
                with gr.Row():
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.1,
                        step=0.1,
                        label="Temperature",
                        info="Lower = more deterministic"
                    )
                
                ollama_url_box = gr.Textbox(
                    label="Ollama URL",
                    value=DEFAULT_OLLAMA_URL,
                    placeholder="http://localhost:11434",
                )
                
                analyze_btn = gr.Button(
                    "🚀 Analyze All Conversations",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("#### 📊 Analysis Results")
                
                results_box = gr.Textbox(
                    label="Batch Analysis Results",
                    lines=30,
                    interactive=False,
                    show_copy_button=True,
                )
                
                gr.Markdown("#### 📦 Combined JSON Output")
                
                combined_json_box = gr.Textbox(
                    label="Complete Results in JSON Format",
                    lines=20,
                    interactive=False,
                    show_copy_button=True,
                    info="All conversations with their extracted stocks in a single JSON structure"
                )
        
        # Connect the analyze button
        analyze_btn.click(
            fn=process_json_batch,
            inputs=[
                json_input_box,
                llm_checkboxes,
                system_message_box,
                ollama_url_box,
                temperature_slider,
                use_vector_correction_checkbox,
            ],
            outputs=[results_box, combined_json_box],
        )

