"""
Tab: JSON Batch Analysis
Process multiple conversations from a JSON string and extract stock information using LLMs.
Each conversation is analyzed sequentially with full stock extraction.
"""

import json
import traceback
import logging
import time
import requests
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter
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

def create_merged_stocks_json(combined_results: List[Dict[str, Any]]) -> str:
    """
    Create a merged/deduplicated JSON output with unique stocks and averaged relevance scores.
    
    This function takes all the stocks from all conversations and LLMs, deduplicates them by stock_number,
    and creates a single consolidated list with averaged relevance scores.
    
    Args:
        combined_results: List of conversation result dictionaries
        
    Returns:
        str: JSON string with merged stocks
    """
    # Dictionary to track stocks by stock_number
    stocks_dict = {}  # key: stock_number, value: list of stock data
    
    # Count total analyses for averaging
    total_analyses = 0
    
    # Collect all stocks from all conversations
    for conv_result in combined_results:
        stocks = conv_result.get("stocks", [])
        for stock in stocks:
            stock_number = stock.get("stock_number", "")
            if stock_number:
                if stock_number not in stocks_dict:
                    stocks_dict[stock_number] = []
                stocks_dict[stock_number].append(stock)
        
        # Count total analyses (number of LLM analyses across all conversations)
        llms_used = conv_result.get("llms_used", [])
        total_analyses += len(llms_used)
    
    # Merge stocks and calculate average relevance_score
    merged_stocks = []
    
    for stock_number, stock_list in stocks_dict.items():
        if not stock_list:
            continue
        
        # Calculate average relevance_score across ALL analyses
        # (not just the ones where the stock appeared)
        relevance_scores = [s.get("relevance_score", 0) for s in stock_list]
        total_score = sum(relevance_scores)
        avg_relevance_score = total_score / total_analyses if total_analyses > 0 else 0
        
        # Use the first stock's data as base
        merged_stock = {
            "stock_number": stock_number,
            "stock_name": stock_list[0].get("stock_name", ""),
            "relevance_score": round(avg_relevance_score, 2),
        }
        
        # Include original_word if present (use first non-empty one)
        original_words = [s.get("original_word", "") for s in stock_list if s.get("original_word")]
        if original_words:
            # Use the most common original word, or first if tied
            word_counts = Counter(original_words)
            most_common_word = word_counts.most_common(1)[0][0]
            merged_stock["original_word"] = most_common_word
        
        # Include quantity and price if present (use most common values)
        quantities = [s.get("quantity", "") for s in stock_list if s.get("quantity")]
        if quantities:
            # Use the most common quantity
            qty_counts = Counter(quantities)
            most_common_qty = qty_counts.most_common(1)[0][0]
            merged_stock["quantity"] = most_common_qty
        
        prices = [s.get("price", "") for s in stock_list if s.get("price")]
        if prices:
            # Use the most common price
            price_counts = Counter(prices)
            most_common_price = price_counts.most_common(1)[0][0]
            merged_stock["price"] = most_common_price
        
        # Include corrected stock information (always populate, use first available or original values)
        corrected_names = [s.get("corrected_stock_name") for s in stock_list if s.get("corrected_stock_name")]
        corrected_numbers = [s.get("corrected_stock_number") for s in stock_list if s.get("corrected_stock_number")]
        correction_confidences = [s.get("correction_confidence") for s in stock_list if s.get("correction_confidence")]
        
        # Always include these fields - use corrected values if available, otherwise use original values
        merged_stock["corrected_stock_number"] = corrected_numbers[0] if corrected_numbers else stock_number
        merged_stock["corrected_stock_name"] = corrected_names[0] if corrected_names else stock_list[0].get("stock_name", "")
        merged_stock["correction_confidence"] = correction_confidences[0] if correction_confidences else 1.0
        
        # Determine confidence - use the most common one, or highest if tied
        confidences = [s.get("confidence", "low").lower() for s in stock_list]
        confidence_priority = {"high": 3, "medium": 2, "low": 1}
        most_confident = max(confidences, key=lambda c: (confidences.count(c), confidence_priority.get(c, 0)))
        merged_stock["confidence"] = most_confident
        
        # Combine reasoning from all sources (if present)
        reasonings = [s.get("reasoning", "") for s in stock_list if s.get("reasoning")]
        if reasonings:
            # Only include unique reasonings
            unique_reasonings = list(dict.fromkeys(reasonings))  # Preserve order while removing duplicates
            if len(unique_reasonings) == 1:
                merged_stock["reasoning"] = unique_reasonings[0]
            else:
                merged_stock["reasoning"] = " | ".join(unique_reasonings)
        
        # Add metadata about how many times this stock was found
        merged_stock["detection_count"] = len(stock_list)
        
        # Track which LLM models detected this stock
        llm_models = [s.get("llm_model", "") for s in stock_list if s.get("llm_model")]
        if llm_models:
            unique_models = list(dict.fromkeys(llm_models))  # Preserve order while removing duplicates
            merged_stock["detected_by_llms"] = unique_models
        
        merged_stocks.append(merged_stock)
    
    # Sort by relevance_score (descending) then by stock_number
    merged_stocks.sort(key=lambda s: (-s["relevance_score"], s["stock_number"]))
    
    # Create simplified combined JSON with only stocks
    merged_data = {
        "stocks": merged_stocks,
        "metadata": {
            "total_conversations": len(combined_results),
            "total_analyses": total_analyses,
            "unique_stocks_found": len(merged_stocks),
            "note": "relevance_score is averaged across all analyses (conversations × LLMs)"
        }
    }
    
    # Format combined JSON
    return json.dumps(merged_data, indent=2, ensure_ascii=False)


def verify_stocks_in_conversations(
    merged_stocks: List[Dict[str, Any]],
    conversations: List[Dict[str, Any]],
    verification_llm: str,
    ollama_url: str,
    temperature: float,
) -> Dict[str, Any]:
    """
    Verify if extracted stocks really exist in the conversations using LLM analysis.
    
    Args:
        merged_stocks: List of merged stocks to verify
        conversations: Original conversation data
        verification_llm: LLM model to use for verification
        ollama_url: Ollama server URL
        temperature: Temperature parameter
        
    Returns:
        Dict with verification results for each stock
    """
    verification_results = []
    
    # Build a comprehensive conversation text
    all_conversation_texts = []
    for conv in conversations:
        transcriptions = conv.get("transcriptions", {})
        if isinstance(transcriptions, dict):
            for source_name, text in transcriptions.items():
                if text and text.strip():
                    all_conversation_texts.append(f"[{source_name}]: {text}")
                    break
        elif isinstance(transcriptions, str) and transcriptions.strip():
            all_conversation_texts.append(transcriptions)
    
    full_conversation_text = "\n\n---\n\n".join(all_conversation_texts)
    
    # Prepare verification prompt
    stocks_list = []
    for idx, stock in enumerate(merged_stocks, 1):
        stock_name = stock.get("corrected_stock_name") or stock.get("stock_name", "N/A")
        stock_number = stock.get("corrected_stock_number") or stock.get("stock_number", "N/A")
        stocks_list.append(f"{idx}. {stock_name} ({stock_number})")
    
    verification_system_message = """You are a stock conversation verification analyst. Your task is to carefully review conversations and verify whether specific stocks were actually mentioned or discussed.

For each stock provided, you need to:
1. Search through the conversation text for any mention of the stock (by name, number, or alias)
2. Determine if the stock was actually discussed or if it was incorrectly extracted
3. Provide evidence (quote) from the conversation if found
4. Assign a verification status: "CONFIRMED", "LIKELY", "UNCERTAIN", or "NOT_FOUND"

Be thorough and look for:
- Direct mentions of stock names or numbers
- Cantonese/Chinese nicknames or abbreviations
- Context clues that indicate the stock (e.g., "騰訊" for Tencent, "700" for 00700.HK)

Respond ONLY with valid JSON in this format:
{
  "verification_results": [
    {
      "stock_number": "00700",
      "stock_name": "騰訊控股",
      "verification_status": "CONFIRMED|LIKELY|UNCERTAIN|NOT_FOUND",
      "evidence": "quote from conversation that mentions this stock",
      "reasoning": "explanation of why you gave this status"
    }
  ]
}"""
    
    verification_prompt = f"""Please verify if the following stocks were actually mentioned in the conversations below:

**STOCKS TO VERIFY:**
{chr(10).join(stocks_list)}

**CONVERSATIONS:**
{full_conversation_text}

Please analyze each stock and determine if it was really discussed in the conversations above. Respond with JSON only."""
    
    try:
        # Call LLM for verification
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": verification_llm,
                "messages": [
                    {"role": "system", "content": verification_system_message},
                    {"role": "user", "content": verification_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result.get("message", {}).get("content", "")
            
            # Try to parse JSON from response
            try:
                # Extract JSON from response (handle cases where LLM adds extra text)
                json_start = llm_response.find("{")
                json_end = llm_response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    parsed_result = json.loads(json_str)
                    verification_results = parsed_result.get("verification_results", [])
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse verification JSON: {e}")
                logging.error(f"LLM Response: {llm_response}")
        else:
            logging.error(f"Verification LLM request failed: {response.status_code}")
    
    except Exception as e:
        logging.error(f"Error during stock verification: {e}")
        logging.error(traceback.format_exc())
    
    # Create verification map
    verification_map = {}
    for result in verification_results:
        stock_number = result.get("stock_number", "")
        if stock_number:
            verification_map[stock_number] = result
    
    return {
        "verification_results": verification_results,
        "verification_map": verification_map,
        "verification_llm": verification_llm,
        "verification_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def process_json_batch(
    json_input: str,
    selected_llms: list[str],
    system_message: str,
    ollama_url: str,
    temperature: float,
    use_vector_correction: bool = True,
    use_contextual_analysis: bool = True,
    enable_stock_verification: bool = False,
    verification_llm: str = None,
) -> tuple[str, str, str, str]:
    """
    Process a JSON batch of conversations and extract stock information.
    
    Args:
        json_input: JSON string containing array of conversation objects
        selected_llms: List of selected LLM names
        system_message: System message for the LLMs
        ollama_url: Ollama server URL
        temperature: Temperature parameter
        use_vector_correction: Whether to use vector store for stock name correction
        use_contextual_analysis: Whether to use previous conversation context for better analysis
        
    Returns:
        tuple[str, str, str, str]: (formatted_results, combined_json, merged_json, verification_results)
    """
    try:
        # Validate inputs
        if not json_input or not json_input.strip():
            return "❌ Error: Please provide a JSON input", "", "", ""
        
        if not selected_llms or len(selected_llms) == 0:
            return "❌ Error: Please select at least one LLM", "", "", ""
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL", "", "", ""
        
        # Parse JSON
        try:
            conversations = json.loads(json_input)
        except json.JSONDecodeError as e:
            return f"❌ Error: Invalid JSON format\n\n{str(e)}", "", "", ""
        
        # Validate that it's an array
        if not isinstance(conversations, list):
            return "❌ Error: JSON must be an array of conversation objects", "", "", ""
        
        if len(conversations) == 0:
            return "❌ Error: JSON array is empty", "", "", ""
        
        # Process each conversation
        all_results = []
        combined_results = []
        previous_contexts = []  # Store context from previous conversations
        
        total_conversations = len(conversations)
        total_llms = len(selected_llms)
        
        # Header
        output_parts = []
        output_parts.append("=" * 100)
        output_parts.append("🔬 JSON BATCH ANALYSIS - STOCK EXTRACTION")
        output_parts.append(f"Total Conversations: {total_conversations}")
        output_parts.append(f"Selected LLMs: {total_llms} ({', '.join(selected_llms)})")
        output_parts.append(f"Vector Store Correction: {'✅ Enabled' if use_vector_correction else '❌ Disabled'}")
        output_parts.append(f"Contextual Analysis: {'✅ Enabled' if use_contextual_analysis else '❌ Disabled'}")
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
                
                # Build contextual information from previous conversations
                contextual_system_message = system_message
                if use_contextual_analysis and previous_contexts:
                    output_parts.append(f"🔗 Using context from {len(previous_contexts)} previous conversation(s)")
                    output_parts.append("")
                    
                    context_summary = "\n\n**===== CONTEXT FROM PREVIOUS CONVERSATIONS =====**\n"
                    context_summary += "The following are summaries of previous conversations in this session. "
                    context_summary += "Use this information to understand references and context in the current conversation.\n\n"
                    
                    for prev_ctx in previous_contexts:
                        context_summary += f"--- Previous Conversation #{prev_ctx['conversation_number']} ---\n"
                        context_summary += f"Summary: {prev_ctx['summary']}\n"
                        if prev_ctx['stocks']:
                            context_summary += "Stocks discussed:\n"
                            for stock in prev_ctx['stocks']:
                                stock_name = stock.get('corrected_stock_name') or stock.get('stock_name', 'N/A')
                                stock_number = stock.get('corrected_stock_number') or stock.get('stock_number', 'N/A')
                                context_summary += f"  - {stock_name} ({stock_number})\n"
                        context_summary += "\n"
                    
                    context_summary += "**===== END OF PREVIOUS CONTEXT =====**\n"
                    context_summary += "\nNow analyze the CURRENT conversation below. When you see abbreviated references "
                    context_summary += "(like '窩輪' without a specific stock name), check if they might be referring to "
                    context_summary += "stocks mentioned in the previous conversations above.\n"
                    
                    # Append context to system message
                    contextual_system_message = system_message + "\n\n" + context_summary
                
                # Store results for this conversation
                conv_stocks = []
                conv_summary = ""
                
                # Process with each LLM
                for llm_idx, model in enumerate(selected_llms, 1):
                    msg = f"[Conversation {conv_number}/{total_conversations}] [LLM {llm_idx}/{total_llms}] Starting analysis with: {model}"
                    logging.info(msg)
                    print(msg)
                    
                    output_parts.append(f"🤖 Analyzing with LLM {llm_idx}/{total_llms}: {model}")
                    output_parts.append("")
                    
                    start_time = time.time()
                    
                    # Extract stocks (using contextual system message if available)
                    result_model, formatted_result, raw_json = extract_stocks_with_single_llm(
                        model=model,
                        conversation_text=transcription_text,
                        system_message=contextual_system_message,
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
                            
                            # Capture summary for contextual analysis (prefer first LLM's summary)
                            if not conv_summary and "summary" in parsed:
                                conv_summary = parsed["summary"]
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
                
                # Add this conversation's context for future conversations
                if use_contextual_analysis:
                    previous_contexts.append({
                        "conversation_number": conv_number,
                        "summary": conv_summary or "No summary available",
                        "stocks": conv_stocks
                    })
                
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
        
        # Format combined JSON (all conversations with all stocks)
        combined_json = json.dumps(combined_results, indent=2, ensure_ascii=False)
        
        # Create merged JSON (deduplicated stocks with averaged relevance scores)
        merged_json_data = json.loads(create_merged_stocks_json(combined_results))
        merged_stocks = merged_json_data.get("stocks", [])
        
        # Perform stock verification if enabled
        verification_output = ""
        if enable_stock_verification and verification_llm and merged_stocks:
            verification_parts = []
            verification_parts.append("=" * 100)
            verification_parts.append("🔍 STOCK VERIFICATION ANALYSIS")
            verification_parts.append(f"Verification LLM: {verification_llm}")
            verification_parts.append(f"Total Stocks to Verify: {len(merged_stocks)}")
            verification_parts.append("=" * 100)
            verification_parts.append("")
            verification_parts.append("Verifying if extracted stocks really exist in conversations...")
            verification_parts.append("")
            
            try:
                verification_result = verify_stocks_in_conversations(
                    merged_stocks=merged_stocks,
                    conversations=conversations,
                    verification_llm=verification_llm,
                    ollama_url=ollama_url,
                    temperature=temperature
                )
                
                verification_map = verification_result.get("verification_map", {})
                
                # Update merged stocks with verification status
                for stock in merged_stocks:
                    stock_number = stock.get("stock_number", "")
                    if stock_number in verification_map:
                        verification = verification_map[stock_number]
                        stock["verification_status"] = verification.get("verification_status", "UNKNOWN")
                        stock["verification_evidence"] = verification.get("evidence", "")
                        stock["verification_reasoning"] = verification.get("reasoning", "")
                    else:
                        stock["verification_status"] = "NOT_VERIFIED"
                
                # Add verification metadata
                merged_json_data["verification_metadata"] = {
                    "verification_enabled": True,
                    "verification_llm": verification_result.get("verification_llm"),
                    "verification_timestamp": verification_result.get("verification_timestamp")
                }
                
                # Display verification results
                verification_parts.append("📋 VERIFICATION RESULTS:")
                verification_parts.append("")
                
                # Group by verification status
                status_groups = {
                    "CONFIRMED": [],
                    "LIKELY": [],
                    "UNCERTAIN": [],
                    "NOT_FOUND": []
                }
                
                for stock in merged_stocks:
                    status = stock.get("verification_status", "NOT_VERIFIED")
                    if status in status_groups:
                        status_groups[status].append(stock)
                
                # Display each group
                for status, stocks_in_group in status_groups.items():
                    if stocks_in_group:
                        emoji_map = {
                            "CONFIRMED": "✅",
                            "LIKELY": "🟢",
                            "UNCERTAIN": "🟡",
                            "NOT_FOUND": "❌"
                        }
                        emoji = emoji_map.get(status, "❓")
                        verification_parts.append(f"{emoji} {status}: {len(stocks_in_group)} stock(s)")
                        
                        for stock in stocks_in_group:
                            stock_name = stock.get("corrected_stock_name") or stock.get("stock_name", "N/A")
                            stock_number = stock.get("corrected_stock_number") or stock.get("stock_number", "N/A")
                            evidence = stock.get("verification_evidence", "")
                            reasoning = stock.get("verification_reasoning", "")
                            
                            verification_parts.append(f"   • {stock_name} ({stock_number})")
                            if reasoning:
                                verification_parts.append(f"     Reasoning: {reasoning}")
                            if evidence:
                                evidence_preview = evidence[:200] + "..." if len(evidence) > 200 else evidence
                                verification_parts.append(f"     Evidence: \"{evidence_preview}\"")
                        verification_parts.append("")
                
                verification_parts.append("=" * 100)
                verification_parts.append("✓ VERIFICATION COMPLETED")
                verification_parts.append("=" * 100)
                
                verification_output = "\n".join(verification_parts)
                
            except Exception as verify_error:
                error_msg = f"❌ Verification Error: {str(verify_error)}"
                verification_parts.append(error_msg)
                verification_parts.append(f"Traceback: {traceback.format_exc()}")
                verification_output = "\n".join(verification_parts)
                logging.error(error_msg)
                logging.error(traceback.format_exc())
        
        # Convert merged_json_data back to JSON string
        merged_json = json.dumps(merged_json_data, indent=2, ensure_ascii=False)
        
        return "\n".join(output_parts), combined_json, merged_json, verification_output
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logging.error(error_msg)
        return error_msg, "", "", ""


# ============================================================================
# Gradio Tab Creation
# ============================================================================

def create_json_batch_analysis_tab():
    """Create and return the JSON Batch Analysis tab"""
    with gr.Tab("🔟 JSON Batch Analysis"):
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
                
                use_contextual_analysis_checkbox = gr.Checkbox(
                    label="🔗 Enable Contextual Analysis",
                    value=True,
                    info="Pass context from previous conversations to improve understanding of references and abbreviated mentions"
                )
                
                use_vector_correction_checkbox = gr.Checkbox(
                    label="🔧 Enable Vector Store Correction",
                    value=True,
                    info="Use Milvus vector store to correct stock names that may have STT errors"
                )
                
                enable_stock_verification_checkbox = gr.Checkbox(
                    label="🔍 Enable Stock Verification Analysis",
                    value=False,
                    info="Run an additional LLM round to verify if extracted stocks really exist in conversations"
                )
                
                verification_llm_dropdown = gr.Dropdown(
                    choices=LLM_OPTIONS,
                    label="Verification LLM Model",
                    value=LLM_OPTIONS[0],
                    info="Select LLM model for stock verification (only used if verification is enabled)",
                    interactive=True
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
                    lines=20,
                    interactive=False,
                    show_copy_button=True,
                )
                
                gr.Markdown("#### 📦 Combined JSON Output (All Conversations)")
                
                combined_json_box = gr.Textbox(
                    label="Complete Results in JSON Format",
                    lines=15,
                    interactive=False,
                    show_copy_button=True,
                    info="All conversations with their extracted stocks - organized by conversation"
                )
                
                gr.Markdown("#### 🎯 Merged JSON Output (Deduplicated Stocks)")
                
                merged_json_box = gr.Textbox(
                    label="Unique Stocks with Averaged Relevance Scores",
                    lines=15,
                    interactive=False,
                    show_copy_button=True,
                    info="All stocks merged and deduplicated by stock_number, with relevance scores averaged across all conversations and LLMs"
                )
                
                gr.Markdown("#### 🔍 Stock Verification Results")
                
                verification_results_box = gr.Textbox(
                    label="Stock Verification Analysis",
                    lines=15,
                    interactive=False,
                    show_copy_button=True,
                    info="Verification analysis results showing if extracted stocks really exist in conversations (only shown when verification is enabled)"
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
                use_contextual_analysis_checkbox,
                enable_stock_verification_checkbox,
                verification_llm_dropdown,
            ],
            outputs=[results_box, combined_json_box, merged_json_box, verification_results_box],
        )

