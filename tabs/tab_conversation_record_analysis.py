"""
Tab: Conversation Record Analysis
Analyze trade records from trades.csv against a conversation to determine confidence scores
"""

import json
import csv
import traceback
from datetime import datetime
from typing import Optional
import gradio as gr
import os

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# Import centralized model configuration
from model_config import MODEL_OPTIONS, DEFAULT_MODEL, DEFAULT_OLLAMA_URL


class RecordConfidence(BaseModel):
    """Confidence analysis for a single trade record"""
    
    order_no: str = Field(
        description="The OrderNo from the trade record"
    )
    
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that this trade was mentioned in the conversation. 0.0=definitely not mentioned, 1.0=definitely mentioned"
    )
    
    reasoning: str = Field(
        description="Detailed reasoning for the confidence score, citing specific evidence from the conversation"
    )
    
    matched_conversation_segments: list[str] = Field(
        default_factory=list,
        description="Specific segments from the conversation that mention this trade (if any)"
    )


class ConversationAnalysisResult(BaseModel):
    """Complete analysis result for all trade records"""
    
    records_analyzed: list[RecordConfidence] = Field(
        description="Confidence analysis for each trade record"
    )
    
    total_confidence_summary: dict = Field(
        description="Summary statistics: average confidence, high confidence count, etc."
    )
    
    conversation_summary: str = Field(
        description="Brief summary of what was discussed in the conversation"
    )
    
    overall_assessment: str = Field(
        description="Overall assessment of how well the trades match the conversation"
    )


def parse_datetime(date_str: str) -> Optional[datetime]:
    """Parse datetime string in multiple formats"""
    formats = [
        "%Y-%m-%dT%H:%M:%S",  # ISO format
        "%Y-%m-%d %H:%M:%S.%f",  # trades.csv format
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",  # Date only
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def extract_date_from_conversation(conversation_data: dict) -> Optional[str]:
    """
    Extract date from conversation JSON
    Looks for hkt_datetime in metadata or transactions
    """
    # Try metadata first
    if "metadata" in conversation_data:
        metadata = conversation_data["metadata"]
        if "hkt_datetime" in metadata:
            hkt_datetime = metadata["hkt_datetime"]
            # Check if it's a valid value (not None, not empty, not "N/A")
            if hkt_datetime and str(hkt_datetime).strip() and str(hkt_datetime).strip().upper() != "N/A":
                return str(hkt_datetime).strip()
    
    # Try transactions
    if "transactions" in conversation_data:
        transactions = conversation_data["transactions"]
        if transactions and len(transactions) > 0:
            first_tx = transactions[0]
            if "hkt_datetime" in first_tx:
                hkt_datetime = first_tx["hkt_datetime"]
                if hkt_datetime and str(hkt_datetime).strip() and str(hkt_datetime).strip().upper() != "N/A":
                    return str(hkt_datetime).strip()
    
    return None


def load_trades_for_date(
    trades_file: str,
    target_date: datetime,
    client_id: Optional[str] = None
) -> list[dict]:
    """
    Load all trade records from trades.csv for a specific date
    
    Args:
        trades_file: Path to trades.csv
        target_date: Target date to filter records
        client_id: Optional client ID to filter records
    
    Returns:
        List of trade records matching the criteria
    """
    if not os.path.exists(trades_file):
        raise FileNotFoundError(f"Trades file not found: {trades_file}")
    
    matching_records = []
    target_date_str = target_date.date()
    
    with open(trades_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Filter by client ID if provided
            if client_id:
                trade_client_id = row.get("ACCode", "")
                if trade_client_id != client_id:
                    continue
            
            # Parse and filter by date
            trade_dt_str = row.get("OrderTime", "")
            trade_dt = parse_datetime(trade_dt_str)
            
            if trade_dt and trade_dt.date() == target_date_str:
                matching_records.append(row)
    
    return matching_records


def format_conversation_text(conversation_data: dict, conversation_label: str = "") -> str:
    """
    Format conversation data into readable text for LLM analysis
    
    Args:
        conversation_data: Single conversation object
        conversation_label: Optional label to identify this conversation (e.g., "Conversation #1")
    """
    text_parts = []
    
    # Add conversation label if provided
    if conversation_label:
        text_parts.append(f"\n{'='*80}")
        text_parts.append(f"📞 {conversation_label}")
        text_parts.append(f"{'='*80}\n")
    
    # Add metadata if available
    if "metadata" in conversation_data:
        metadata = conversation_data["metadata"]
        text_parts.append("=== CALL METADATA ===")
        for key, value in metadata.items():
            text_parts.append(f"{key}: {value}")
        text_parts.append("")
    
    # Add transcriptions (handle both formats)
    if "transcriptions" in conversation_data:
        transcriptions = conversation_data["transcriptions"]
        
        # Format 1: Array of transcription objects
        # [{"model": "wsyue-asr", "text": "..."}]
        if isinstance(transcriptions, list):
            for idx, trans in enumerate(transcriptions, 1):
                text_parts.append(f"=== TRANSCRIPTION {idx} ===")
                text_parts.append(trans.get("text", ""))
                text_parts.append("")
        
        # Format 2: Dictionary with model names as keys
        # {"sensevoice": "...", "wsyue-asr": "..."}
        elif isinstance(transcriptions, dict):
            for idx, (model_name, text) in enumerate(transcriptions.items(), 1):
                text_parts.append(f"=== TRANSCRIPTION {idx} ({model_name}) ===")
                text_parts.append(str(text))
                text_parts.append("")
    
    # Add conversations
    if "conversations" in conversation_data:
        conversations = conversation_data["conversations"]
        for idx, conv in enumerate(conversations, 1):
            text_parts.append(f"=== CONVERSATION {idx} ===")
            for turn in conv:
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "")
                text_parts.append(f"[{speaker}]: {text}")
            text_parts.append("")
    
    return "\n".join(text_parts)


def combine_multiple_conversations(conversations_list: list[dict]) -> str:
    """
    Combine multiple conversation objects into a single unified context
    
    This is useful when conversations are related (e.g., same day, same client)
    and context from earlier conversations can help analyze later ones.
    
    Args:
        conversations_list: List of conversation objects
        
    Returns:
        Combined conversation text with all conversations merged
    """
    combined_parts = []
    
    combined_parts.append(f"{'='*80}")
    combined_parts.append(f"📞 COMBINED ANALYSIS OF {len(conversations_list)} CONVERSATIONS")
    combined_parts.append(f"{'='*80}\n")
    combined_parts.append(f"Note: This analysis treats all {len(conversations_list)} conversations as one unified context.")
    combined_parts.append(f"Context from earlier conversations may help understand later ones.\n")
    
    # Combine all conversations
    for idx, conv_data in enumerate(conversations_list, 1):
        conv_label = f"Conversation #{idx}"
        if "filename" in conv_data:
            conv_label += f" ({conv_data['filename']})"
        elif "metadata" in conv_data and "filename" in conv_data["metadata"]:
            conv_label += f" ({conv_data['metadata']['filename']})"
        
        conv_text = format_conversation_text(conv_data, conv_label)
        combined_parts.append(conv_text)
        combined_parts.append("")  # Blank line between conversations
    
    return "\n".join(combined_parts)


def format_trade_record(record: dict) -> str:
    """Format a trade record for LLM analysis"""
    return f"""OrderNo: {record.get('OrderNo', 'N/A')}
OrderTime: {record.get('OrderTime', 'N/A')}
Stock Code: {record.get('SCTYCode', 'N/A')}
Stock Name: {record.get('stock_name', 'N/A')}
Order Side: {record.get('OrderSide', 'N/A')} ({'Buy/Bid' if record.get('OrderSide')=='B' else 'Sell/Ask' if record.get('OrderSide')=='A' else 'Unknown'})
Quantity: {record.get('OrderQty', 'N/A')}
Price: {record.get('OrderPrice', 'N/A')}
Status: {record.get('OrderStatus', 'N/A')}
Client: {record.get('ACCode', 'N/A')}
Broker: {record.get('AECode', 'N/A')}"""


def analyze_records_with_llm(
    conversation_text: str,
    trade_records: list[dict],
    model_name: str,
    ollama_url: str,
    temperature: float = 0.3
) -> dict:
    """
    Use LLM to analyze each trade record against the conversation
    
    Args:
        conversation_text: Formatted conversation text
        trade_records: List of trade records to analyze
        model_name: LLM model to use
        ollama_url: Ollama API URL
        temperature: LLM temperature
    
    Returns:
        Analysis results as dictionary
    """
    # Initialize LLM
    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=temperature,
    )
    
    # Create system message
    system_message = """You are an expert financial analyst specializing in analyzing phone call transcriptions and trade records.

Your task is to analyze trade records from a trading system and determine how confident you are that each trade was mentioned or discussed in the phone conversation.

For each trade record, you need to:
1. Carefully examine the conversation for any mention of the stock, quantities, prices, buy/sell actions
2. Assign a confidence score from 0.0 to 1.0:
   - 0.0: Definitely NOT mentioned in the conversation
   - 0.1-0.3: Possibly related but very uncertain
   - 0.4-0.6: Some evidence but not clear
   - 0.7-0.9: Strong evidence, likely mentioned
   - 1.0: Definitely mentioned with clear confirmation
3. Provide detailed reasoning citing specific conversation segments
4. Note any exact matches (stock codes, prices, quantities)

Consider:
- Speech recognition errors (homophone confusion in Cantonese)
- Different ways to refer to the same stock
- Implied vs explicit confirmations
- Timing and context of the conversation

Be thorough but conservative in your confidence scores. Only give high scores when there is clear evidence."""

    # Analyze all records in one call for efficiency
    records_summary = []
    for idx, record in enumerate(trade_records, 1):
        records_summary.append(f"\n--- Record #{idx} ---\n{format_trade_record(record)}")
    
    all_records_text = "\n".join(records_summary)
    
    # Create analysis prompt
    user_message = f"""Here is the phone conversation transcript:

{conversation_text}

===================================

Now analyze the following {len(trade_records)} trade record(s) and determine the confidence that each was mentioned in the conversation:

{all_records_text}

===================================

For each record, provide:
1. The OrderNo
2. Confidence score (0.0-1.0)
3. Detailed reasoning
4. Any matching conversation segments

Then provide:
- Overall summary statistics
- Brief conversation summary
- Overall assessment of how well trades match the conversation"""

    # Get structured output from LLM
    try:
        structured_llm = llm.with_structured_output(ConversationAnalysisResult)
        result = structured_llm.invoke([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        
        # Convert Pydantic model to dict
        return result.model_dump()
        
    except Exception as e:
        # Fallback: try without structured output
        print(f"Structured output failed, falling back to regular output: {e}")
        response = llm.invoke([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        
        # Return raw response
        return {
            "error": "Structured output failed",
            "raw_response": response.content if hasattr(response, 'content') else str(response)
        }


def analyze_conversation_records(
    conversation_json: str,
    trades_file_path: str,
    client_id_filter: str,
    model_name: str,
    ollama_url: str,
    temperature: float,
    use_combined_analysis: bool = False
) -> tuple[str, str]:
    """
    Main function to analyze trade records against conversation
    
    Args:
        conversation_json: JSON string containing conversation(s)
        trades_file_path: Path to trades CSV file
        client_id_filter: Optional client ID filter
        model_name: LLM model to use
        ollama_url: Ollama API URL
        temperature: LLM temperature
        use_combined_analysis: If True and input is array, analyze all conversations combined
    
    Returns:
        tuple: (formatted_text_result, json_result)
    """
    try:
        # Parse conversation JSON
        if not conversation_json.strip():
            error_msg = "❌ Error: Please provide conversation JSON"
            return (error_msg, "")
        
        try:
            conversation_data = json.loads(conversation_json)
            
            # Handle array format: [{"metadata": ...}, {"metadata": ...}]
            if isinstance(conversation_data, list):
                if len(conversation_data) == 0:
                    error_msg = "❌ Error: Conversation array is empty"
                    return (error_msg, "")
                
                # Check if combined analysis is requested
                if use_combined_analysis and len(conversation_data) > 1:
                    print(f"✨ Combined Analysis Mode: Analyzing {len(conversation_data)} conversations as one unified context")
                    # Store the full list for combined analysis
                    conversation_list = conversation_data
                    # Use first conversation for date extraction
                    conversation_data = conversation_data[0]
                    is_combined_mode = True
                else:
                    # Use the first conversation only
                    conversation_data = conversation_data[0]
                    conversation_list = None
                    is_combined_mode = False
                    if len(json.loads(conversation_json)) > 1:
                        print(f"⚠️ Note: Input is an array with {len(json.loads(conversation_json))} conversations. Analyzing the first one only.")
            else:
                conversation_list = None
                is_combined_mode = False
            
        except json.JSONDecodeError as e:
            error_msg = f"❌ Error: Cannot parse conversation JSON: {str(e)}"
            return (error_msg, "")
        
        # Extract date from conversation
        date_str = extract_date_from_conversation(conversation_data)
        if not date_str:
            # Provide detailed error message showing what was found
            debug_info = []
            if "metadata" in conversation_data:
                meta = conversation_data["metadata"]
                if "hkt_datetime" in meta:
                    debug_info.append(f"Found metadata.hkt_datetime = '{meta['hkt_datetime']}'")
                else:
                    debug_info.append("metadata exists but no 'hkt_datetime' field")
                    debug_info.append(f"Available metadata fields: {list(meta.keys())}")
            else:
                debug_info.append("No 'metadata' field in JSON")
            
            if "transactions" in conversation_data:
                debug_info.append("Found 'transactions' field")
            
            error_msg = f"""❌ Error: Cannot find valid date in conversation JSON

Required: hkt_datetime field in metadata or transactions[0]

Debug info:
{chr(10).join(f"  • {info}" for info in debug_info)}

Please ensure your JSON has:
  {{"metadata": {{"hkt_datetime": "2025-10-20T10:01:20"}}}}
  
The hkt_datetime value must not be empty, None, or "N/A"."""
            return (error_msg, "")
        
        target_date = parse_datetime(date_str)
        if not target_date:
            error_msg = f"❌ Error: Cannot parse datetime: {date_str}"
            return (error_msg, "")
        
        # Get client ID filter
        client_id = client_id_filter.strip() if client_id_filter.strip() else None
        if not client_id:
            # Try to extract from conversation metadata
            if "metadata" in conversation_data:
                client_id = conversation_data["metadata"].get("client_id", None)
        
        # Load matching trade records
        trade_records = load_trades_for_date(trades_file_path, target_date, client_id)
        
        if len(trade_records) == 0:
            error_msg = f"❌ No trade records found for date {target_date.date()}"
            if client_id:
                error_msg += f" and client {client_id}"
            return (error_msg, "")
        
        # Format conversation for analysis
        if is_combined_mode and conversation_list:
            # Combined analysis: merge all conversations
            conversation_text = combine_multiple_conversations(conversation_list)
        else:
            # Single conversation analysis
            conversation_text = format_conversation_text(conversation_data)
        
        # Check if conversation has actual content
        if not conversation_text.strip() or len(conversation_text.strip()) < 50:
            error_msg = """❌ Error: Conversation appears to be empty or too short

Your JSON must include either:
  • "conversations": [...] - array of conversation turns
  OR
  • "transcriptions": [...] - array of transcription objects

Example:
{
  "metadata": {"hkt_datetime": "2025-10-20T10:01:20", ...},
  "conversations": [
    [
      {"speaker": "Broker", "text": "你好"},
      {"speaker": "Client", "text": "我想買股票"}
    ]
  ]
}"""
            return (error_msg, "")
        
        # Analyze with LLM
        analysis_result = analyze_records_with_llm(
            conversation_text,
            trade_records,
            model_name,
            ollama_url,
            temperature
        )
        
        # Build formatted text output
        # Check if original input was an array
        original_data = json.loads(conversation_json)
        is_array = isinstance(original_data, list)
        
        output_text = f"""{'='*80}
📊 CONVERSATION RECORD ANALYSIS
{'='*80}
"""
        
        # Add note about analysis mode
        if is_combined_mode and conversation_list:
            output_text += f"""
✨ COMBINED ANALYSIS MODE ENABLED
   Analyzing {len(conversation_list)} conversations as one unified context.
   Context from earlier conversations helps understand later ones.
   
   Conversations analyzed:
"""
            for idx, conv in enumerate(conversation_list, 1):
                filename = conv.get('filename', conv.get('metadata', {}).get('filename', f'Conversation #{idx}'))
                output_text += f"   {idx}. {filename}\n"
            output_text += "\n"
        elif is_array:
            conv_num = conversation_data.get('conversation_number', 1)
            filename = conversation_data.get('filename', 'N/A')
            output_text += f"""
⚠️ NOTE: Input contains {len(original_data)} conversations.
         Analyzing conversation #{conv_num}: {filename}
         (Enable "Combined Analysis" to analyze all conversations together)

"""
        
        output_text += f"""
📅 Date: {target_date.date()}
👤 Client: {client_id if client_id else 'All clients'}
📂 Trades File: {trades_file_path}
🤖 Model: {model_name}
📋 Total Records Found: {len(trade_records)}

{'='*80}
💬 CONVERSATION SUMMARY
{'='*80}
{analysis_result.get('conversation_summary', 'N/A')}

{'='*80}
📈 OVERALL ASSESSMENT
{'='*80}
{analysis_result.get('overall_assessment', 'N/A')}

{'='*80}
📊 CONFIDENCE SUMMARY
{'='*80}
"""
        
        # Add summary statistics
        if 'total_confidence_summary' in analysis_result:
            summary = analysis_result['total_confidence_summary']
            for key, value in summary.items():
                output_text += f"{key}: {value}\n"
        
        output_text += f"\n{'='*80}\n"
        output_text += f"🔍 INDIVIDUAL RECORD ANALYSIS ({len(analysis_result.get('records_analyzed', []))} records)\n"
        output_text += f"{'='*80}\n\n"
        
        # Add individual record analyses
        for idx, record_analysis in enumerate(analysis_result.get('records_analyzed', []), 1):
            confidence = record_analysis.get('confidence_score', 0.0)
            
            # Determine emoji based on confidence
            if confidence >= 0.8:
                emoji = "✅"
            elif confidence >= 0.5:
                emoji = "⚠️"
            else:
                emoji = "❌"
            
            output_text += f"{emoji} Record #{idx} - OrderNo: {record_analysis.get('order_no', 'N/A')}\n"
            output_text += f"{'─'*80}\n"
            output_text += f"Confidence Score: {confidence:.2f} ({confidence*100:.0f}%)\n\n"
            output_text += f"Reasoning:\n{record_analysis.get('reasoning', 'N/A')}\n\n"
            
            matched_segments = record_analysis.get('matched_conversation_segments', [])
            if matched_segments:
                output_text += f"Matched Conversation Segments:\n"
                for segment in matched_segments:
                    output_text += f"  • {segment}\n"
                output_text += "\n"
            
            output_text += f"{'─'*80}\n\n"
        
        # Build JSON output
        json_output = {
            "status": "success",
            "analysis_info": {
                "date": target_date.date().isoformat(),
                "client_id": client_id,
                "trades_file": trades_file_path,
                "model": model_name,
                "total_records": len(trade_records),
                "combined_analysis_mode": is_combined_mode,
                "conversations_analyzed": len(conversation_list) if is_combined_mode and conversation_list else 1
            },
            "analysis_result": analysis_result,
            "trade_records": trade_records
        }
        
        # Add conversation list info if in combined mode
        if is_combined_mode and conversation_list:
            json_output["conversations_info"] = [
                {
                    "index": idx,
                    "filename": conv.get('filename', conv.get('metadata', {}).get('filename', f'Conversation #{idx}')),
                    "hkt_datetime": conv.get('metadata', {}).get('hkt_datetime', 'N/A')
                }
                for idx, conv in enumerate(conversation_list, 1)
            ]
        
        json_output_str = json.dumps(json_output, indent=2, ensure_ascii=False, default=str)
        
        return (output_text, json_output_str)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
        error_json = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        error_json_str = json.dumps(error_json, indent=2, ensure_ascii=False)
        return (error_msg, error_json_str)


def create_conversation_record_analysis_tab():
    """Create and return the Conversation Record Analysis tab"""
    with gr.Tab("🎯 Conversation Record Analysis"):
        gr.Markdown(
            """
            ### 對話記錄分析 - Analyze Trade Records Against Conversation
            
            This tool takes a conversation transcript and analyzes all trade records from that date to determine 
            which trades were actually discussed in the conversation.
            
            **How it works:**
            1. 📝 Paste your conversation JSON (must include `hkt_datetime`)
            2. 📅 Tool extracts the date and loads all trades from that day
            3. 🤖 LLM analyzes each trade record to determine confidence (0.0-1.0)
            4. 📊 Get detailed analysis with confidence scores and reasoning
            
            **Confidence Scores:**
            - `1.0`: Definitely mentioned with clear confirmation
            - `0.7-0.9`: Strong evidence, likely mentioned
            - `0.4-0.6`: Some evidence but not clear
            - `0.1-0.3`: Possibly related but very uncertain
            - `0.0`: Definitely NOT mentioned
            
            **Use Cases:**
            - Verify that executed trades were actually authorized in the call
            - Identify unauthorized or suspicious trades
            - Check compliance and audit trails
            - Analyze broker-client communication patterns
            
            **💡 New Feature: Combined Analysis**
            - When input is an array of conversations, enable "Combined Analysis" to analyze all conversations together
            - This is useful when conversations are related (same day, same client)
            - Context from conversation 1 can help analyze trades mentioned in conversation 2
            - Example: Stock mentioned in call #1 may be referred to as "that stock" in call #2
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📥 Input")
                
                conversation_json_box = gr.Textbox(
                    label="Conversation JSON",
                    placeholder='Single: {"metadata": {"hkt_datetime": "2025-10-20T10:15:30", ...}, "transcriptions": {...}}\n\nOR\n\nArray: [{"metadata": {...}, "transcriptions": {...}}, {...}]\n\n(Arrays: Enable Combined Analysis to analyze all together)',
                    lines=15,
                    info="Paste conversation JSON. Supports single object or array. Must include hkt_datetime in metadata."
                )
                
                combined_analysis_checkbox = gr.Checkbox(
                    label="🔗 Enable Combined Analysis",
                    value=False,
                    info="When input is an array, analyze ALL conversations together as one unified context (recommended for related conversations)"
                )
                
                gr.Markdown("#### ⚙️ Settings")
                
                trades_file_box = gr.Textbox(
                    label="Trades CSV File Path",
                    value="trades.csv",
                    placeholder="trades.csv",
                    info="Path to the trades CSV file"
                )
                
                client_id_box = gr.Textbox(
                    label="Client ID Filter (Optional)",
                    placeholder="e.g., P77197",
                    info="Filter trades by client ID (leave empty to analyze all clients on that date)"
                )
                
                model_dropdown = gr.Dropdown(
                    choices=MODEL_OPTIONS,
                    value=DEFAULT_MODEL,
                    label="LLM Model",
                    info="Select the LLM model for analysis"
                )
                
                ollama_url_box = gr.Textbox(
                    label="Ollama API URL",
                    value=DEFAULT_OLLAMA_URL,
                    placeholder="http://localhost:11434",
                    info="Ollama API endpoint"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="Temperature",
                    info="Lower = more focused, Higher = more creative"
                )
                
                analyze_btn = gr.Button(
                    "🎯 Analyze Records",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("#### 📊 Analysis Results")
                
                results_box = gr.Textbox(
                    label="Analysis Results (Formatted Text)",
                    lines=25,
                    interactive=False,
                    show_copy_button=True,
                )
                
                json_output_box = gr.Textbox(
                    label="Complete Analysis (JSON Format)",
                    lines=25,
                    interactive=False,
                    show_copy_button=True,
                    info="Complete analysis including all trade records and confidence scores"
                )
        
        # Connect the button
        analyze_btn.click(
            fn=analyze_conversation_records,
            inputs=[
                conversation_json_box,
                trades_file_box,
                client_id_box,
                model_dropdown,
                ollama_url_box,
                temperature_slider,
                combined_analysis_checkbox,
            ],
            outputs=[
                results_box,
                json_output_box,
            ],
        )

