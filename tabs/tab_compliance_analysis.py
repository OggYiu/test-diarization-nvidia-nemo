"""
Tab: Compliance Analysis
Analyze results from Trade Verification and Conversation Record Analysis
to determine broker compliance and need for human review
"""

import json
import csv
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import gradio as gr
import os
import logging
from langchain_ollama import ChatOllama
from model_config import DEFAULT_MODEL, DEFAULT_OLLAMA_URL


def parse_trade_verification_json(json_str: str) -> Optional[Dict]:
    """Parse trade verification JSON result"""
    try:
        data = json.loads(json_str)
        if data.get("status") == "error":
            return None
        return data
    except json.JSONDecodeError:
        return None


def parse_conversation_record_json(json_str: str) -> Optional[Dict]:
    """Parse conversation record analysis JSON result"""
    try:
        data = json.loads(json_str)
        if data.get("status") == "error":
            return None
        return data
    except json.JSONDecodeError:
        return None


def calculate_compliance_score(
    trade_verification: Dict,
    conversation_analysis: Dict
) -> Dict:
    """
    Calculate overall compliance score and determine if human review is needed
    
    Args:
        trade_verification: JSON from trade verification tab
        conversation_analysis: JSON from conversation record analysis tab
    
    Returns:
        Dict with compliance analysis results
    """
    result = {
        "overall_confidence": 0.0,
        "human_review_needed": False,
        "broker_compliance": "Unknown",
        "analysis_details": {},
        "recommendations": [],
        "risk_factors": []
    }
    
    # Extract transaction verifications from trade verification
    transaction_verifications = trade_verification.get("transaction_verifications", [])
    
    # Extract record analyses from conversation analysis
    records_analyzed = conversation_analysis.get("analysis_result", {}).get("records_analyzed", [])
    
    # Analysis metrics
    metrics = {
        "total_transactions": len(transaction_verifications),
        "total_records": len(records_analyzed),
        "high_confidence_matches": 0,
        "medium_confidence_matches": 0,
        "low_confidence_matches": 0,
        "unmatched_transactions": 0,
        "unmatched_records": 0,
        "confidence_scores": []
    }
    
    # Analyze transaction verifications (conversation → trade records)
    for tx_verification in transaction_verifications:
        best_match_confidence = tx_verification.get("verification_summary", {}).get("best_match_confidence", 0.0)
        
        # Normalize to 0-1 scale (trade verification uses 0-100)
        normalized_confidence = best_match_confidence / 100.0
        metrics["confidence_scores"].append(normalized_confidence)
        
        if normalized_confidence >= 0.7:
            metrics["high_confidence_matches"] += 1
        elif normalized_confidence >= 0.4:
            metrics["medium_confidence_matches"] += 1
        elif normalized_confidence > 0:
            metrics["low_confidence_matches"] += 1
        else:
            metrics["unmatched_transactions"] += 1
    
    # Analyze conversation records (trade records → conversation)
    for record_analysis in records_analyzed:
        confidence = record_analysis.get("confidence_score", 0.0)
        metrics["confidence_scores"].append(confidence)
        
        if confidence >= 0.7:
            metrics["high_confidence_matches"] += 1
        elif confidence >= 0.4:
            metrics["medium_confidence_matches"] += 1
        elif confidence > 0:
            metrics["low_confidence_matches"] += 1
        else:
            metrics["unmatched_records"] += 1
    
    # Calculate overall confidence score (0.0 - 1.0)
    if metrics["confidence_scores"]:
        average_confidence = sum(metrics["confidence_scores"]) / len(metrics["confidence_scores"])
        result["overall_confidence"] = round(average_confidence, 3)
    else:
        result["overall_confidence"] = 0.0
    
    # Determine broker compliance level
    if result["overall_confidence"] >= 0.8:
        result["broker_compliance"] = "✅ COMPLIANT - High Confidence"
        result["compliance_level"] = "high"
    elif result["overall_confidence"] >= 0.6:
        result["broker_compliance"] = "⚠️ LIKELY COMPLIANT - Medium Confidence"
        result["compliance_level"] = "medium"
    elif result["overall_confidence"] >= 0.4:
        result["broker_compliance"] = "⚠️ UNCLEAR - Low Confidence"
        result["compliance_level"] = "low"
    else:
        result["broker_compliance"] = "❌ POTENTIAL NON-COMPLIANCE - Very Low Confidence"
        result["compliance_level"] = "very_low"
    
    # Determine if human review is needed
    review_reasons = []
    
    if result["overall_confidence"] < 0.7:
        review_reasons.append(f"Overall confidence ({result['overall_confidence']:.1%}) is below threshold")
    
    if metrics["unmatched_transactions"] > 0:
        review_reasons.append(f"{metrics['unmatched_transactions']} transaction(s) found in conversation but not in trade records")
        result["risk_factors"].append("Potential undocumented trades mentioned")
    
    if metrics["unmatched_records"] > 0:
        review_reasons.append(f"{metrics['unmatched_records']} trade record(s) not clearly mentioned in conversation")
        result["risk_factors"].append("Potential unauthorized trades executed")
    
    if metrics["low_confidence_matches"] > metrics["high_confidence_matches"]:
        review_reasons.append("More low confidence matches than high confidence matches")
        result["risk_factors"].append("Inconsistent information quality")
    
    # Check for discrepancies between transaction analysis and record analysis
    discrepancy_count = abs(metrics["total_transactions"] - metrics["total_records"])
    if discrepancy_count > 0:
        review_reasons.append(f"Discrepancy: {metrics['total_transactions']} transactions vs {metrics['total_records']} records")
        result["risk_factors"].append("Mismatch between conversation and records count")
    
    result["human_review_needed"] = len(review_reasons) > 0
    result["review_reasons"] = review_reasons
    
    # Generate recommendations
    if result["human_review_needed"]:
        result["recommendations"].append("🎧 Listen to the full audio recording for verification")
        result["recommendations"].append("📝 Review transcription accuracy, especially for stock codes and quantities")
        
        if metrics["unmatched_transactions"] > 0:
            result["recommendations"].append("🔍 Investigate transactions mentioned in conversation but missing from trade records")
        
        if metrics["unmatched_records"] > 0:
            result["recommendations"].append("🔍 Verify if trade records were properly authorized by client")
        
        if result["overall_confidence"] < 0.5:
            result["recommendations"].append("⚠️ Consider escalating to compliance officer for investigation")
    else:
        result["recommendations"].append("✅ No immediate action required - Appears compliant")
        result["recommendations"].append("📊 Review statistics show good alignment between conversation and trades")
    
    result["analysis_details"] = metrics
    
    return result


def format_compliance_report(
    compliance_result: Dict,
    trade_verification: Dict,
    conversation_analysis: Dict
) -> str:
    """Format compliance analysis results as readable text"""
    
    report = f"""{'='*80}
🛡️ BROKER COMPLIANCE ANALYSIS REPORT
{'='*80}

📊 OVERALL ASSESSMENT
{'─'*80}
Confidence Score: {compliance_result['overall_confidence']:.1%} ({compliance_result['overall_confidence']:.3f})
Compliance Status: {compliance_result['broker_compliance']}
Human Review Required: {'✅ YES' if compliance_result['human_review_needed'] else '❌ NO'}

{'='*80}
📈 ANALYSIS METRICS
{'─'*80}
"""
    
    metrics = compliance_result.get("analysis_details", {})
    report += f"""Total Transactions Analyzed: {metrics.get('total_transactions', 0)}
Total Trade Records Analyzed: {metrics.get('total_records', 0)}

Confidence Distribution:
  ✅ High Confidence (≥70%): {metrics.get('high_confidence_matches', 0)}
  ⚠️ Medium Confidence (40-69%): {metrics.get('medium_confidence_matches', 0)}
  ❌ Low Confidence (<40%): {metrics.get('low_confidence_matches', 0)}

Unmatched Items:
  📝 Transactions without trade records: {metrics.get('unmatched_transactions', 0)}
  📋 Trade records not in conversation: {metrics.get('unmatched_records', 0)}

"""
    
    # Risk factors
    if compliance_result.get("risk_factors"):
        report += f"{'='*80}\n"
        report += f"⚠️ RISK FACTORS IDENTIFIED\n"
        report += f"{'─'*80}\n"
        for idx, risk in enumerate(compliance_result["risk_factors"], 1):
            report += f"{idx}. {risk}\n"
        report += "\n"
    
    # Review reasons
    if compliance_result.get("human_review_needed"):
        report += f"{'='*80}\n"
        report += f"🔍 REASONS FOR HUMAN REVIEW\n"
        report += f"{'─'*80}\n"
        for idx, reason in enumerate(compliance_result.get("review_reasons", []), 1):
            report += f"{idx}. {reason}\n"
        report += "\n"
    
    # Recommendations
    if compliance_result.get("recommendations"):
        report += f"{'='*80}\n"
        report += f"💡 RECOMMENDATIONS\n"
        report += f"{'─'*80}\n"
        for idx, rec in enumerate(compliance_result["recommendations"], 1):
            report += f"{idx}. {rec}\n"
        report += "\n"
    
    # Summary from verification info
    if "verification_info" in trade_verification:
        info = trade_verification["verification_info"]
        report += f"{'='*80}\n"
        report += f"📋 VERIFICATION CONTEXT\n"
        report += f"{'─'*80}\n"
        report += f"Client ID: {info.get('client_id', 'N/A')}\n"
        report += f"Client Name: {info.get('client_name', 'N/A')}\n"
        report += f"Broker ID: {info.get('broker_id', 'N/A')}\n"
        report += f"Broker Name: {info.get('broker_name', 'N/A')}\n"
        report += f"Date/Time: {info.get('hkt_datetime', 'N/A')}\n"
        report += f"Conversation Number: {info.get('conversation_number', 'N/A')}\n"
        report += "\n"
    
    report += f"{'='*80}\n"
    report += f"✅ COMPLIANCE ANALYSIS COMPLETE\n"
    report += f"{'='*80}\n"
    
    return report


def save_to_compliance_csv(
    compliance_result: Dict,
    trade_verification: Dict,
    conversation_analysis: Dict,
    compliance_file: str = "compliance.csv"
) -> str:
    """
    Save compliance analysis results to CSV file
    
    Args:
        compliance_result: Compliance analysis results
        trade_verification: Trade verification JSON
        conversation_analysis: Conversation record analysis JSON
        compliance_file: Output CSV file path
    
    Returns:
        Status message
    """
    try:
        # Extract verification info
        info = trade_verification.get("verification_info", {})
        client_id = info.get("client_id", "N/A")
        broker_id = info.get("broker_id", "N/A")
        hkt_datetime = info.get("hkt_datetime", "N/A")
        
        # Extract analysis info
        analysis_info = conversation_analysis.get("analysis_info", {})
        model_used = analysis_info.get("model", "N/A")
        
        metrics = compliance_result.get("analysis_details", {})
        
        # Prepare new row
        new_row = {
            "client_id": client_id,
            "broker_id": broker_id,
            "hkt_datetime": hkt_datetime,
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_confidence": f"{compliance_result['overall_confidence']:.3f}",
            "compliance_status": compliance_result["compliance_level"],
            "human_review_needed": "Yes" if compliance_result["human_review_needed"] else "No",
            "total_transactions": metrics.get("total_transactions", 0),
            "total_records": metrics.get("total_records", 0),
            "high_confidence_matches": metrics.get("high_confidence_matches", 0),
            "medium_confidence_matches": metrics.get("medium_confidence_matches", 0),
            "low_confidence_matches": metrics.get("low_confidence_matches", 0),
            "unmatched_transactions": metrics.get("unmatched_transactions", 0),
            "unmatched_records": metrics.get("unmatched_records", 0),
            "risk_factors": "; ".join(compliance_result.get("risk_factors", [])),
            "review_reasons": "; ".join(compliance_result.get("review_reasons", [])),
            "model_used": model_used,
        }
        
        # Define CSV columns
        fieldnames = [
            "client_id",
            "broker_id",
            "hkt_datetime",
            "analysis_timestamp",
            "overall_confidence",
            "compliance_status",
            "human_review_needed",
            "total_transactions",
            "total_records",
            "high_confidence_matches",
            "medium_confidence_matches",
            "low_confidence_matches",
            "unmatched_transactions",
            "unmatched_records",
            "risk_factors",
            "review_reasons",
            "model_used",
        ]
        
        # Read existing records if file exists
        existing_rows = []
        file_exists = os.path.exists(compliance_file)
        
        if file_exists:
            try:
                with open(compliance_file, 'r', encoding='utf-8-sig', newline='') as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)
            except Exception as e:
                logging.warning(f"Could not read existing {compliance_file}: {e}")
        
        # Check for duplicates based on (client_id, broker_id, hkt_datetime)
        key = (client_id, broker_id, hkt_datetime)
        deleted_count = 0
        filtered_rows = []
        
        for row in existing_rows:
            existing_key = (
                row.get("client_id", ""),
                row.get("broker_id", ""),
                row.get("hkt_datetime", "")
            )
            if existing_key == key:
                deleted_count += 1
            else:
                filtered_rows.append(row)
        
        # Add new row
        final_rows = filtered_rows + [new_row]
        
        # Write to file with UTF-8-BOM
        try:
            with open(compliance_file, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(final_rows)
        except UnicodeEncodeError:
            logging.warning("UTF-8 encoding failed, trying GB2312")
            with open(compliance_file, 'w', encoding='gb2312', newline='', errors='ignore') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(final_rows)
        
        # Build status message
        status_msg = f"✅ Compliance report saved to {compliance_file} (UTF-8 encoding)\n"
        status_msg += f"   📝 Added: 1 new record\n"
        if deleted_count > 0:
            status_msg += f"   🗑️ Deleted: {deleted_count} old record(s) with same client_id + broker_id + datetime\n"
        status_msg += f"   📊 Total records in file: {len(final_rows)}"
        
        return status_msg
        
    except Exception as e:
        error_msg = f"❌ Error saving to {compliance_file}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return error_msg


def analyze_with_llm(
    compliance_report: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL
) -> str:
    """
    Use LLM to analyze the compliance report and provide insights
    
    Args:
        compliance_report: The formatted compliance report text
        model: Ollama model to use
        ollama_url: Ollama server URL
    
    Returns:
        LLM analysis result as string
    """
    try:
        # Initialize LLM
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=0.3,  # Lower temperature for more focused analysis
        )
        
        # Create the analysis prompt
        system_message = """You are a compliance analyst expert specializing in financial regulations and broker-client interactions. 
Your task is to analyze the compliance report and provide:
1. A brief executive summary (2-3 sentences)
2. Key risks and concerns identified
3. Specific recommendations for compliance officers
4. Any red flags that require immediate attention

Please provide your analysis in a clear, structured format using Traditional Chinese (繁體中文) and English where appropriate."""
        
        user_prompt = f"""Please analyze the following broker compliance report and provide your expert insights:

{compliance_report}

Please provide your analysis with:
1. 執行摘要 (Executive Summary)
2. 主要風險與疑慮 (Key Risks & Concerns)
3. 合規建議 (Compliance Recommendations)
4. 需立即關注事項 (Immediate Attention Required)
"""
        
        messages = [
            ("system", system_message),
            ("human", user_prompt),
        ]
        
        # Call LLM
        resp = chat_llm.invoke(messages)
        analysis_result = getattr(resp, "content", str(resp))
        
        # Format the result with header
        formatted_result = f"""{'='*80}
🤖 LLM COMPLIANCE ANALYSIS
{'='*80}
Model: {model}
Analysis Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'─'*80}

{analysis_result}

{'='*80}
✅ LLM ANALYSIS COMPLETE
{'='*80}
"""
        
        return formatted_result
        
    except Exception as e:
        error_msg = f"""{'='*80}
❌ LLM ANALYSIS ERROR
{'='*80}

Error: {str(e)}

Details:
{traceback.format_exc()}

{'='*80}
"""
        logging.error(f"LLM analysis error: {e}")
        return error_msg


def analyze_compliance(
    trade_verification_json: str,
    conversation_analysis_json: str,
    use_llm: bool = True,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL
) -> Tuple[str, str, str, str]:
    """
    Main function to perform compliance analysis
    
    Args:
        trade_verification_json: JSON from trade verification tab
        conversation_analysis_json: JSON from conversation record analysis tab
        use_llm: Whether to use LLM for additional analysis
        model: Ollama model to use for LLM analysis
        ollama_url: Ollama server URL
    
    Returns:
        Tuple of (formatted_report, json_result, csv_save_status, llm_analysis)
    """
    try:
        # Validate inputs
        if not trade_verification_json.strip():
            error_msg = "❌ Error: Please provide Trade Verification JSON (run Trade Verification tab first)"
            return (error_msg, "", "", "")
        
        if not conversation_analysis_json.strip():
            error_msg = "❌ Error: Please provide Conversation Record Analysis JSON (run Conversation Record Analysis tab first)"
            return (error_msg, "", "", "")
        
        # Parse JSONs
        trade_verification = parse_trade_verification_json(trade_verification_json)
        if not trade_verification:
            error_msg = "❌ Error: Cannot parse Trade Verification JSON or it contains an error status"
            return (error_msg, "", "", "")
        
        conversation_analysis = parse_conversation_record_json(conversation_analysis_json)
        if not conversation_analysis:
            error_msg = "❌ Error: Cannot parse Conversation Record Analysis JSON or it contains an error status"
            return (error_msg, "", "", "")
        
        # Perform compliance analysis
        compliance_result = calculate_compliance_score(
            trade_verification,
            conversation_analysis
        )
        
        # Format report
        formatted_report = format_compliance_report(
            compliance_result,
            trade_verification,
            conversation_analysis
        )
        
        # Build JSON output
        json_output = {
            "status": "success",
            "compliance_analysis": compliance_result,
            "input_summaries": {
                "trade_verification_summary": {
                    "total_transactions": len(trade_verification.get("transaction_verifications", [])),
                    "verification_info": trade_verification.get("verification_info", {})
                },
                "conversation_analysis_summary": {
                    "total_records": len(conversation_analysis.get("analysis_result", {}).get("records_analyzed", [])),
                    "analysis_info": conversation_analysis.get("analysis_info", {})
                }
            }
        }
        
        json_output_str = json.dumps(json_output, indent=2, ensure_ascii=False)
        
        # Save to CSV
        csv_status = save_to_compliance_csv(
            compliance_result,
            trade_verification,
            conversation_analysis,
            "compliance.csv"
        )
        
        # Perform LLM analysis if enabled
        llm_analysis = ""
        if use_llm:
            llm_analysis = analyze_with_llm(formatted_report, model, ollama_url)
        else:
            llm_analysis = "LLM analysis is disabled. Enable the checkbox to use LLM analysis."
        
        return (formatted_report, json_output_str, csv_status, llm_analysis)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
        error_json = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        error_json_str = json.dumps(error_json, indent=2, ensure_ascii=False)
        return (error_msg, error_json_str, "", "")


def create_compliance_analysis_tab(
    trade_verification_state=None,
    conversation_analysis_state=None
):
    """
    Create and return the Compliance Analysis tab
    
    Args:
        trade_verification_state: Optional gr.State for trade verification JSON
        conversation_analysis_state: Optional gr.State for conversation analysis JSON
    """
    with gr.Tab("🛡️ Compliance Analysis"):
        gr.Markdown("""
        ### Broker Compliance Analysis
        
        This tab analyzes results from **Trade Verification** and **Conversation Record Analysis** tabs to:
        - Provide an overall confidence score (0.0 to 1.0)
        - Determine if human review is needed
        - Assess if the broker performed their duties according to compliance requirements
        
        **How it works:**
        1. Load JSON results from both previous tabs
        2. Compare conversation-to-trades and trades-to-conversation analyses
        3. Calculate overall confidence and identify discrepancies
        4. Provide compliance assessment and recommendations
        5. Optional: Use LLM to provide additional expert analysis
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📥 Input Data")
                
                # Load buttons if states are provided
                if trade_verification_state is not None:
                    load_trade_verification_btn = gr.Button(
                        "📥 Load Trade Verification Results",
                        variant="secondary",
                        size="sm"
                    )
                
                trade_verification_json_box = gr.Textbox(
                    label="Trade Verification JSON",
                    placeholder="Paste JSON output from Trade Verification tab...",
                    lines=15,
                    info="JSON output from 🔍 Trade Verification tab"
                )
                
                if conversation_analysis_state is not None:
                    load_conversation_analysis_btn = gr.Button(
                        "📥 Load Conversation Record Analysis Results",
                        variant="secondary",
                        size="sm"
                    )
                
                conversation_analysis_json_box = gr.Textbox(
                    label="Conversation Record Analysis JSON",
                    placeholder="Paste JSON output from Conversation Record Analysis tab...",
                    lines=15,
                    info="JSON output from 🎯 Conversation Record Analysis tab"
                )
                
                # LLM settings
                gr.Markdown("#### 🤖 LLM Analysis Settings")
                
                use_llm_checkbox = gr.Checkbox(
                    label="🤖 Enable LLM Analysis",
                    value=True,
                    info="Use LLM to provide additional expert analysis of the compliance report"
                )
                
                with gr.Row():
                    llm_model_input = gr.Textbox(
                        label="LLM Model",
                        value=DEFAULT_MODEL,
                        placeholder="e.g., qwen3:32b",
                        info="Ollama model name"
                    )
                    
                    ollama_url_input = gr.Textbox(
                        label="Ollama URL",
                        value=DEFAULT_OLLAMA_URL,
                        placeholder="e.g., http://localhost:11434",
                        info="Ollama server URL"
                    )
                
                analyze_btn = gr.Button(
                    "🛡️ Analyze Compliance",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("#### 📊 Compliance Analysis Results")
                
                results_box = gr.Textbox(
                    label="Compliance Report (Formatted Text)",
                    lines=20,
                    interactive=False,
                    show_copy_button=True,
                )
                
                llm_analysis_box = gr.Textbox(
                    label="🤖 LLM Expert Analysis",
                    lines=20,
                    interactive=False,
                    show_copy_button=True,
                    info="AI-powered expert analysis of the compliance report"
                )
                
                csv_status_box = gr.Textbox(
                    label="📊 Compliance.csv Save Status",
                    lines=3,
                    interactive=False,
                    info="Status of saving compliance results to compliance.csv"
                )
                
                json_output_box = gr.Textbox(
                    label="Complete Analysis (JSON Format)",
                    lines=15,
                    interactive=False,
                    show_copy_button=True,
                    info="Complete compliance analysis in JSON format"
                )
        
        # Connect the analyze button
        analyze_btn.click(
            fn=analyze_compliance,
            inputs=[
                trade_verification_json_box,
                conversation_analysis_json_box,
                use_llm_checkbox,
                llm_model_input,
                ollama_url_input,
            ],
            outputs=[
                results_box,
                json_output_box,
                csv_status_box,
                llm_analysis_box,
            ],
        )
        
        # Connect load buttons if states are provided
        if trade_verification_state is not None:
            def load_trade_verification_from_state(json_data):
                """Load trade verification JSON from shared state"""
                if json_data:
                    return json_data
                return "⚠️ No data from Trade Verification tab. Please run that tab first."
            
            load_trade_verification_btn.click(
                fn=load_trade_verification_from_state,
                inputs=[trade_verification_state],
                outputs=[trade_verification_json_box]
            )
        
        if conversation_analysis_state is not None:
            def load_conversation_analysis_from_state(json_data):
                """Load conversation analysis JSON from shared state"""
                if json_data:
                    return json_data
                return "⚠️ No data from Conversation Record Analysis tab. Please run that tab first."
            
            load_conversation_analysis_btn.click(
                fn=load_conversation_analysis_from_state,
                inputs=[conversation_analysis_state],
                outputs=[conversation_analysis_json_box]
            )

