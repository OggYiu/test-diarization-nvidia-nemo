"""
Tab: Full Pipeline
Chains all existing tab functions together to produce complete analysis from audio file to compliance report.

Pipeline flow:
1. STT (Speech-to-Text) → Conversation JSON
2. JSON Batch Analysis → Merged Stocks JSON
3. Transaction Analysis JSON → Transaction JSON
4. Trade Verification → Verification JSON
5. Conversation Record Analysis → Analysis JSON
6. Compliance Analysis → Final Compliance Report
"""

import gradio as gr
import json
import traceback
import logging
from typing import Tuple

# Import processing functions from other tabs
# Note: We import the core processing functions, not the tab creation functions
from tabs.tab_stt import process_audio_or_folder
from tabs.tab_json_batch_analysis import process_json_batch
from tabs.tab_transaction_analysis_json import (
    analyze_transactions_with_json,
    DEFAULT_SYSTEM_MESSAGE as TRANSACTION_DEFAULT_SYSTEM_MESSAGE
)
from tabs.tab_trade_verification import verify_transactions
from tabs.tab_conversation_record_analysis import analyze_conversation_records
from tabs.tab_compliance_analysis import analyze_compliance

# Import configurations
from model_config import DEFAULT_OLLAMA_URL
from tabs.tab_stt_stock_comparison import LLM_OPTIONS, DEFAULT_SYSTEM_MESSAGE as STOCK_EXTRACTION_SYSTEM_MESSAGE


# ============================================================================
# Main Pipeline Function
# ============================================================================

def run_full_pipeline(
    # STT parameters
    audio_input,
    language: str,
    use_sensevoice: bool,
    use_whisperv3_cantonese: bool,
    overwrite_diarization: bool,
    padding_ms: int,
    use_enhanced_format: bool,
    apply_corrections: bool,
    correction_json: str,
    vad_model: str,
    max_single_segment_time: int,
    
    # JSON Batch Analysis parameters
    selected_llms: list,
    system_message: str,
    ollama_url: str,
    temperature: float,
    use_vector_correction: bool,
    use_contextual_analysis: bool,
    enable_stock_verification: bool,
    verification_llm: str,
    
    # Transaction Analysis parameters
    transaction_model: str,
    transaction_system_message: str,
    transaction_temperature: float,
    
    # Trade Verification parameters
    trades_file_path: str,
    time_window: float,
    
    # Conversation Record Analysis parameters
    client_id_filter: str,
    record_analysis_model: str,
    record_analysis_temperature: float,
    use_combined_analysis: bool,
    
    # Compliance Analysis parameters
    use_llm_compliance: bool,
    compliance_model: str,
    
    progress=gr.Progress()
) -> Tuple[str, str, str, str, str, str, str]:
    """
    Run the complete pipeline from audio file to compliance report.
    
    Returns:
        Tuple of:
        - Pipeline status/log
        - Conversation JSON
        - Merged Stocks JSON
        - Transaction JSON
        - Verification JSON
        - Analysis JSON
        - Final Compliance Report
    """
    
    log_parts = []
    log_parts.append("=" * 80)
    log_parts.append("🚀 STARTING FULL PIPELINE")
    log_parts.append("=" * 80)
    log_parts.append("")
    
    try:
        # ====================================================================
        # STEP 1: Speech-to-Text (STT)
        # ====================================================================
        progress(0.0, desc="Step 1/6: Running Speech-to-Text...")
        log_parts.append("📝 STEP 1/6: Speech-to-Text Processing")
        log_parts.append("-" * 80)
        
        if audio_input is None:
            error_msg = "❌ Error: No audio file provided"
            log_parts.append(error_msg)
            return ("\n".join(log_parts), "", "", "", "", "", "")
        
        if not use_sensevoice and not use_whisperv3_cantonese:
            error_msg = "❌ Error: Please select at least one STT model"
            log_parts.append(error_msg)
            return ("\n".join(log_parts), "", "", "", "", "", "")
        
        try:
            stt_result = process_audio_or_folder(
                audio_input,
                language,
                use_sensevoice,
                use_whisperv3_cantonese,
                overwrite_diarization,
                padding_ms,
                use_enhanced_format,
                apply_corrections,
                correction_json,
                vad_model,
                max_single_segment_time,
                progress
            )
            
            # Extract conversation JSON (last item in the tuple)
            conversation_json = stt_result[-1]
            
            if not conversation_json or conversation_json.strip() == "":
                error_msg = "❌ Error: STT failed to produce conversation JSON"
                log_parts.append(error_msg)
                return ("\n".join(log_parts), "", "", "", "", "", "")
            
            log_parts.append("✅ STT completed successfully")
            log_parts.append(f"   Generated conversation JSON ({len(conversation_json)} chars)")
            log_parts.append("")
            
        except Exception as e:
            error_msg = f"❌ Error in STT: {str(e)}\n{traceback.format_exc()}"
            log_parts.append(error_msg)
            logging.error(error_msg)
            return ("\n".join(log_parts), "", "", "", "", "", "")
        
        # ====================================================================
        # STEP 2: JSON Batch Analysis (Stock Extraction)
        # ====================================================================
        progress(0.17, desc="Step 2/6: Analyzing stocks with LLM...")
        log_parts.append("🔍 STEP 2/6: JSON Batch Analysis (Stock Extraction)")
        log_parts.append("-" * 80)
        
        try:
            batch_result = process_json_batch(
                conversation_json,
                selected_llms,
                system_message,
                ollama_url,
                temperature,
                use_vector_correction,
                use_contextual_analysis,
                enable_stock_verification,
                verification_llm,
            )
            
            # Extract merged stocks JSON (3rd item in the tuple)
            # Result: (formatted_results, combined_json, merged_json, verification_results)
            merged_stocks_json = batch_result[2]
            
            if not merged_stocks_json or merged_stocks_json.strip() == "":
                error_msg = "❌ Error: JSON Batch Analysis failed to produce merged stocks"
                log_parts.append(error_msg)
                return ("\n".join(log_parts), conversation_json, "", "", "", "", "")
            
            log_parts.append("✅ JSON Batch Analysis completed successfully")
            log_parts.append(f"   Generated merged stocks JSON ({len(merged_stocks_json)} chars)")
            log_parts.append("")
            
        except Exception as e:
            error_msg = f"❌ Error in JSON Batch Analysis: {str(e)}\n{traceback.format_exc()}"
            log_parts.append(error_msg)
            logging.error(error_msg)
            return ("\n".join(log_parts), conversation_json, "", "", "", "", "")
        
        # ====================================================================
        # STEP 3: Transaction Analysis JSON
        # ====================================================================
        progress(0.34, desc="Step 3/6: Analyzing transactions...")
        log_parts.append("💰 STEP 3/6: Transaction Analysis")
        log_parts.append("-" * 80)
        
        try:
            transaction_result = analyze_transactions_with_json(
                conversation_json,
                merged_stocks_json,
                transaction_model,
                ollama_url,
                transaction_system_message,
                transaction_temperature,
            )
            
            # Extract transaction JSON (2nd item in the tuple)
            # Result: (summary_result, json_result)
            transaction_json = transaction_result[1]
            
            if not transaction_json or transaction_json.strip() == "":
                error_msg = "❌ Error: Transaction Analysis failed to produce transaction JSON"
                log_parts.append(error_msg)
                return ("\n".join(log_parts), conversation_json, merged_stocks_json, "", "", "", "")
            
            log_parts.append("✅ Transaction Analysis completed successfully")
            log_parts.append(f"   Generated transaction JSON ({len(transaction_json)} chars)")
            log_parts.append("")
            
        except Exception as e:
            error_msg = f"❌ Error in Transaction Analysis: {str(e)}\n{traceback.format_exc()}"
            log_parts.append(error_msg)
            logging.error(error_msg)
            return ("\n".join(log_parts), conversation_json, merged_stocks_json, "", "", "", "")
        
        # ====================================================================
        # STEP 4: Trade Verification
        # ====================================================================
        progress(0.51, desc="Step 4/6: Verifying trades...")
        log_parts.append("🔎 STEP 4/6: Trade Verification")
        log_parts.append("-" * 80)
        
        try:
            verification_result = verify_transactions(
                transaction_json,
                trades_file_path,
                time_window,
            )
            
            # Extract verification JSON (2nd item in the tuple)
            # Result: (formatted_text, json_analysis, csv_records, all_client_records, report_status)
            verification_json = verification_result[1]
            
            if not verification_json or verification_json.strip() == "":
                error_msg = "❌ Error: Trade Verification failed to produce verification JSON"
                log_parts.append(error_msg)
                return ("\n".join(log_parts), conversation_json, merged_stocks_json, transaction_json, "", "", "")
            
            log_parts.append("✅ Trade Verification completed successfully")
            log_parts.append(f"   Generated verification JSON ({len(verification_json)} chars)")
            log_parts.append("")
            
        except Exception as e:
            error_msg = f"❌ Error in Trade Verification: {str(e)}\n{traceback.format_exc()}"
            log_parts.append(error_msg)
            logging.error(error_msg)
            return ("\n".join(log_parts), conversation_json, merged_stocks_json, transaction_json, "", "", "")
        
        # ====================================================================
        # STEP 5: Conversation Record Analysis
        # ====================================================================
        progress(0.68, desc="Step 5/6: Analyzing conversation records...")
        log_parts.append("🎯 STEP 5/6: Conversation Record Analysis")
        log_parts.append("-" * 80)
        
        try:
            record_analysis_result = analyze_conversation_records(
                conversation_json,
                trades_file_path,
                client_id_filter,
                record_analysis_model,
                ollama_url,
                record_analysis_temperature,
                use_combined_analysis,
            )
            
            # Extract analysis JSON (2nd item in the tuple)
            # Result: (formatted_text_result, json_result, csv_save_status)
            analysis_json = record_analysis_result[1]
            
            if not analysis_json or analysis_json.strip() == "":
                error_msg = "❌ Error: Conversation Record Analysis failed to produce analysis JSON"
                log_parts.append(error_msg)
                return ("\n".join(log_parts), conversation_json, merged_stocks_json, transaction_json, verification_json, "", "")
            
            log_parts.append("✅ Conversation Record Analysis completed successfully")
            log_parts.append(f"   Generated analysis JSON ({len(analysis_json)} chars)")
            log_parts.append("")
            
        except Exception as e:
            error_msg = f"❌ Error in Conversation Record Analysis: {str(e)}\n{traceback.format_exc()}"
            log_parts.append(error_msg)
            logging.error(error_msg)
            return ("\n".join(log_parts), conversation_json, merged_stocks_json, transaction_json, verification_json, "", "")
        
        # ====================================================================
        # STEP 6: Compliance Analysis (Final Step)
        # ====================================================================
        progress(0.85, desc="Step 6/6: Running compliance analysis...")
        log_parts.append("🛡️ STEP 6/6: Compliance Analysis (Final)")
        log_parts.append("-" * 80)
        
        try:
            compliance_result = analyze_compliance(
                verification_json,
                analysis_json,
                use_llm_compliance,
                compliance_model,
                ollama_url,
            )
            
            # Extract compliance report (1st item in the tuple)
            # Result: (formatted_report, json_result, csv_save_status, llm_analysis)
            compliance_report = compliance_result[0]
            
            if not compliance_report or compliance_report.strip() == "":
                error_msg = "❌ Error: Compliance Analysis failed to produce compliance report"
                log_parts.append(error_msg)
                return ("\n".join(log_parts), conversation_json, merged_stocks_json, transaction_json, verification_json, analysis_json, "")
            
            log_parts.append("✅ Compliance Analysis completed successfully")
            log_parts.append(f"   Generated compliance report ({len(compliance_report)} chars)")
            log_parts.append("")
            
        except Exception as e:
            error_msg = f"❌ Error in Compliance Analysis: {str(e)}\n{traceback.format_exc()}"
            log_parts.append(error_msg)
            logging.error(error_msg)
            return ("\n".join(log_parts), conversation_json, merged_stocks_json, transaction_json, verification_json, analysis_json, "")
        
        # ====================================================================
        # PIPELINE COMPLETE
        # ====================================================================
        progress(1.0, desc="Pipeline complete!")
        log_parts.append("=" * 80)
        log_parts.append("✅ FULL PIPELINE COMPLETED SUCCESSFULLY")
        log_parts.append("=" * 80)
        log_parts.append("")
        log_parts.append("📊 All 6 steps completed:")
        log_parts.append("   1. ✅ Speech-to-Text")
        log_parts.append("   2. ✅ JSON Batch Analysis (Stock Extraction)")
        log_parts.append("   3. ✅ Transaction Analysis")
        log_parts.append("   4. ✅ Trade Verification")
        log_parts.append("   5. ✅ Conversation Record Analysis")
        log_parts.append("   6. ✅ Compliance Analysis")
        log_parts.append("")
        log_parts.append("🎉 Final compliance report generated successfully!")
        
        return (
            "\n".join(log_parts),
            conversation_json,
            merged_stocks_json,
            transaction_json,
            verification_json,
            analysis_json,
            compliance_report
        )
        
    except Exception as e:
        error_msg = f"❌ Unexpected error in pipeline: {str(e)}\n{traceback.format_exc()}"
        log_parts.append(error_msg)
        logging.error(error_msg)
        return ("\n".join(log_parts), "", "", "", "", "", "")


# ============================================================================
# Gradio Tab Creation
# ============================================================================

def create_full_pipeline_tab():
    """Create and return the Full Pipeline tab"""
    
    with gr.Tab("🔗 Full Pipeline"):
        gr.Markdown("""
        ### 🚀 Complete Analysis Pipeline
        
        This tab automatically chains all processing steps together:
        1. **Speech-to-Text** → Transcribe audio and identify speakers
        2. **Stock Extraction** → Extract stocks mentioned using LLM
        3. **Transaction Analysis** → Identify potential trades
        4. **Trade Verification** → Match against actual trade records
        5. **Conversation Analysis** → Analyze if trades were discussed
        6. **Compliance Analysis** → Final compliance report
        
        Simply upload your audio file, configure the settings, and click **Run Full Pipeline**.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # ============================================================
                # STEP 1: STT Settings
                # ============================================================
                gr.Markdown("#### 📝 Step 1: Speech-to-Text Settings")
                
                audio_input = gr.File(
                    label="Audio File(s) or Folder",
                    file_count="multiple",
                    file_types=["audio"]
                )
                
                with gr.Row():
                    use_sensevoice = gr.Checkbox(
                        label="Use SenseVoice",
                        value=True,
                        info="Recommended for Cantonese"
                    )
                    use_whisperv3_cantonese = gr.Checkbox(
                        label="Use Whisper v3 Cantonese",
                        value=False,
                    )
                
                language = gr.Dropdown(
                    choices=["auto", "zh", "yue", "en"],
                    value="auto",
                    label="Language",
                )
                
                with gr.Accordion("Advanced STT Settings", open=False):
                    overwrite_diarization = gr.Checkbox(
                        label="Overwrite Diarization Cache",
                        value=False,
                    )
                    padding_ms = gr.Slider(
                        minimum=0,
                        maximum=1000,
                        value=100,
                        step=10,
                        label="Padding (ms)",
                    )
                    use_enhanced_format = gr.Checkbox(
                        label="Use Enhanced Format",
                        value=True,
                    )
                    apply_corrections = gr.Checkbox(
                        label="Apply Text Corrections",
                        value=False,
                    )
                    correction_json = gr.Textbox(
                        label="Correction JSON",
                        visible=False,
                        lines=3,
                    )
                    vad_model = gr.Dropdown(
                        choices=["silero-vad", "fsmn-vad"],
                        value="fsmn-vad",
                        label="VAD Model",
                    )
                    max_single_segment_time = gr.Slider(
                        minimum=10000,
                        maximum=60000,
                        value=30000,
                        step=1000,
                        label="Max Single Segment Time (ms)",
                    )
                
                # ============================================================
                # STEP 2: JSON Batch Analysis Settings
                # ============================================================
                gr.Markdown("#### 🔍 Step 2: Stock Extraction Settings")
                
                selected_llms = gr.CheckboxGroup(
                    choices=LLM_OPTIONS,
                    value=[LLM_OPTIONS[0]],
                    label="Select LLMs for Stock Extraction",
                )
                
                with gr.Accordion("Advanced Stock Extraction Settings", open=False):
                    system_message = gr.Textbox(
                        label="System Message",
                        value=STOCK_EXTRACTION_SYSTEM_MESSAGE,
                        lines=5,
                    )
                    use_vector_correction = gr.Checkbox(
                        label="Use Vector Correction",
                        value=True,
                    )
                    use_contextual_analysis = gr.Checkbox(
                        label="Use Contextual Analysis",
                        value=True,
                    )
                    enable_stock_verification = gr.Checkbox(
                        label="Enable Stock Verification",
                        value=False,
                    )
                    verification_llm = gr.Dropdown(
                        choices=LLM_OPTIONS,
                        value=LLM_OPTIONS[0],
                        label="Verification LLM",
                    )
                
                # ============================================================
                # STEP 3: Transaction Analysis Settings
                # ============================================================
                gr.Markdown("#### 💰 Step 3: Transaction Analysis Settings")
                
                transaction_model = gr.Dropdown(
                    choices=LLM_OPTIONS,
                    value=LLM_OPTIONS[0],
                    label="Transaction Analysis Model",
                )
                
                with gr.Accordion("Advanced Transaction Settings", open=False):
                    transaction_system_message = gr.Textbox(
                        label="System Message",
                        value=TRANSACTION_DEFAULT_SYSTEM_MESSAGE,
                        lines=5,
                    )
                    transaction_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Temperature",
                    )
                
                # ============================================================
                # STEP 4: Trade Verification Settings
                # ============================================================
                gr.Markdown("#### 🔎 Step 4: Trade Verification Settings")
                
                trades_file_path = gr.Textbox(
                    label="Trades CSV File Path",
                    value="trades.csv",
                )
                
                time_window = gr.Slider(
                    minimum=0.1,
                    maximum=24.0,
                    value=1.0,
                    step=0.1,
                    label="Time Window (hours)",
                )
                
                # ============================================================
                # STEP 5: Conversation Record Analysis Settings
                # ============================================================
                gr.Markdown("#### 🎯 Step 5: Record Analysis Settings")
                
                client_id_filter = gr.Textbox(
                    label="Client ID Filter (optional)",
                    value="",
                )
                
                record_analysis_model = gr.Dropdown(
                    choices=LLM_OPTIONS,
                    value=LLM_OPTIONS[0],
                    label="Record Analysis Model",
                )
                
                with gr.Accordion("Advanced Record Analysis Settings", open=False):
                    record_analysis_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Temperature",
                    )
                    use_combined_analysis = gr.Checkbox(
                        label="Use Combined Analysis",
                        value=True,
                    )
                
                # ============================================================
                # STEP 6: Compliance Analysis Settings
                # ============================================================
                gr.Markdown("#### 🛡️ Step 6: Compliance Settings")
                
                use_llm_compliance = gr.Checkbox(
                    label="Use LLM for Compliance Analysis",
                    value=True,
                )
                
                compliance_model = gr.Dropdown(
                    choices=LLM_OPTIONS,
                    value=LLM_OPTIONS[0],
                    label="Compliance Analysis Model",
                )
                
                # ============================================================
                # Global Settings
                # ============================================================
                gr.Markdown("#### ⚙️ Global Settings")
                
                ollama_url = gr.Textbox(
                    label="Ollama URL",
                    value=DEFAULT_OLLAMA_URL,
                )
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="Default Temperature",
                )
                
                # ============================================================
                # Action Button
                # ============================================================
                run_button = gr.Button("🚀 Run Full Pipeline", variant="primary", size="lg")
            
            # ============================================================
            # RIGHT COLUMN: Results
            # ============================================================
            with gr.Column(scale=1):
                gr.Markdown("#### 📊 Pipeline Results")
                
                pipeline_log = gr.Textbox(
                    label="Pipeline Status Log",
                    lines=15,
                    interactive=False,
                    show_copy_button=True,
                    info="Real-time status of each pipeline step"
                )
                
                gr.Markdown("#### 🎉 Final Results")
                
                with gr.Tabs():
                    with gr.Tab("Compliance Report"):
                        compliance_output = gr.Textbox(
                            label="Final Compliance Report",
                            lines=20,
                            interactive=False,
                            show_copy_button=True,
                        )
                    
                    with gr.Tab("Conversation JSON"):
                        conversation_output = gr.Textbox(
                            label="Conversation JSON (Step 1)",
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )
                    
                    with gr.Tab("Stocks JSON"):
                        stocks_output = gr.Textbox(
                            label="Merged Stocks JSON (Step 2)",
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )
                    
                    with gr.Tab("Transactions JSON"):
                        transactions_output = gr.Textbox(
                            label="Transaction JSON (Step 3)",
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )
                    
                    with gr.Tab("Verification JSON"):
                        verification_output = gr.Textbox(
                            label="Verification JSON (Step 4)",
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )
                    
                    with gr.Tab("Analysis JSON"):
                        analysis_output = gr.Textbox(
                            label="Analysis JSON (Step 5)",
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )
        
        # Wire up the correction JSON visibility
        apply_corrections.change(
            fn=lambda checked: gr.update(visible=checked),
            inputs=[apply_corrections],
            outputs=[correction_json]
        )
        
        # Wire up the run button
        run_button.click(
            fn=run_full_pipeline,
            inputs=[
                # STT
                audio_input,
                language,
                use_sensevoice,
                use_whisperv3_cantonese,
                overwrite_diarization,
                padding_ms,
                use_enhanced_format,
                apply_corrections,
                correction_json,
                vad_model,
                max_single_segment_time,
                # JSON Batch Analysis
                selected_llms,
                system_message,
                ollama_url,
                temperature,
                use_vector_correction,
                use_contextual_analysis,
                enable_stock_verification,
                verification_llm,
                # Transaction Analysis
                transaction_model,
                transaction_system_message,
                transaction_temperature,
                # Trade Verification
                trades_file_path,
                time_window,
                # Conversation Record Analysis
                client_id_filter,
                record_analysis_model,
                record_analysis_temperature,
                use_combined_analysis,
                # Compliance Analysis
                use_llm_compliance,
                compliance_model,
            ],
            outputs=[
                pipeline_log,
                conversation_output,
                stocks_output,
                transactions_output,
                verification_output,
                analysis_output,
                compliance_output,
            ],
        )

