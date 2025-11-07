import gradio as gr

# Import tab creation functions from the tabs module
from tabs import (
    create_stt_tab,
    create_json_batch_analysis_tab,
    # create_speaker_separation_tab,
    # create_audio_enhancement_tab,
    # create_llm_comparison_tab,
    # create_multi_llm_tab,
    # create_stt_stock_comparison_tab,
    # create_transcription_merger_tab,
    # create_transaction_analysis_tab,
    create_transaction_analysis_json_tab,
    # create_milvus_search_tab,
    # create_transaction_stock_search_tab,
    create_trade_verification_tab,
    # create_text_correction_tab,
    # create_llm_chat_tab,
    create_conversation_record_analysis_tab,
    create_compliance_analysis_tab,
    # create_csv_stock_enrichment_tab,
    create_full_pipeline_tab,
)


def create_unified_interface():
    """Create the unified Gradio interface with all tools in tabs."""
    
    with gr.Blocks(title="Phone Call Analysis Suite", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📞 Phone Call Analysis Suite
            ### All-in-one tool for speaker diarization, audio processing, transcription, and analysis
            """
        )
        
        # ====================================================================
        # SHARED STATE COMPONENTS - Data Pipeline
        # ====================================================================
        # These hidden state components pass data between tabs for chaining
        # Complete chain: STT → JSON Batch Analysis → Transaction Analysis JSON → Trade Verification → Compliance Analysis
        shared_conversation_json = gr.State(None)  # Conversation JSON from STT
        shared_merged_stocks_json = gr.State(None)  # Merged stocks from JSON Batch Analysis
        shared_transaction_json = gr.State(None)  # Transaction analysis results
        shared_trade_verification_json = gr.State(None)  # Trade verification results
        shared_conversation_analysis_json = gr.State(None)  # Conversation record analysis results
        
        with gr.Tabs():
            # Create all tabs by calling their respective functions
            # Pass shared state to tabs that support chaining
            # create_file_metadata_tab()
            # create_diarization_tab()
            
            # Full Pipeline Tab - Chains all steps automatically
            create_full_pipeline_tab()
            
            # Chain 1: STT → JSON Batch Analysis
            create_stt_tab(output_json_state=shared_conversation_json)
            
            # create_llm_analysis_tab()
            # create_speaker_separation_tab()
            # create_audio_enhancement_tab()
            # create_llm_comparison_tab()
            
            # Chain 2: JSON Batch Analysis → Transaction Analysis JSON
            create_json_batch_analysis_tab(
                input_json_state=shared_conversation_json,
                output_stocks_state=shared_merged_stocks_json
            )
            
            # create_multi_llm_tab()
            # create_stt_stock_comparison_tab()
            # create_transcription_merger_tab()
            # create_transaction_analysis_tab()
            
            # Chain 3: Transaction Analysis JSON → Trade Verification
            create_transaction_analysis_json_tab(
                input_conversation_state=shared_conversation_json,
                input_stocks_state=shared_merged_stocks_json,
                output_transaction_state=shared_transaction_json
            )
            
            # create_milvus_search_tab()
            # create_transaction_stock_search_tab()
            
            # Chain 4: Trade Verification (outputs to shared state for compliance analysis)
            create_trade_verification_tab(
                input_transaction_state=shared_transaction_json,
                output_verification_state=shared_trade_verification_json
            )
            
            # Chain 5: Conversation Record Analysis (receives conversation JSON from STT, outputs to shared state)
            create_conversation_record_analysis_tab(
                input_json_state=shared_conversation_json,
                output_analysis_state=shared_conversation_analysis_json
            )
            
            # Chain 6: Compliance Analysis (final step - analyzes both verification and analysis results)
            create_compliance_analysis_tab(
                trade_verification_state=shared_trade_verification_json,
                conversation_analysis_state=shared_conversation_analysis_json
            )
            
            # create_csv_stock_enrichment_tab()
            # create_text_correction_tab()
            # create_llm_chat_tab()
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Unified Phone Call Analysis Suite...", flush=True)
    print("📝 All tools available in one interface!", flush=True)
    print("=" * 60, flush=True)
    
    demo = create_unified_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
