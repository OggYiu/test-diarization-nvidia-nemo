"""
Unified Gradio GUI combining all phone call analysis tools:
1. Speaker Diarization
2. Batch Speech-to-Text
3. LLM Analysis
4. Speaker Separation
5. Audio Enhancement
6. LLM Comparison
7. File Metadata
8. Multi-LLM Query
9. STT Stock Comparison
10. Transcription Merger
11. Transaction Analysis
12. Milvus Vector Search
13. Transaction Stock Search
14. Trade Verification
15. Text Correction

Modular version - each tab is in a separate file in the tabs/ directory
"""

import gradio as gr

# Import tab creation functions from the tabs module
from tabs import (
    # create_file_metadata_tab,
    # create_diarization_tab,
    create_stt_tab,
    # create_llm_analysis_tab,
    # create_speaker_separation_tab,
    # create_audio_enhancement_tab,
    # create_llm_comparison_tab,
    # create_multi_llm_tab,
    create_stt_stock_comparison_tab,
    # create_transcription_merger_tab,
    create_transaction_analysis_tab,
    # create_milvus_search_tab,
    # create_transaction_stock_search_tab,
    create_trade_verification_tab,
    create_text_correction_tab,
    create_json_batch_analysis_tab,
    create_llm_chat_tab,
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
        
        with gr.Tabs():
            # Create all tabs by calling their respective functions
            # create_file_metadata_tab()
            # create_diarization_tab()
            create_stt_tab()
            # create_llm_analysis_tab()
            # create_speaker_separation_tab()
            # create_audio_enhancement_tab()
            # create_llm_comparison_tab()
            create_json_batch_analysis_tab()
            # create_multi_llm_tab()
            create_stt_stock_comparison_tab()
            # create_transcription_merger_tab()
            create_transaction_analysis_tab()
            # create_milvus_search_tab()
            # create_transaction_stock_search_tab()
            create_trade_verification_tab()
            create_text_correction_tab()
            create_llm_chat_tab()
    
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
