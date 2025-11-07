"""
Tabs module for the Unified Phone Call Analysis Suite
Each tab is in a separate file for better organization
"""

from .tab_stt import create_stt_tab
from .tab_speaker_separation import create_speaker_separation_tab
from .tab_llm_comparison import create_llm_comparison_tab
from .tab_llm_chat import create_llm_chat_tab
from .tab_multi_llm import create_multi_llm_tab
from .tab_stt_stock_comparison import create_stt_stock_comparison_tab
from .tab_transcription_merger import create_transcription_merger_tab
from .tab_transaction_analysis import create_transaction_analysis_tab
from .tab_transaction_analysis_json import create_transaction_analysis_json_tab
from .tab_milvus_search import create_milvus_search_tab
from .tab_transaction_stock_search import create_transaction_stock_search_tab
from .tab_trade_verification import create_trade_verification_tab
from .tab_text_correction import create_text_correction_tab
from .tab_json_batch_analysis import create_json_batch_analysis_tab
from .tab_csv_stock_enrichment import create_csv_stock_enrichment_tab
from .tab_conversation_record_analysis import create_conversation_record_analysis_tab
from .tab_compliance_analysis import create_compliance_analysis_tab

__all__ = [
    'create_file_metadata_tab',
    'create_diarization_tab',
    'create_stt_tab',
    'create_llm_analysis_tab',
    'create_speaker_separation_tab',
    'create_audio_enhancement_tab',
    'create_llm_comparison_tab',
    'create_llm_chat_tab',
    'create_multi_llm_tab',
    'create_stt_stock_comparison_tab',
    'create_transcription_merger_tab',
    'create_transaction_analysis_tab',
    'create_transaction_analysis_json_tab',
    'create_milvus_search_tab',
    'create_transaction_stock_search_tab',
    'create_trade_verification_tab',
    'create_text_correction_tab',
    'create_json_batch_analysis_tab',
    # 'create_csv_stock_enrichment_tab',
    'create_conversation_record_analysis_tab',
    'create_compliance_analysis_tab',
]

