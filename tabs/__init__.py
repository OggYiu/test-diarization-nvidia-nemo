"""
Tabs module for the Unified Phone Call Analysis Suite
Each tab is in a separate file for better organization
"""

from .tab_file_metadata import create_file_metadata_tab
from .tab_diarization import create_diarization_tab
from .tab_stt import create_stt_tab
from .tab_llm_analysis import create_llm_analysis_tab
from .tab_speaker_separation import create_speaker_separation_tab
from .tab_audio_enhancement import create_audio_enhancement_tab
from .tab_llm_comparison import create_llm_comparison_tab
from .tab_multi_llm import create_multi_llm_tab
from .tab_stt_stock_comparison import create_stt_stock_comparison_tab
from .tab_transcription_merger import create_transcription_merger_tab
from .tab_transaction_analysis import create_transaction_analysis_tab
from .tab_milvus_search import create_milvus_search_tab
from .tab_transaction_stock_search import create_transaction_stock_search_tab

__all__ = [
    'create_file_metadata_tab',
    'create_diarization_tab',
    'create_stt_tab',
    'create_llm_analysis_tab',
    'create_speaker_separation_tab',
    'create_audio_enhancement_tab',
    'create_llm_comparison_tab',
    'create_multi_llm_tab',
    'create_stt_stock_comparison_tab',
    'create_transcription_merger_tab',
    'create_transaction_analysis_tab',
    'create_milvus_search_tab',
    'create_transaction_stock_search_tab',
]

