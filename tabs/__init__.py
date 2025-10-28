"""
Tabs module for the Unified Phone Call Analysis Suite
Each tab is in a separate file for better organization
"""

from .tab_file_metadata import create_file_metadata_tab
from .tab_diarization import create_diarization_tab
from .tab_chopper import create_chopper_tab
from .tab_stt import create_stt_tab
from .tab_llm_analysis import create_llm_analysis_tab
from .tab_speaker_separation import create_speaker_separation_tab
from .tab_audio_enhancement import create_audio_enhancement_tab
from .tab_llm_comparison import create_llm_comparison_tab

__all__ = [
    'create_file_metadata_tab',
    'create_diarization_tab',
    'create_chopper_tab',
    'create_stt_tab',
    'create_llm_analysis_tab',
    'create_speaker_separation_tab',
    'create_audio_enhancement_tab',
    'create_llm_comparison_tab',
]

