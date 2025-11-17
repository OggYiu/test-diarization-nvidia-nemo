"""
Settings configuration for the agent.

This file contains all configurable settings for the agent and its tools.
Modify these values to change the default behavior of the agent.
"""

# Diarization tool settings
DIARIZATION_OVERWRITE = False  # If True, re-run diarization even if RTTM file already exists

# Audio chopper tool settings
AUDIO_CHOPPER_OVERWRITE = False  # If True, overwrite existing chopped segments

# STT (Speech-to-Text) tool settings
STT_OVERWRITE = False  # If True, overwrite existing transcription files

