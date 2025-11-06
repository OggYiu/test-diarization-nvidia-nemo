"""
OpenCC Utilities

Shared utilities for Chinese text conversion using OpenCC (Open Chinese Convert).
This module provides a centralized translation function used across the application.
"""

import logging
from opencc import OpenCC


# Initialize OpenCC converter (Simplified to Traditional Chinese)
opencc_converter = OpenCC('s2t')  # s2t = Simplified to Traditional


def translate_to_traditional_chinese(text: str) -> str:
    """
    Convert Simplified Chinese text to Traditional Chinese using OpenCC.
    
    Args:
        text: Input text (may contain Simplified Chinese)
        
    Returns:
        str: Text with Simplified Chinese converted to Traditional Chinese
    """
    if not text or not text.strip():
        return text
    
    try:
        return opencc_converter.convert(text)
    except Exception as e:
        logging.warning(f"OpenCC translation failed: {e}")
        return text  # Return original text if translation fails

