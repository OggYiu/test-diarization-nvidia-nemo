"""
Path normalization utilities for agent tools.
Handles path conversion between OS paths and LLM-friendly paths.
"""

import os


def normalize_path_for_llm(path: str) -> str:
    """
    Normalize a file path for LLM consumption by converting backslashes to forward slashes.
    This prevents issues where the LLM incorrectly handles Windows paths with backslashes.
    
    Args:
        path: File path (may contain backslashes on Windows)
        
    Returns:
        Path with forward slashes (works on both Windows and Unix)
    """
    if path:
        # Convert backslashes to forward slashes for LLM compatibility
        # Forward slashes work on Windows too, so this is safe
        return path.replace('\\', '/')
    return path


def normalize_path_from_llm(path: str) -> str:
    """
    Normalize a file path received from the LLM by converting it to a proper OS path.
    Handles cases where the LLM might have incorrectly modified paths (e.g., $$ instead of \\)
    
    Args:
        path: File path (may contain forward slashes or incorrectly escaped backslashes)
        
    Returns:
        Properly normalized OS path
    """
    if not path:
        return path
    
    # First, fix any incorrect $$ patterns that might have been introduced
    # This handles the bug where $$ appears instead of \\
    path = path.replace('$$', '\\')
    
    # Normalize the path using os.path
    normalized = os.path.normpath(path)
    
    return normalized

