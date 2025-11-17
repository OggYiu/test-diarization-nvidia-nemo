"""
Cantonese Corrector Tool for Agent
Applies text corrections to transcriptions using the CantoneseCorrector class
"""

import os
import sys
import json
import time
from typing import Annotated
from langchain.tools import tool
from pathlib import Path

# Add parent directory to path to import CantoneseCorrector
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import path normalization utilities
from .path_utils import normalize_path_for_llm, normalize_path_from_llm

# Global corrector instance
corrector = None

"""
Cantonese STT Correction Module

This module provides functionality to correct common STT (Speech-to-Text) errors
in Cantonese text using PyCantonese and a configurable word/phrase correction list.

Features:
- Basic text correction using dictionary lookup
- Advanced JSON-based correction with context awareness
- Avoids replacing text inside brackets
- Prevents double-corrections on already-corrected text
- Configurable correction lists with file persistence
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CantoneseCorrector:
    """
    A class to correct common STT errors in Cantonese text.
    
    Supports both single word and phrase corrections, with the ability to
    load corrections from a JSON file or add them programmatically.
    """
    
    def __init__(self, corrections_file: str = None):
        """
        Initialize the Cantonese corrector.
        
        Args:
            corrections_file: Path to JSON file containing corrections.
                            If None, uses default corrections.
        """
        self.corrections: Dict[str, str] = {}
        self.corrections_file = corrections_file or "cantonese_corrections.json"
        
        # Load default corrections
        self._load_default_corrections()
        
        # Load from file if exists
        if Path(self.corrections_file).exists():
            self.load_corrections_from_file(self.corrections_file)
    
    def _load_default_corrections(self):
        """Load default common corrections."""
        default_corrections = {
            # From the example provided
            "滾張子": "搵張紙",
            "寫得": "寫低",
            "滾": "搵",
            
            # Common Cantonese STT errors (add more as needed)
            "張子": "張紙",
            "係咪": "係唔係",
            "聽日": "聽朝",
            "依家": "而家",
            "李個": "呢個",
            "果個": "嗰個",
            "乜野": "乜嘢",
            "邊到": "邊度",
            "點解": "點解",
            "幾多": "幾多",
            "冇野": "冇嘢",
            "做野": "做嘢",
            "食野": "食嘢",
            "講野": "講嘢",
        }
        self.corrections.update(default_corrections)
    
    def add_correction(self, incorrect: str, correct: str):
        """
        Add a single correction pair.
        
        Args:
            incorrect: The incorrect word/phrase from STT
            correct: The correct word/phrase
        """
        self.corrections[incorrect] = correct
        logger.info(f"Added correction: {incorrect} → {correct}")
    
    def add_corrections(self, corrections: Dict[str, str]):
        """
        Add multiple correction pairs.
        
        Args:
            corrections: Dictionary of incorrect -> correct mappings
        """
        self.corrections.update(corrections)
        logger.info(f"Added {len(corrections)} corrections")
    
    def load_corrections_from_file(self, filepath: str):
        """
        Load corrections from a JSON file.
        
        Args:
            filepath: Path to JSON file with format: {"incorrect": "correct", ...}
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                corrections = json.load(f)
                self.add_corrections(corrections)
                logger.info(f"Loaded corrections from {filepath}")
        except Exception as e:
            logger.error(f"Error loading corrections from {filepath}: {e}")
    
    def save_corrections_to_file(self, filepath: str = None):
        """
        Save current corrections to a JSON file.
        
        Args:
            filepath: Path to save corrections. If None, uses default file.
        """
        filepath = filepath or self.corrections_file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.corrections, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved corrections to {filepath}")
        except Exception as e:
            logger.error(f"Error saving corrections to {filepath}: {e}")
    
    def _sort_corrections_by_length(self) -> List[Tuple[str, str]]:
        """
        Sort corrections by length (longest first) to avoid partial replacements.
        
        Returns:
            List of (incorrect, correct) tuples sorted by length
        """
        return sorted(self.corrections.items(), key=lambda x: len(x[0]), reverse=True)
    
    def correct_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Apply corrections to the input text.
        
        Args:
            text: The text to correct
            preserve_case: Whether to preserve case (not typically needed for Chinese)
        
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        corrected = text
        sorted_corrections = self._sort_corrections_by_length()
        
        # Apply corrections (longest phrases first to avoid partial matches)
        for incorrect, correct in sorted_corrections:
            corrected = corrected.replace(incorrect, correct)
        
        return corrected
    
    def apply_text_corrections(self, text_content: str, correction_json: str) -> tuple:
        """
        Apply text corrections to transcription content using JSON format.
        This is an advanced correction method that:
        - Avoids replacing text inside brackets
        - Avoids replacing words already part of the correct phrase
        - Adds corrections in format: wrong_word(correct_word)
        
        Args:
            text_content: The transcription text to correct
            correction_json: JSON string with correction rules
                Format: [{"wrong_words": ["誤", "錯"], "correct_word": "正確"}, ...]
            
        Returns:
            tuple: (corrected_text, error_message)
                   - corrected_text: The corrected text (or original if corrections failed)
                   - error_message: Empty string if successful, error message otherwise
        """
        if not correction_json or not correction_json.strip():
            # No corrections to apply
            return text_content, ""
        
        if not text_content or not text_content.strip():
            # No text to correct
            return text_content, ""
        
        try:
            # Parse JSON
            correction_data = json.loads(correction_json)
            
            corrected_text = text_content
            corrections_applied = []
            
            # Debug: Show original text
            logger.info("\n" + "="*80)
            logger.info("🔍 TEXT CORRECTION DEBUG")
            logger.info("="*80)
            logger.info(f"📝 ORIGINAL TEXT:\n{text_content}")
            logger.info("="*80)
            
            # Handle both single correction object and array of corrections
            if isinstance(correction_data, dict):
                correction_list = [correction_data]
            elif isinstance(correction_data, list):
                correction_list = correction_data
            else:
                return text_content, "❌ Correction JSON must be an object or array"
            
            # Apply each correction
            for idx, correction in enumerate(correction_list):
                if not isinstance(correction, dict):
                    return text_content, f"❌ Correction item {idx} is not an object"
                
                if "wrong_words" not in correction or "correct_word" not in correction:
                    return text_content, f"❌ Correction item {idx} missing required fields"
                
                wrong_words = correction["wrong_words"]
                correct_word = correction["correct_word"]
                
                if not isinstance(wrong_words, list):
                    return text_content, f"❌ 'wrong_words' must be an array"
                
                if not isinstance(correct_word, str):
                    return text_content, f"❌ 'correct_word' must be a string"
                
                # Apply replacements
                for wrong_word in wrong_words:
                    if not isinstance(wrong_word, str):
                        return text_content, f"❌ All items in 'wrong_words' must be strings"
                    
                    if wrong_word in corrected_text:
                        # Use a custom replacement function to avoid replacing text inside brackets
                        # and to avoid replacing wrong words that are already part of the correct phrase
                        pattern = re.escape(wrong_word)
                        
                        # Create a closure with the current text to avoid stale references
                        current_text = corrected_text
                        
                        # Debug: Track replacement decisions
                        debug_skipped = []
                        debug_replaced = []
                        replacement_count = [0]  # Use list to allow modification in nested function
                        
                        def replace_func(match):
                            # Check if this match is inside brackets by looking at the context
                            start_pos = match.start()
                            end_pos = match.end()
                            
                            # Count opening and closing brackets before this position
                            text_before = current_text[:start_pos]
                            open_count = text_before.count('(')
                            close_count = text_before.count(')')
                            
                            # If we're inside brackets, don't replace
                            if open_count > close_count:
                                debug_skipped.append(f"  ⏭️  Skipped '{match.group(0)}' at pos {start_pos} (inside brackets)")
                                return match.group(0)  # Return original text
                            
                            # Check if this match is already part of the correct word/phrase
                            # Extract surrounding context to see if it forms the correct word
                            correct_word_len = len(correct_word)
                            wrong_word_len = len(wrong_word)
                            
                            # Check all possible positions where correct_word could contain this wrong_word
                            for offset in range(correct_word_len - wrong_word_len + 1):
                                # Check if correct_word contains wrong_word at this offset
                                if correct_word[offset:offset + wrong_word_len] == wrong_word:
                                    # Extract the corresponding context from current_text
                                    context_start = start_pos - offset
                                    context_end = context_start + correct_word_len
                                    
                                    # Make sure we're within bounds
                                    if context_start >= 0 and context_end <= len(current_text):
                                        context = current_text[context_start:context_end]
                                        # If the context matches the correct word, skip replacement
                                        if context == correct_word:
                                            debug_skipped.append(f"  ⏭️  Skipped '{match.group(0)}' at pos {start_pos} (already part of '{correct_word}')")
                                            return match.group(0)  # Return original text
                            
                            # Otherwise, apply the correction
                            replacement = f"{match.group(0)}({correct_word})"
                            debug_replaced.append(f"  ✅ Replaced '{match.group(0)}' → '{replacement}' at pos {start_pos}")
                            replacement_count[0] += 1  # Track actual replacements
                            return replacement
                        
                        # Debug: Show text before this replacement
                        logger.info(f"\n🔄 Processing: '{wrong_word}' → '{correct_word}'")
                        logger.info(f"   Text before: {current_text[:100]}..." if len(current_text) > 100 else f"   Text before: {current_text}")
                        
                        # Apply replacement
                        new_text = re.sub(pattern, replace_func, current_text)
                        count = replacement_count[0]  # Use actual replacement count
                        
                        # Debug: Show what happened
                        if debug_replaced:
                            for msg in debug_replaced:
                                logger.info(msg)
                        if debug_skipped:
                            for msg in debug_skipped:
                                logger.info(msg)
                        
                        if count > 0:
                            corrected_text = new_text
                            corrections_applied.append(f"'{wrong_word}' → '{wrong_word}({correct_word})' ({count}x)")
                            logger.info(f"   Text after:  {corrected_text[:100]}..." if len(corrected_text) > 100 else f"   Text after:  {corrected_text}")
                        else:
                            logger.info(f"   ⚠️  No changes made (all occurrences were skipped)")
            
            # Return corrected text with success
            logger.info("\n" + "="*80)
            if corrections_applied:
                logger.info(f"✅ Text corrections applied: {', '.join(corrections_applied)}")
            else:
                logger.info("ℹ️  No corrections were applied")
            logger.info("="*80)
            logger.info(f"📝 FINAL CORRECTED TEXT:\n{corrected_text}")
            logger.info("="*80 + "\n")
            
            return corrected_text, ""
            
        except json.JSONDecodeError as e:
            error_msg = f"❌ Invalid correction JSON: {str(e)}"
            return text_content, error_msg
        except Exception as e:
            error_msg = f"❌ Error applying corrections: {str(e)}"
            return text_content, error_msg
    
    def correct_text_with_context(self, text: str, context_window: int = 3) -> str:
        """
        Apply corrections with context awareness using PyCantonese if available.
        
        Args:
            text: The text to correct
            context_window: Number of characters to consider for context
        
        Returns:
            Corrected text
        """
        return self.correct_text(text)
    
    def list_corrections(self) -> Dict[str, str]:
        """
        Get all current corrections.
        
        Returns:
            Dictionary of all corrections
        """
        return self.corrections.copy()
    
    def remove_correction(self, incorrect: str):
        """
        Remove a correction.
        
        Args:
            incorrect: The incorrect word/phrase to remove from corrections
        """
        if incorrect in self.corrections:
            del self.corrections[incorrect]
            logger.info(f"Removed correction for: {incorrect}")
        else:
            logger.warning(f"Correction not found: {incorrect}")

def initialize_corrector():
    """Initialize the Cantonese corrector with corrections file."""
    global corrector
    
    if corrector is not None:
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Cantonese corrector already initialized, skipping")
        return corrector
    
    # Find the corrections file
    agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corrections_file = os.path.join(agent_dir, "cantonese_corrections.json")
    
    trace_start = time.time()
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Initializing Cantonese corrector...")
    corrector = CantoneseCorrector(corrections_file=corrections_file)
    trace_elapsed = time.time() - trace_start
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Cantonese corrector initialization completed in {trace_elapsed:.4f}s")
    print(f"✅ Cantonese corrector initialized with {len(corrector.corrections)} correction rules")
    
    return corrector


@tool
def correct_transcriptions(
    transcriptions_text_path: Annotated[str, "Path to the transcriptions_text.txt file to correct"]
) -> str:
    """
    Apply Cantonese text corrections to transcriptions from a simple text file.
    
    This tool reads a transcriptions_text.txt file (output from transcribe_audio_segments),
    applies Cantonese corrections to each line, and saves the corrected transcriptions
    to a new file (transcriptions_text_corrected.txt).
    
    Expected input format:
    speaker_0:transcription text here
    speaker_1:another transcription text
    
    The corrections include:
    - Common Cantonese STT errors
    - Custom corrections from cantonese_corrections.json
    - Context-aware replacements that avoid double-corrections
    
    Args:
        transcriptions_text_path: Path to transcriptions_text.txt file
        
    Returns:
        str: Summary of corrections applied with before/after examples
    """
    try:
        # Normalize the input path to handle any LLM path manipulation issues
        transcriptions_text_path = normalize_path_from_llm(transcriptions_text_path)
        
        # Initialize corrector
        corrector_instance = initialize_corrector()
        
        # Check if file exists
        if not os.path.exists(transcriptions_text_path):
            return f"❌ Error: Transcriptions file not found: {transcriptions_text_path}"
        
        # Read transcriptions text file
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Reading transcriptions file...")
        with open(transcriptions_text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Transcriptions file read completed in {trace_elapsed:.4f}s - {len(lines)} lines")
        
        print(f"\n{'='*80}")
        print(f"🔧 Correcting {len(lines)} transcription lines...")
        print(f"{'='*80}\n")
        
        # Load correction rules from JSON file
        agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corrections_file = os.path.join(agent_dir, "cantonese_corrections.json")
        
        if not os.path.exists(corrections_file):
            return f"❌ Error: Corrections file not found: {corrections_file}"
        
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Loading correction rules from JSON...")
        with open(corrections_file, 'r', encoding='utf-8') as f:
            corrections_data = json.load(f)
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Correction rules loaded in {trace_elapsed:.4f}s")
        
        # Convert format from {"correct": ["wrong1", "wrong2"]} to 
        # [{"wrong_words": ["wrong1", "wrong2"], "correct_word": "correct"}]
        correction_list = []
        for correct_word, wrong_words in corrections_data.items():
            correction_list.append({
                "wrong_words": wrong_words,
                "correct_word": correct_word
            })
        
        # Convert to JSON string for apply_text_corrections
        correction_json = json.dumps(correction_list, ensure_ascii=False)
        
        # Apply corrections to each line
        corrected_lines = []
        corrections_summary = []
        
        correction_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Starting text correction for {len(lines)} lines...")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or ':' not in line:
                corrected_lines.append(line)
                continue
            
            # Split speaker and transcription
            parts = line.split(':', 1)
            if len(parts) != 2:
                corrected_lines.append(line)
                continue
            
            speaker, original_text = parts
            
            print(f"📝 Correcting line {i+1}: {speaker}: {original_text}")
            
            # Apply corrections using the advanced method
            corrected_text, error = corrector_instance.apply_text_corrections(
                original_text, 
                correction_json
            )
            
            if error:
                print(f"   ⚠️ Error: {error}")
                corrected_lines.append(line)
                continue
            
            # Create corrected line
            corrected_line = f"{speaker}:{corrected_text}"
            corrected_lines.append(corrected_line)
            
            # Track if changes were made
            if corrected_text != original_text:
                corrections_summary.append({
                    'line_num': i+1,
                    'speaker': speaker,
                    'original': original_text[:100] + '...' if len(original_text) > 100 else original_text,
                    'corrected': corrected_text[:100] + '...' if len(corrected_text) > 100 else corrected_text
                })
                print(f"   ✅ Corrections applied: {corrected_text}")
            else:
                print(f"   ℹ️  No changes needed")
        
        correction_elapsed = time.time() - correction_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Text correction completed in {correction_elapsed:.2f}s")
        
        # Save corrected transcriptions to new file
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Saving corrected transcriptions to file...")
        output_path = transcriptions_text_path.replace('.txt', '_corrected.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in corrected_lines:
                f.write(line + '\n')
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] File save completed in {trace_elapsed:.4f}s")
        
        # Normalize output path for LLM consumption (use forward slashes)
        output_path_for_llm = normalize_path_for_llm(output_path)
        
        # Format summary
        summary = f"\n{'='*80}\n"
        summary += f"✅ Corrected {len(lines)} transcription lines\n"
        summary += f"📝 Applied corrections to {len(corrections_summary)} lines\n"
        summary += f"💾 Saved corrected transcriptions to: {output_path_for_llm}\n"
        summary += f"{'='*80}\n\n"
        
        if corrections_summary:
            summary += "📋 Correction Examples:\n"
            summary += "=" * 80 + "\n\n"
            
            # Show first 3 examples
            for example in corrections_summary[:3]:
                summary += f"📄 Line {example['line_num']} ({example['speaker']})\n"
                summary += f"   Before: {example['original']}\n"
                summary += f"   After:  {example['corrected']}\n"
                summary += "-" * 80 + "\n\n"
        
        # Add explicit instruction to continue to next step
        summary += f"\n{'='*80}\n"
        summary += "✅ Text correction complete. Continue with the next step in the pipeline.\n"
        summary += f"   Use corrected file: {output_path_for_llm}\n"
        summary += f"{'='*80}\n"
        
        return summary
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error during correction: {str(e)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        return error_msg


@tool
def correct_transcriptions_text(
    transcriptions_text_path: Annotated[str, "Path to the transcriptions_text.txt file to correct"]
) -> str:
    """
    Apply Cantonese text corrections to transcriptions from a simple text file.
    
    This tool reads a transcriptions_text.txt file (output from transcribe_audio_segments),
    applies Cantonese corrections to each line, and saves the corrected transcriptions
    to a new file (transcriptions_text_corrected.txt).
    
    Expected input format:
    speaker_0:transcription text here
    speaker_1:another transcription text
    
    The corrections include:
    - Common Cantonese STT errors
    - Custom corrections from cantonese_corrections.json
    - Context-aware replacements that avoid double-corrections
    
    Args:
        transcriptions_text_path: Path to transcriptions_text.txt file
        
    Returns:
        str: Summary of corrections applied with before/after examples
    """
    try:
        # Initialize corrector
        corrector_instance = initialize_corrector()
        
        # Check if file exists
        if not os.path.exists(transcriptions_text_path):
            return f"❌ Error: Transcriptions file not found: {transcriptions_text_path}"
        
        # Read transcriptions text file
        with open(transcriptions_text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"\n{'='*80}")
        print(f"🔧 Correcting {len(lines)} transcription lines...")
        print(f"{'='*80}\n")
        
        # Load correction rules from JSON file
        agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corrections_file = os.path.join(agent_dir, "cantonese_corrections.json")
        
        if not os.path.exists(corrections_file):
            return f"❌ Error: Corrections file not found: {corrections_file}"
        
        with open(corrections_file, 'r', encoding='utf-8') as f:
            corrections_data = json.load(f)
        
        # Convert format from {"correct": ["wrong1", "wrong2"]} to 
        # [{"wrong_words": ["wrong1", "wrong2"], "correct_word": "correct"}]
        correction_list = []
        for correct_word, wrong_words in corrections_data.items():
            correction_list.append({
                "wrong_words": wrong_words,
                "correct_word": correct_word
            })
        
        # Convert to JSON string for apply_text_corrections
        correction_json = json.dumps(correction_list, ensure_ascii=False)
        
        # Apply corrections to each line
        corrected_lines = []
        corrections_summary = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or ':' not in line:
                corrected_lines.append(line)
                continue
            
            # Split speaker and transcription
            parts = line.split(':', 1)
            if len(parts) != 2:
                corrected_lines.append(line)
                continue
            
            speaker, original_text = parts
            
            print(f"📝 Correcting line {i+1}: {speaker}")
            
            # Apply corrections using the advanced method
            corrected_text, error = corrector_instance.apply_text_corrections(
                original_text, 
                correction_json
            )
            
            if error:
                print(f"   ⚠️ Error: {error}")
                corrected_lines.append(line)
                continue
            
            # Create corrected line
            corrected_line = f"{speaker}:{corrected_text}"
            corrected_lines.append(corrected_line)
            
            # Track if changes were made
            if corrected_text != original_text:
                corrections_summary.append({
                    'line_num': i+1,
                    'speaker': speaker,
                    'original': original_text[:100] + '...' if len(original_text) > 100 else original_text,
                    'corrected': corrected_text[:100] + '...' if len(corrected_text) > 100 else corrected_text
                })
                print(f"   ✅ Corrections applied")
            else:
                print(f"   ℹ️  No changes needed")
        
        # Save corrected transcriptions to new file
        output_path = transcriptions_text_path.replace('.txt', '_corrected.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in corrected_lines:
                f.write(line + '\n')
        
        # Format summary
        summary = f"\n{'='*80}\n"
        summary += f"✅ Corrected {len(lines)} transcription lines\n"
        summary += f"📝 Applied corrections to {len(corrections_summary)} lines\n"
        summary += f"💾 Saved corrected transcriptions to: {output_path}\n"
        summary += f"{'='*80}\n\n"
        
        if corrections_summary:
            summary += "📋 Correction Examples:\n"
            summary += "=" * 80 + "\n\n"
            
            # Show first 3 examples
            for example in corrections_summary[:3]:
                summary += f"📄 Line {example['line_num']} ({example['speaker']})\n"
                summary += f"   Before: {example['original']}\n"
                summary += f"   After:  {example['corrected']}\n"
                summary += "-" * 80 + "\n\n"
        
        return summary
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error during correction: {str(e)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        return error_msg

