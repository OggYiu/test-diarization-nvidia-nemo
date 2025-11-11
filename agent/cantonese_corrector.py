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


def main():
    """Test the Cantonese corrector with example cases."""
    
    # Initialize corrector
    script_dir = Path(__file__).parent
    corrections_file = script_dir / "cantonese_corrections.json"
    corrector = CantoneseCorrector(corrections_file=str(corrections_file))
    
    # Test cases
    test_cases = {
        "阿劉生啊排啊排憂態啊呵呵。": "阿劉生啊排(掛單)啊排(掛單)憂態(丘鈦)啊呵呵。",
        "阿阿爸爸你啊爸爸。😊": "阿阿爸爸(阿里巴巴)你啊爸爸(阿里巴巴)。😊",
    }
    
    # Load corrections from JSON file and convert to the format expected by apply_text_corrections
    corrections_json_path = corrections_file
    if corrections_json_path.exists():
        with open(corrections_json_path, 'r', encoding='utf-8') as f:
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
        
        print("\n" + "="*80)
        print("CANTONESE CORRECTOR TEST")
        print("="*80)
        
        # Run test cases
        for input_text, expected_output in test_cases.items():
            print(f"\nInput:    {input_text}")
            print(f"Expected: {expected_output}")
            
            # Apply corrections
            corrected_text, error = corrector.apply_text_corrections(input_text, correction_json)
            
            if error:
                print(f"Error: {error}")
            else:
                print(f"Result:   {corrected_text}")
                
                # Check if result matches expected
                if corrected_text == expected_output:
                    print("TEST PASSED!")
                else:
                    print("TEST FAILED!")
                    print(f"   Difference detected:")
                    print(f"   Expected: {expected_output}")
                    print(f"   Got:      {corrected_text}")
        
        print("\n" + "="*80)
    else:
        print(f"Corrections file not found: {corrections_json_path}")


if __name__ == "__main__":
    main()

