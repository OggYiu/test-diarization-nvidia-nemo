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
from agent.cantonese_corrector import CantoneseCorrector

# Import path normalization utilities
from .path_utils import normalize_path_for_llm, normalize_path_from_llm

# Global corrector instance
corrector = None


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

