"""
Path normalization utilities for agent tools.
Handles path conversion between OS paths and LLM-friendly paths.
"""

import os
import re
import csv
import json
import time
from datetime import datetime, timedelta


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


def parse_filename_metadata(filename: str, csv_path: str = "client.csv") -> dict:
    """
    Parse audio filename to extract metadata and format output.
    
    Expected filename format:
    [Broker Name Broker_ID]_Unknown1-ClientPhone_YYYYMMDDHHMMSS(Unknown2).wav
    or
    [Broker Name]_Unknown1-ClientPhone_YYYYMMDDHHMMSS(Unknown2).wav
    
    Args:
        filename: The audio filename to parse
        csv_path: Path to client.csv file for name lookup
        
    Returns:
        dict: Dictionary with 'status' (success/error), 'formatted_output' (display string), 
              and 'data' (structured metadata dict)
    """
    try:
        # Remove extension
        base_name = os.path.splitext(filename)[0]
        
        # Parse filename using regex
        # Pattern 1 (with brackets and parentheses): [Broker Name Optional_ID]_Unknown1-ClientPhone_DateTime(Unknown2)
        pattern1 = r'\[(.*?)\]_(\d+)-(\d+)_(\d{14})\((\d+)\)'
        match = re.match(pattern1, base_name)
        
        # Pattern 2 (sanitized by Gradio - no brackets or parentheses): Broker Name Optional_ID_Unknown1-ClientPhone_DateTime Unknown2
        # Example: "Dickson Lau 0489_8330-97501167_2025101001451020981"
        if not match:
            pattern2 = r'(.*?)_(\d+)-(\d+)_(\d{14})(\d+)'
            match = re.match(pattern2, base_name)
        
        if not match:
            error_msg = f"""Error: Filename does not match expected pattern.

Expected format:
[Broker Name ID]_8330-97501167_20251010014510(20981).wav

Received:
{filename}

Note: The filename may have been sanitized by the system. 
Please ensure special characters are preserved or manually enter the correct format.
"""
            return {
                'status': 'error',
                'formatted_output': error_msg,
                'data': None
            }
        
        broker_info = match.group(1)  # e.g., "Dickson Lau 0489" or "Dickson Lau"
        # unknown_1 = match.group(2)     # e.g., "8330"
        client_number = match.group(3) # e.g., "97501167"
        datetime_str = match.group(4)  # e.g., "20251010014510"
        # unknown_2 = match.group(5)     # e.g., "20981"
        
        # Parse broker name and ID
        broker_parts = broker_info.rsplit(' ', 1)  # Split from right to handle multi-word names
        if len(broker_parts) == 2 and broker_parts[1].isdigit():
            broker_name = broker_parts[0]
            broker_id = broker_parts[1]
        else:
            broker_name = broker_info
            broker_id = "N/A"
        
        # Parse datetime (UTC)
        try:
            utc_dt = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
            # Convert to HKT (UTC+8)
            hkt_dt = utc_dt + timedelta(hours=8)
            
            utc_formatted = utc_dt.strftime("%Y-%m-%dT%H:%M:%S")
            hkt_formatted = hkt_dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError as e:
            error_msg = f"Error parsing datetime: {str(e)}"
            return {
                'status': 'error',
                'formatted_output': error_msg,
                'data': None
            }
        
        # Look up client name and ID in CSV
        client_name = "Not found"
        client_id = "Not found"
        if os.path.exists(csv_path):
            trace_start = time.time()
            print(f"[TRACE {time.strftime('%H:%M:%S')}] Reading CSV file for client lookup: {os.path.basename(csv_path)}")
            try:
                with open(csv_path, 'r', encoding='utf-8') as csvfile:
                    csv_reader = csv.DictReader(csvfile)
                    for row in csv_reader:
                        # Find name column
                        name_col = None
                        for col in row.keys():
                            if 'name' in col.lower() or 'client' in col.lower():
                                name_col = col
                                break
                        
                        # Check if phone number matches any of the three phone columns
                        # CSV format: AE,acctno,name,mobile,home,office
                        phone_columns = ['mobile', 'home', 'office']
                        for phone_col in phone_columns:
                            if phone_col in row and row[phone_col].strip() == client_number:
                                if name_col:
                                    client_name = row[name_col].strip()
                                # Get client ID (acctno column)
                                if 'acctno' in row:
                                    client_id = row['acctno'].strip()
                                break
                        
                        # If we found the client, exit the loop
                        if client_name != "Not found":
                            break
                trace_elapsed = time.time() - trace_start
                print(f"[TRACE {time.strftime('%H:%M:%S')}] CSV lookup completed in {trace_elapsed:.4f}s")
            except Exception as e:
                trace_elapsed = time.time() - trace_start
                print(f"[TRACE {time.strftime('%H:%M:%S')}] CSV lookup failed after {trace_elapsed:.4f}s: {e}")
                client_name = f"Error reading CSV: {str(e)}"
                client_id = "Error reading CSV"
        else:
            client_name = "client.csv not found"
            client_id = "client.csv not found"
        
        # Format output for display
        formatted_output = f"""Metadata extracted successfully

Broker Name: {broker_name}
Broker Id: {broker_id}
Client Number: {client_number}
Client Name: {client_name}
Client Id: {client_id}
UTC: {utc_formatted}
HKT: {hkt_formatted}
"""
        
        # Prepare structured data
        metadata_dict = {
            'filename': filename,
            'broker_name': broker_name,
            'broker_id': broker_id,
            'client_number': client_number,
            'client_name': client_name,
            'client_id': client_id,
            'utc_datetime': utc_formatted,
            'hkt_datetime': hkt_formatted,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return {
            'status': 'success',
            'formatted_output': formatted_output,
            'data': metadata_dict
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error parsing filename: {str(e)}\n\n{traceback.format_exc()}"
        return {
            'status': 'error',
            'formatted_output': error_msg,
            'data': None
        }


def identify_speakers_from_filename(audio_file_path: str) -> str:
    """
    Identify speakers (broker and client) and extract metadata from audio filename.
    
    CRITICAL: Use this tool FIRST before diarizing audio files. This tool identifies who the speakers are:
    - The broker (one speaker in the conversation)
    - The client (the other speaker in the conversation)
    
    When diarization identifies "speaker 0" and "speaker 1", this tool's output tells you which one is the broker and which is the client.
    
    This tool parses audio filenames in the format:
    [Broker Name ID]_8330-97501167_20251010014510(20981).wav
    
    It extracts:
    - Broker name and ID (one of the speakers)
    - Client phone number, name, and ID (the other speaker)
    - UTC and HKT timestamps
    
    The extracted metadata is saved to agent/output/metadata/ as a JSON file for later reference.
    
    Args:
        audio_file_path: Path to the audio file (filename will be extracted from path)
        
    Returns:
        A formatted string with the extracted metadata including broker and client information, or an error message if parsing fails
    """
    # Extract just the filename from the full path
    filename = os.path.basename(audio_file_path)
    
    # Determine the CSV path - look in agent/assets/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    agent_dir = os.path.dirname(current_dir)  # agent/
    csv_path = os.path.join(agent_dir, "assets", "client.csv")
    
    # Fallback to current directory if not found in parent
    if not os.path.exists(csv_path):
        csv_path = "client.csv"
    
    # Parse the metadata
    result = parse_filename_metadata(filename, csv_path)
    
    # Save metadata to output/metadata/ if parsing was successful
    if result['status'] == 'success' and result['data']:
        try:
            # Determine the output directory - use agent/output/metadata/
            # current_dir is agent/tools/, so agent_dir is agent/
            agent_dir = os.path.dirname(current_dir)
            metadata_output_dir = os.path.join(agent_dir, "output", "metadata")
            
            # Create directory if it doesn't exist
            os.makedirs(metadata_output_dir, exist_ok=True)
            
            # Create JSON filename based on audio filename (replace extension with .json)
            audio_basename = os.path.splitext(filename)[0]
            json_filename = f"{audio_basename}.json"
            json_filepath = os.path.join(metadata_output_dir, json_filename)
            
            # Save metadata as JSON
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(result['data'], f, indent=2, ensure_ascii=False)
            
            # Update formatted output to include save confirmation
            result['formatted_output'] += f"\n💾 Metadata saved to: {json_filepath}"
        except Exception as e:
            # If saving fails, log but don't fail the tool
            error_msg = f"\n⚠️  Warning: Could not save metadata to file: {str(e)}"
            result['formatted_output'] += error_msg
    
    # Add explicit instruction to continue to next step in pipeline
    result['formatted_output'] += f"\n\n{'='*80}\n"
    result['formatted_output'] += "✅ Metadata extraction complete. Continue with the next step in the pipeline.\n"
    result['formatted_output'] += f"{'='*80}\n"
    
    # Return the formatted output
    return result['formatted_output']
