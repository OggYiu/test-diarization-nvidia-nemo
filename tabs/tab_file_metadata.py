"""
Tab 8: File Metadata Extraction
Parse audio filenames to extract broker info, client info, and timestamps
"""

import os
import csv
import re
import traceback
from datetime import datetime, timedelta
import gradio as gr

from mongodb_utils import save_to_mongodb, find_one_from_mongodb


# MongoDB collection name for file metadata
METADATA_COLLECTION = "file_metadata"


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
            error_msg = f"""❌ Error: Filename does not match expected pattern.

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
            error_msg = f"❌ Error parsing datetime: {str(e)}"
            return {
                'status': 'error',
                'formatted_output': error_msg,
                'data': None
            }
        
        # Look up client name and ID in CSV
        client_name = "Not found"
        client_id = "Not found"
        if os.path.exists(csv_path):
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
            except Exception as e:
                client_name = f"Error reading CSV: {str(e)}"
                client_id = "Error reading CSV"
        else:
            client_name = "client.csv not found"
            client_id = "client.csv not found"
        
        # Format output for display
        formatted_output = f"""✅ Metadata extracted successfully

Broker Name: {broker_name}
Broker Id: {broker_id}
Client Number: {client_number}
Client Name: {client_name}
Client Id: {client_id}
UTC: {utc_formatted}
HKT: {hkt_formatted}

💾 Saved to MongoDB
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
            'utc_datetime_obj': utc_dt,
            'hkt_datetime_obj': hkt_dt,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return {
            'status': 'success',
            'formatted_output': formatted_output,
            'data': metadata_dict
        }
        
    except Exception as e:
        error_msg = f"❌ Error parsing filename: {str(e)}\n\n{traceback.format_exc()}"
        return {
            'status': 'error',
            'formatted_output': error_msg,
            'data': None
        }


def process_file_metadata(audio_file):
    """
    Process uploaded audio file and extract metadata from filename.
    Also saves the metadata to MongoDB.
    
    Args:
        audio_file: Audio file from Gradio interface
        
    Returns:
        str: Formatted metadata string
    """
    if audio_file is None:
        return "❌ No file uploaded. Please drag and drop an audio file."
    
    try:
        # Get filename
        filename = os.path.basename(audio_file)
        
        # Parse metadata
        result = parse_filename_metadata(filename)
        
        # If parsing was successful, save to MongoDB
        if result['status'] == 'success' and result['data']:
            save_to_mongodb(METADATA_COLLECTION, result['data'], unique_key='filename')
        
        # Return the formatted output string for display
        return result['formatted_output']
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg


def create_file_metadata_tab():
    """Create and return the File Metadata tab"""
    with gr.Tab("8️⃣ File Metadata"):
        gr.Markdown("### Extract metadata from audio filename")
        gr.Markdown("*Parse filename to extract broker info, client info, and timestamps*")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Input")
                meta_audio_input = gr.File(
                    label="Upload Audio File",
                    type="filepath",
                    file_types=["audio"]
                )
                meta_process_btn = gr.Button("📋 Extract Metadata", variant="primary", size="lg")
                
                gr.Markdown("""
                #### Expected Filename Format:
                ```
                [Broker Name ID]_8330-97501167_20251010014510(20981).wav
                ```
                
                #### Components:
                - **[Broker Name ID]**: Broker's name and optional ID
                - **8330**: Unknown field 1
                - **97501167**: Client phone number
                - **20251010014510**: UTC timestamp (YYYYMMDDHHMMSS)
                - **(20981)**: Unknown field 2
                
                #### Example:
                `[Dickson Lau 0489]_8330-97501167_20251010014510(20981).wav`
                
                #### Client Lookup:
                Place a `client.csv` file in the same directory with columns for phone number and client name.
                The system will automatically look up the client name based on the phone number.
                """)
                
            with gr.Column(scale=2):
                gr.Markdown("#### Extracted Metadata")
                meta_output = gr.Textbox(
                    label="Results",
                    lines=25,
                    max_lines=35,
                    interactive=False,
                    show_copy_button=True
                )
        
        meta_process_btn.click(
            fn=process_file_metadata,
            inputs=[meta_audio_input],
            outputs=[meta_output]
        )

