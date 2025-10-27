import gradio as gr
import os
import shutil
import re
import csv
from pathlib import Path
from datetime import datetime

def load_client_data(csv_path):
    """
    Load client data from CSV file.
    Returns: dict mapping phone_number -> {'client_id': ..., 'name': ...}
    """
    client_data = {}
    
    if not os.path.exists(csv_path):
        return client_data
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            
            for row in reader:
                if len(row) >= 4:
                    ae = row[0]
                    client_id = row[1]
                    name = row[2]
                    mobile = row[3]
                    
                    # Store mobile phone number as key
                    if mobile and mobile != 'NULL' and mobile.strip():
                        # Clean up phone number (remove spaces, special chars if any)
                        mobile_clean = mobile.strip()
                        client_data[mobile_clean] = {
                            'client_id': client_id,
                            'name': name
                        }
    except Exception as e:
        print(f"Error loading client CSV: {e}")
    
    return client_data

def parse_filename(filename):
    """
    Parse filename with format: [Name]_ClientID-PhoneNumber_DateTimeStamp(ID).wav
    Example: [Dickson Lau 0489]_8330-97501167_20251010014510(20981).wav
    
    Returns: dict with name, client_id, phone_number, date, time, extra_id
    """
    # Pattern: [Name]_ClientID-PhoneNumber_DateTimeStamp(ID).extension
    pattern = r'\[([^\]]+)\]_(\d+)-(\d+)_(\d{8})(\d{6})\((\d+)\)\.(.*)'
    
    match = re.match(pattern, filename)
    if match:
        name = match.group(1)
        client_id = match.group(2)
        phone_number = match.group(3)
        date = match.group(4)  # YYYYMMDD
        time = match.group(5)  # HHMMSS
        extra_id = match.group(6)
        extension = match.group(7)
        
        return {
            'name': name,
            'client_id': client_id,
            'phone_number': phone_number,
            'date': date,
            'time': time,
            'extra_id': extra_id,
            'extension': extension
        }
    return None

def organize_files(folder_path, progress=gr.Progress()):
    """
    Organize files from input folder into structured output folder
    """
    try:
        if not folder_path or not os.path.exists(folder_path):
            return "❌ Error: Invalid folder path", ""
        
        # Get all files in the folder
        input_path = Path(folder_path)
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
        
        if not files:
            return "❌ No files found in the specified folder", ""
        
        # Load client data from CSV
        csv_path = "client.csv"
        client_data = load_client_data(csv_path)
        
        if not client_data:
            return "❌ Error: client.csv not found or empty in the specified folder", ""
        
        # Create output folder
        output_folder = input_path / "output"
        output_folder.mkdir(exist_ok=True)
        
        processed_files = 0
        skipped_files = 0
        error_files = []
        file_structure = {}
        missing_phone_numbers = []
        
        progress(0, desc="Starting file organization...")
        
        for idx, filename in enumerate(files):
            # Skip the CSV file itself
            if filename.lower() == 'client.csv':
                continue
                
            progress((idx + 1) / len(files), desc=f"Processing {filename}...")
            
            # Parse filename
            parsed = parse_filename(filename)
            
            if not parsed:
                skipped_files += 1
                error_files.append(f"Could not parse: {filename}")
                continue
            
            # Look up client info from CSV using phone number
            phone_number = parsed['phone_number']
            
            if phone_number in client_data:
                # Use data from CSV
                client_info = client_data[phone_number]
                client_id = client_info['client_id']
                client_name = client_info['name']
            else:
                # Phone number not found in CSV
                skipped_files += 1
                error_files.append(f"Phone number {phone_number} not found in CSV: {filename}")
                missing_phone_numbers.append(phone_number)
                continue
            
            # Create folder structure: PhoneNumber_ClientID_Name/Date/
            person_folder_name = f"{phone_number}_{client_id}_{client_name}"
            date_folder_name = parsed['date']  # YYYYMMDD format
            
            # Create full path
            person_folder = output_folder / person_folder_name
            date_folder = person_folder / date_folder_name
            
            # Create directories if they don't exist
            date_folder.mkdir(parents=True, exist_ok=True)
            
            # Copy file to destination
            source_file = input_path / filename
            dest_file = date_folder / filename
            
            try:
                shutil.copy2(source_file, dest_file)
                processed_files += 1
                
                # Track file structure for display
                if person_folder_name not in file_structure:
                    file_structure[person_folder_name] = {}
                if date_folder_name not in file_structure[person_folder_name]:
                    file_structure[person_folder_name][date_folder_name] = []
                file_structure[person_folder_name][date_folder_name].append(filename)
                
            except Exception as e:
                error_files.append(f"Error copying {filename}: {str(e)}")
                continue
        
        # Generate summary report
        summary = f"✅ **File Organization Complete!**\n\n"
        summary += f"📁 **Output Location:** `{output_folder}`\n\n"
        summary += f"**Statistics:**\n"
        summary += f"- ✅ Successfully processed: {processed_files} files\n"
        summary += f"- ⚠️ Skipped: {skipped_files} files\n"
        summary += f"- ❌ Errors: {len(error_files)} files\n"
        summary += f"- 📋 Client records loaded: {len(client_data)} phone numbers\n\n"
        
        if missing_phone_numbers:
            summary += f"⚠️ **Missing phone numbers in CSV:** {len(set(missing_phone_numbers))}\n"
            unique_missing = list(set(missing_phone_numbers))[:5]
            summary += f"   (e.g., {', '.join(unique_missing)})\n\n"
        
        # Generate folder structure visualization
        structure_text = "**📂 Folder Structure Created:**\n\n"
        structure_text += f"```\n{output_folder.name}/\n"
        
        for person_folder, dates in sorted(file_structure.items()):
            structure_text += f"├── {person_folder}/\n"
            date_list = sorted(dates.items())
            for idx, (date_folder, files) in enumerate(date_list):
                is_last_date = (idx == len(date_list) - 1)
                date_prefix = "└──" if is_last_date else "├──"
                file_prefix = "    " if is_last_date else "│   "
                
                # Format date for display
                try:
                    formatted_date = datetime.strptime(date_folder, "%Y%m%d").strftime("%Y-%m-%d")
                except:
                    formatted_date = date_folder
                
                structure_text += f"│   {date_prefix} {date_folder}/ ({formatted_date}) - {len(files)} file(s)\n"
        
        structure_text += "```\n\n"
        
        # Add error details if any
        if error_files:
            structure_text += "**⚠️ Issues Encountered:**\n\n"
            for error in error_files[:10]:  # Show first 10 errors
                structure_text += f"- {error}\n"
            if len(error_files) > 10:
                structure_text += f"- ... and {len(error_files) - 10} more\n"
        
        return summary, structure_text
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

def create_gui():
    """
    Create Gradio interface for file organization
    """
    with gr.Blocks(title="Audio File Organizer", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 📁 Audio File Organizer
            
            Organize audio files by person (phone number) and date based on filename pattern.
            
            **Requirements:**  
            - A `client.csv` file must be present in the input folder
            - CSV format: `AE,acctno,name,mobile,home,office`
            - Client ID and Name are looked up from CSV using the phone number
            
            **Expected filename format:**  
            `[Name]_ClientID-PhoneNumber_DateTimeStamp(ID).extension`
            
            **Example:**  
            `[Dickson Lau 0489]_8330-97501167_20251010014510(20981).wav`
            
            **Output structure:**  
            `output/PhoneNumber_ClientID_Name/Date/filename.wav`
            
            *Note: Client ID and Name are retrieved from client.csv based on the phone number in the filename*
            """
        )
        
        with gr.Row():
            with gr.Column():
                folder_input = gr.Textbox(
                    label="Input Folder Path",
                    placeholder="Enter the path to the folder containing audio files...",
                    lines=1
                )
                
                gr.Markdown("💡 *Tip: You can paste the folder path or drag and drop below*")
                
                # Alternative file upload (for drag and drop)
                file_upload = gr.File(
                    label="Or Drag & Drop Files Here",
                    file_count="multiple",
                    type="filepath"
                )
                
                organize_btn = gr.Button("🚀 Organize Files", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                summary_output = gr.Markdown(label="Summary")
            
        with gr.Row():
            with gr.Column():
                structure_output = gr.Markdown(label="Folder Structure")
        
        # Handle folder path input
        def process_folder(folder_path):
            return organize_files(folder_path)
        
        # Handle file upload (extract folder from first file)
        def process_uploaded_files(files):
            if files and len(files) > 0:
                # Get the directory of the first uploaded file
                first_file_path = Path(files[0])
                folder_path = str(first_file_path.parent)
                return organize_files(folder_path)
            return "❌ No files uploaded", ""
        
        # Connect button click
        organize_btn.click(
            fn=process_folder,
            inputs=[folder_input],
            outputs=[summary_output, structure_output]
        )
        
        # Also allow organizing when files are uploaded
        file_upload.change(
            fn=process_uploaded_files,
            inputs=[file_upload],
            outputs=[summary_output, structure_output]
        )
        
        gr.Markdown(
            """
            ---
            ### 📋 Instructions:
            1. Ensure `client.csv` is in the same folder as your audio files
            2. Enter the folder path containing your audio files, or drag and drop files
            3. Click "Organize Files"
            4. Files will be copied to an `output` folder with organized structure
            5. Original files remain unchanged
            
            ### 📁 Output Format:
            - Folder name: `PhoneNumber_ClientID_Name` (from CSV lookup)
            - Subfolders by date: `YYYYMMDD` (from filename)
            - Files are copied (not moved) to preserve originals
            
            ### 📝 CSV File Requirements:
            - File must be named `client.csv` and located in the input folder
            - Must have headers: `AE,acctno,name,mobile,home,office`
            - Phone numbers in the `mobile` column are matched with filenames
            """
        )
    
    return app

if __name__ == "__main__":
    app = create_gui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

