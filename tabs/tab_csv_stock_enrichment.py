"""
Tab: CSV Stock Name Enrichment
Upload a CSV file, find SCTYCode column, enrich with stock names from vector store
"""

import csv
import io
import os
import traceback
from typing import Dict, List, Optional
import gradio as gr

# Import stock verification functionality
from stock_verifier_module.stock_verifier_improved import (
    get_vector_store,
    verify_and_correct_stock,
    normalize_stock_code,
)


def enrich_csv_with_stock_names(csv_file: str, progress=gr.Progress()) -> tuple[str, str, str]:
    """
    Process uploaded CSV file and enrich with stock names from vector store
    
    Args:
        csv_file: Path to uploaded CSV file
        progress: Gradio progress tracker
        
    Returns:
        tuple of (status_message, enriched_csv_path, error_log)
    """
    try:
        if not csv_file:
            return "❌ No file uploaded", None, ""
        
        progress(0, desc="Reading CSV file...")
        
        # Read the CSV file
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.DictReader(f)
            fieldnames = csv_reader.fieldnames
            
            if not fieldnames:
                return "❌ CSV file is empty or invalid", None, ""
            
            # Check if SCTYCode column exists
            if 'SCTYCode' not in fieldnames:
                available_columns = ', '.join(fieldnames)
                return f"❌ SCTYCode column not found. Available columns: {available_columns}", None, ""
            
            rows = list(csv_reader)
        
        if not rows:
            return "❌ CSV file contains no data rows", None, ""
        
        progress(0.1, desc=f"Found {len(rows)} rows. Initializing vector store...")
        
        # Initialize vector store
        vector_store = get_vector_store()
        if not vector_store.is_available:
            if not vector_store.initialize():
                return "❌ Failed to initialize vector store", None, "Vector store initialization failed"
        
        progress(0.2, desc="Vector store ready. Processing stock codes...")
        
        # Create new fieldnames with stock_name next to SCTYCode
        scty_index = fieldnames.index('SCTYCode')
        new_fieldnames = list(fieldnames)
        new_fieldnames.insert(scty_index + 1, 'stock_name')
        
        # Process each row
        enriched_rows = []
        error_log = []
        stock_cache = {}  # Cache to avoid duplicate lookups
        total_rows = len(rows)
        
        for idx, row in enumerate(rows):
            progress((0.2 + (idx / total_rows) * 0.7), desc=f"Processing row {idx + 1}/{total_rows}...")
            
            stock_code = row.get('SCTYCode', '').strip()
            stock_name = ''
            
            if stock_code:
                # Normalize stock code
                normalized_code = normalize_stock_code(stock_code)
                
                # Check cache first
                if normalized_code in stock_cache:
                    stock_name = stock_cache[normalized_code]
                else:
                    # Query vector store
                    try:
                        result = verify_and_correct_stock(
                            stock_code=normalized_code,
                            vector_store=vector_store,
                            confidence_threshold=0.5,
                            top_k=1
                        )
                        
                        if result.corrected_stock_name:
                            stock_name = result.corrected_stock_name
                            stock_cache[normalized_code] = stock_name
                        else:
                            stock_name = 'N/A'
                            error_log.append(f"Row {idx + 1}: Could not find stock name for code '{stock_code}'")
                    
                    except Exception as e:
                        stock_name = 'ERROR'
                        error_log.append(f"Row {idx + 1}: Error processing '{stock_code}': {str(e)}")
            else:
                stock_name = ''
            
            # Create new row with stock_name inserted
            new_row = {}
            for field in new_fieldnames:
                if field == 'stock_name':
                    new_row[field] = stock_name
                else:
                    new_row[field] = row.get(field, '')
            
            enriched_rows.append(new_row)
        
        progress(0.9, desc="Writing enriched CSV...")
        
        # Write enriched CSV to output file
        output_filename = 'enriched_' + os.path.basename(csv_file)
        output_path = os.path.join(os.path.dirname(csv_file), output_filename)
        
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(enriched_rows)
        
        progress(1.0, desc="Complete!")
        
        # Generate status message
        unique_stocks = len(stock_cache)
        errors_count = len(error_log)
        error_text = '\n'.join(error_log) if error_log else 'No errors'
        
        status_msg = f"""✅ **CSV Enrichment Complete!**

📊 **Statistics:**
- Total rows processed: {total_rows}
- Unique stock codes found: {unique_stocks}
- Stock names added: {sum(1 for r in enriched_rows if r.get('stock_name') and r.get('stock_name') not in ['', 'N/A', 'ERROR'])}
- Errors/Not found: {errors_count}

📁 **Output file:** {output_filename}

Click the download button below to get your enriched CSV file.
"""
        
        return status_msg, output_path, error_text
    
    except Exception as e:
        error_msg = f"❌ Error processing CSV:\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None, traceback.format_exc()


def create_csv_stock_enrichment_tab():
    """Create the CSV Stock Name Enrichment tab"""
    
    with gr.Tab("CSV Stock Enrichment"):
        gr.Markdown(
            """
            ## 📊 CSV Stock Name Enrichment
            
            Upload a CSV file containing a **SCTYCode** column. This tool will:
            1. 🔍 Find all stock codes in the SCTYCode column
            2. 🔎 Look up stock names from the vector store
            3. ➕ Add a new **stock_name** column next to SCTYCode
            4. 💾 Generate an enriched CSV file for download
            
            **Supported format:** CSV files with a column named `SCTYCode`
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📁 Upload CSV File")
                
                csv_input = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"],
                    type="filepath",
                    file_count="single"
                )
                
                process_btn = gr.Button(
                    "🚀 Enrich CSV with Stock Names",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown(
                    """
                    **Note:** Large CSV files may take several minutes to process.
                    The vector store will be queried for each unique stock code.
                    """
                )
            
            with gr.Column(scale=1):
                gr.Markdown("#### 📊 Processing Results")
                
                status_box = gr.Textbox(
                    label="Status & Statistics",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )
                
                enriched_csv_output = gr.File(
                    label="Download Enriched CSV",
                    interactive=False
                )
                
                error_log_box = gr.Textbox(
                    label="Error Log",
                    lines=10,
                    interactive=False,
                    show_copy_button=True,
                    info="Errors encountered during processing (if any)"
                )
        
        # Connect the button
        process_btn.click(
            fn=enrich_csv_with_stock_names,
            inputs=[csv_input],
            outputs=[status_box, enriched_csv_output, error_log_box],
        )

