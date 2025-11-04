"""
Tab: Trade Verification
Verify transactions against actual trade records in trades.csv
"""

import json
import csv
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import gradio as gr
import os


def parse_broker_code(broker_id: str, ae_code: str) -> bool:
    """
    Check if broker_id matches AECode
    Example: broker_id="0489" should match ae_code="CK489"
    """
    if not broker_id or not ae_code:
        return False
    
    # Remove leading zeros from broker_id
    broker_clean = broker_id.lstrip('0')
    
    # Check if ae_code ends with the broker_id digits
    return ae_code.endswith(broker_clean) or ae_code.endswith(broker_id)


def parse_datetime(date_str: str) -> Optional[datetime]:
    """Parse datetime string in multiple formats"""
    formats = [
        "%Y-%m-%dT%H:%M:%S",  # ISO format
        "%Y-%m-%d %H:%M:%S.%f",  # trades.csv format
        "%Y-%m-%d %H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def match_transaction_to_trades(
    transaction: Dict,
    metadata: Dict,
    trades_file: str = "trades.csv",
    time_window_hours: float = 1.0
) -> Dict:
    """
    Match a single transaction against trades.csv records
    
    Args:
        transaction: Transaction dict with transaction_type, stock_code, quantity, price, etc.
        metadata: Metadata dict with client_id, broker_id, hkt_datetime, etc.
        trades_file: Path to trades.csv
        time_window_hours: Time window to search (before and after) in hours
    
    Returns:
        Dict with matching results
    """
    result = {
        "transaction": transaction,
        "matches": [],
        "analysis": {},
        "confidence_score": 0.0,
    }
    
    # Extract transaction details
    tx_type = transaction.get("transaction_type", "").lower()
    tx_stock_code = transaction.get("stock_code", "")
    tx_quantity = transaction.get("quantity", "")
    tx_price = transaction.get("price", "")
    
    # Extract metadata
    client_id = metadata.get("client_id", "")
    broker_id = metadata.get("broker_id", "")
    hkt_datetime_str = metadata.get("hkt_datetime", "")
    
    # Parse datetime
    if not hkt_datetime_str:
        result["analysis"]["error"] = "No HKT datetime provided in metadata"
        return result
    
    target_dt = parse_datetime(hkt_datetime_str)
    if not target_dt:
        result["analysis"]["error"] = f"Cannot parse datetime: {hkt_datetime_str}"
        return result
    
    # Define time window
    time_start = target_dt - timedelta(hours=time_window_hours)
    time_end = target_dt + timedelta(hours=time_window_hours)
    
    # Map transaction type to OrderSide
    order_side_map = {
        "buy": "B",  # Bid
        "sell": "A",  # Ask
        "queue": None,  # Queue could be either
    }
    expected_order_side = order_side_map.get(tx_type)
    
    # Normalize stock code (remove leading zeros for comparison)
    tx_stock_normalized = tx_stock_code.lstrip('0') if tx_stock_code else ""
    
    # Read and search trades.csv
    if not os.path.exists(trades_file):
        result["analysis"]["error"] = f"Trades file not found: {trades_file}"
        return result
    
    try:
        with open(trades_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse trade datetime
                trade_dt_str = row.get("OrderTime", "")
                trade_dt = parse_datetime(trade_dt_str)
                
                if not trade_dt:
                    continue
                
                # Check if within time window
                if not (time_start <= trade_dt <= time_end):
                    continue
                
                # Check client ID
                trade_client_id = row.get("ACCode", "")
                if trade_client_id != client_id:
                    continue
                
                # Check broker ID
                trade_broker_code = row.get("AECode", "")
                if not parse_broker_code(broker_id, trade_broker_code):
                    continue
                
                # If we reach here, this is a potential match
                # Now check the transaction details
                trade_stock_code = row.get("SCTYCode", "")
                trade_stock_normalized = trade_stock_code.lstrip('0') if trade_stock_code else ""
                trade_order_side = row.get("OrderSide", "")
                trade_qty = row.get("OrderQty", "")
                trade_price = row.get("OrderPrice", "")
                
                # Calculate match details
                match_details = {
                    "trade_record": row,
                    "matches": {},
                    "mismatches": {},
                    "partial_matches": {},
                }
                
                # Check stock code
                if tx_stock_normalized and trade_stock_normalized:
                    if tx_stock_normalized == trade_stock_normalized:
                        match_details["matches"]["stock_code"] = f"✅ {tx_stock_code} matches {trade_stock_code}"
                    else:
                        match_details["mismatches"]["stock_code"] = f"❌ {tx_stock_code} does NOT match {trade_stock_code}"
                
                # Check order side (buy/sell)
                if expected_order_side:
                    if trade_order_side == expected_order_side:
                        side_desc = "buy" if expected_order_side == "B" else "sell"
                        match_details["matches"]["order_side"] = f"✅ {side_desc} matches OrderSide={trade_order_side}"
                    else:
                        match_details["mismatches"]["order_side"] = f"❌ {tx_type} does NOT match OrderSide={trade_order_side}"
                else:
                    match_details["partial_matches"]["order_side"] = f"⚠️ Transaction type 'queue' - OrderSide={trade_order_side}"
                
                # Check quantity
                if tx_quantity and trade_qty:
                    # Normalize quantities (remove commas, convert to numbers)
                    try:
                        tx_qty_num = float(tx_quantity.replace(',', ''))
                        trade_qty_num = float(trade_qty.replace(',', ''))
                        
                        if tx_qty_num == trade_qty_num:
                            match_details["matches"]["quantity"] = f"✅ Quantity {tx_quantity} matches {trade_qty}"
                        else:
                            match_details["mismatches"]["quantity"] = f"❌ Quantity {tx_quantity} does NOT match {trade_qty}"
                    except ValueError:
                        match_details["partial_matches"]["quantity"] = f"⚠️ Cannot compare quantities: {tx_quantity} vs {trade_qty}"
                
                # Check price
                if tx_price and trade_price:
                    try:
                        tx_price_num = float(tx_price.replace(',', ''))
                        trade_price_num = float(trade_price.replace(',', ''))
                        
                        # Allow small tolerance for price matching (e.g., 0.01)
                        if abs(tx_price_num - trade_price_num) < 0.01:
                            match_details["matches"]["price"] = f"✅ Price {tx_price} matches {trade_price}"
                        else:
                            match_details["mismatches"]["price"] = f"❌ Price {tx_price} does NOT match {trade_price}"
                    except ValueError:
                        match_details["partial_matches"]["price"] = f"⚠️ Cannot compare prices: {tx_price} vs {trade_price}"
                
                # Check datetime (how close is it?)
                time_diff = abs((trade_dt - target_dt).total_seconds())
                time_diff_minutes = time_diff / 60
                
                if time_diff_minutes < 5:
                    match_details["matches"]["time"] = f"✅ Time difference: {time_diff_minutes:.1f} minutes (very close)"
                elif time_diff_minutes < 15:
                    match_details["partial_matches"]["time"] = f"⚠️ Time difference: {time_diff_minutes:.1f} minutes"
                else:
                    match_details["mismatches"]["time"] = f"❌ Time difference: {time_diff_minutes:.1f} minutes (far apart)"
                
                # Calculate confidence score for this match
                num_matches = len(match_details["matches"])
                num_mismatches = len(match_details["mismatches"])
                num_partial = len(match_details["partial_matches"])
                
                # Confidence calculation:
                # - Each full match adds points
                # - Each mismatch reduces points significantly
                # - Partial matches add smaller points
                confidence = 0.0
                
                # Base confidence from client/broker/time window match
                confidence += 20.0
                
                # Add points for each field match
                field_weights = {
                    "stock_code": 30.0,
                    "order_side": 20.0,
                    "quantity": 15.0,
                    "price": 10.0,
                    "time": 5.0,
                }
                
                for field, weight in field_weights.items():
                    if field in match_details["matches"]:
                        confidence += weight
                    elif field in match_details["mismatches"]:
                        confidence -= weight * 1.5  # Penalties are stronger
                    elif field in match_details["partial_matches"]:
                        confidence += weight * 0.3
                
                # Cap confidence between 0 and 100
                confidence = max(0.0, min(100.0, confidence))
                
                match_details["confidence"] = confidence
                
                # Add to matches list
                result["matches"].append(match_details)
        
        # Sort matches by confidence score (highest first)
        result["matches"].sort(key=lambda x: x["confidence"], reverse=True)
        
        # Overall analysis
        if len(result["matches"]) == 0:
            result["analysis"]["summary"] = "❌ No matching trade records found in the time window"
            result["confidence_score"] = 0.0
        elif len(result["matches"]) == 1:
            best_match = result["matches"][0]
            result["confidence_score"] = best_match["confidence"]
            result["analysis"]["summary"] = f"✅ Found 1 potential match with {result['confidence_score']:.1f}% confidence"
        else:
            best_match = result["matches"][0]
            result["confidence_score"] = best_match["confidence"]
            result["analysis"]["summary"] = f"⚠️ Found {len(result['matches'])} potential matches. Best match: {result['confidence_score']:.1f}% confidence"
        
    except Exception as e:
        result["analysis"]["error"] = f"Error reading trades.csv: {str(e)}"
    
    return result


def verify_transactions(
    transaction_json: str,
    metadata_json: str,
    trades_file_path: str,
    time_window: float,
) -> tuple[str, str, str]:
    """
    Main function to verify transactions against trade records
    
    Returns:
        tuple: (formatted_text_result, json_analysis_result, csv_records_json)
    """
    try:
        # Parse transaction JSON
        if not transaction_json.strip():
            error_msg = "❌ Error: Please provide transaction analysis JSON"
            return (error_msg, "", "")
        
        try:
            transaction_data = json.loads(transaction_json)
        except json.JSONDecodeError as e:
            error_msg = f"❌ Error: Cannot parse transaction JSON: {str(e)}"
            return (error_msg, "", "")
        
        # Parse metadata JSON
        if not metadata_json.strip():
            error_msg = "❌ Error: Please provide metadata JSON"
            return (error_msg, "", "")
        
        try:
            metadata = json.loads(metadata_json)
            # Extract metadata if it's nested
            if "metadata" in metadata:
                metadata = metadata["metadata"]
        except json.JSONDecodeError as e:
            error_msg = f"❌ Error: Cannot parse metadata JSON: {str(e)}"
            return (error_msg, "", "")
        
        # Extract transactions list
        transactions = transaction_data.get("transactions", [])
        
        if len(transactions) == 0:
            error_msg = "❌ No transactions found in the transaction analysis JSON"
            return (error_msg, "", "")
        
        # Verify trades file exists
        if not os.path.exists(trades_file_path):
            error_msg = f"❌ Error: Trades file not found: {trades_file_path}"
            return (error_msg, "", "")
        
        # Build results structures
        results_text = f"""{'='*80}
📋 TRADE VERIFICATION RESULTS
{'='*80}

📂 Trades File: {trades_file_path}
⏰ Time Window: ±{time_window} hours
👤 Client ID: {metadata.get('client_id', 'N/A')}
👔 Broker ID: {metadata.get('broker_id', 'N/A')}
📅 HKT DateTime: {metadata.get('hkt_datetime', 'N/A')}

{'='*80}
📊 FOUND {len(transactions)} TRANSACTION(S) TO VERIFY
{'='*80}

"""
        
        # JSON analysis structure
        json_analysis = {
            "status": "success",
            "verification_info": {
                "trades_file": trades_file_path,
                "time_window_hours": time_window,
                "client_id": metadata.get('client_id', 'N/A'),
                "broker_id": metadata.get('broker_id', 'N/A'),
                "hkt_datetime": metadata.get('hkt_datetime', 'N/A'),
                "total_transactions": len(transactions)
            },
            "transaction_verifications": []
        }
        
        # CSV records structure
        csv_records = {
            "total_matched_records": 0,
            "matched_csv_records": []
        }
        
        # Process each transaction
        for idx, transaction in enumerate(transactions, 1):
            results_text += f"\n{'─'*80}\n"
            results_text += f"🔍 TRANSACTION #{idx}\n"
            results_text += f"{'─'*80}\n"
            
            # Show transaction details
            results_text += f"Type: {transaction.get('transaction_type', 'N/A')}\n"
            results_text += f"Stock Code: {transaction.get('stock_code', 'N/A')}\n"
            results_text += f"Stock Name: {transaction.get('stock_name', 'N/A')}\n"
            results_text += f"Quantity: {transaction.get('quantity', 'N/A')}\n"
            results_text += f"Price: {transaction.get('price', 'N/A')}\n"
            results_text += f"LLM Confidence: {transaction.get('confidence_score', 'N/A')}/2.0\n"
            results_text += f"\n"
            
            # Match against trades
            match_result = match_transaction_to_trades(
                transaction, 
                metadata, 
                trades_file_path,
                time_window
            )
            
            # Build JSON verification for this transaction
            verification_json = {
                "transaction_index": idx,
                "transaction_details": {
                    "type": transaction.get('transaction_type', 'N/A'),
                    "stock_code": transaction.get('stock_code', 'N/A'),
                    "stock_name": transaction.get('stock_name', 'N/A'),
                    "quantity": transaction.get('quantity', 'N/A'),
                    "price": transaction.get('price', 'N/A'),
                    "llm_confidence_score": transaction.get('confidence_score', 'N/A'),
                    "explanation": transaction.get('explanation', '')
                },
                "verification_summary": {
                    "status": "error" if "error" in match_result["analysis"] else "completed",
                    "error_message": match_result["analysis"].get("error", None),
                    "summary": match_result["analysis"].get("summary", ""),
                    "total_matches_found": len(match_result["matches"]),
                    "best_match_confidence": match_result["matches"][0]["confidence"] if match_result["matches"] else 0.0
                },
                "matched_records": []
            }
            
            # Show analysis
            if "error" in match_result["analysis"]:
                results_text += f"❌ ERROR: {match_result['analysis']['error']}\n"
                json_analysis["transaction_verifications"].append(verification_json)
                continue
            
            results_text += f"{match_result['analysis'].get('summary', '')}\n\n"
            
            # Show matches
            if len(match_result["matches"]) == 0:
                results_text += "❌ No matching trade records found.\n"
                results_text += "Possible reasons:\n"
                results_text += "  - Trade was not executed\n"
                results_text += "  - Trade time is outside the time window\n"
                results_text += "  - Client ID or Broker ID does not match\n"
                results_text += "  - Trade was executed in a different system\n"
            else:
                # Show each match
                for match_idx, match in enumerate(match_result["matches"], 1):
                    results_text += f"\n{'─'*60}\n"
                    results_text += f"📌 POTENTIAL MATCH #{match_idx} - Confidence: {match['confidence']:.1f}%\n"
                    results_text += f"{'─'*60}\n"
                    
                    trade = match["trade_record"]
                    results_text += f"\n📄 Trade Record Details:\n"
                    results_text += f"  Order No: {trade.get('OrderNo', 'N/A')}\n"
                    results_text += f"  Order Time: {trade.get('OrderTime', 'N/A')}\n"
                    results_text += f"  Stock Code: {trade.get('SCTYCode', 'N/A')}\n"
                    results_text += f"  Order Side: {trade.get('OrderSide', 'N/A')} ({'Buy/Bid' if trade.get('OrderSide')=='B' else 'Sell/Ask' if trade.get('OrderSide')=='A' else 'Unknown'})\n"
                    results_text += f"  Quantity: {trade.get('OrderQty', 'N/A')}\n"
                    results_text += f"  Price: {trade.get('OrderPrice', 'N/A')}\n"
                    results_text += f"  Status: {trade.get('OrderStatus', 'N/A')}\n"
                    results_text += f"  AC Code: {trade.get('ACCode', 'N/A')}\n"
                    results_text += f"  AE Code: {trade.get('AECode', 'N/A')}\n"
                    
                    # Show matches
                    if match["matches"]:
                        results_text += f"\n✅ MATCHES:\n"
                        for field, desc in match["matches"].items():
                            results_text += f"  {desc}\n"
                    
                    # Show mismatches
                    if match["mismatches"]:
                        results_text += f"\n❌ MISMATCHES:\n"
                        for field, desc in match["mismatches"].items():
                            results_text += f"  {desc}\n"
                    
                    # Show partial matches
                    if match["partial_matches"]:
                        results_text += f"\n⚠️ PARTIAL MATCHES:\n"
                        for field, desc in match["partial_matches"].items():
                            results_text += f"  {desc}\n"
                    
                    results_text += f"\n"
                    
                    # Add to JSON structures
                    match_json = {
                        "match_index": match_idx,
                        "confidence_score": match["confidence"],
                        "comparison": {
                            "matches": match["matches"],
                            "mismatches": match["mismatches"],
                            "partial_matches": match["partial_matches"]
                        }
                    }
                    verification_json["matched_records"].append(match_json)
                    
                    # Add CSV record
                    csv_record = {
                        "transaction_index": idx,
                        "match_index": match_idx,
                        "confidence_score": match["confidence"],
                        "csv_data": trade
                    }
                    csv_records["matched_csv_records"].append(csv_record)
            
            # Add verification to JSON analysis
            json_analysis["transaction_verifications"].append(verification_json)
        
        results_text += f"\n{'='*80}\n"
        results_text += f"✅ VERIFICATION COMPLETE\n"
        results_text += f"{'='*80}\n"
        
        # Update CSV records count
        csv_records["total_matched_records"] = len(csv_records["matched_csv_records"])
        
        # Format JSON outputs
        json_analysis_str = json.dumps(json_analysis, indent=2, ensure_ascii=False)
        csv_records_str = json.dumps(csv_records, indent=2, ensure_ascii=False)
        
        return (results_text, json_analysis_str, csv_records_str)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
        error_json = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        error_json_str = json.dumps(error_json, indent=2, ensure_ascii=False)
        return (error_msg, error_json_str, "")


def create_trade_verification_tab():
    """Create and return the Trade Verification tab"""
    with gr.Tab("🔍 Trade Verification"):
        gr.Markdown(
            """
            ### 交易記錄核對 - Verify Transactions Against Trade Records
            
            This tool matches transactions identified from call recordings against actual trade records in `trades.csv`.
            
            **Matching Criteria:**
            - ✅ **DateTime**: OrderTime column (HKT) within time window
            - ✅ **Client ID**: ACCode column matches client_id
            - ✅ **Broker ID**: AECode column matches broker_id (e.g., CK489 matches 0489)
            - ✅ **Stock Code**: SCTYCode column matches stock_code
            - ✅ **Order Side**: 'A' = Sell (Ask), 'B' = Buy (Bid)
            - ✅ **Quantity**: OrderQty column matches quantity
            - ✅ **Price**: OrderPrice column matches price
            
            **Output:**
            - Shows all potential matching records
            - Highlights what matches ✅ and what doesn't ❌
            - Provides confidence score (0-100%) for each match
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📥 Input Data")
                
                transaction_json_box = gr.Textbox(
                    label="Transaction Analysis JSON",
                    placeholder='{\n  "transactions": [\n    {\n      "transaction_type": "buy",\n      "confidence_score": 2.0,\n      "stock_code": "18138",\n      ...\n    }\n  ],\n  ...\n}',
                    lines=15,
                    info="Paste the JSON output from Transaction Analysis tab"
                )
                
                metadata_json_box = gr.Textbox(
                    label="Call Metadata JSON",
                    placeholder='{\n  "metadata": {\n    "filename": "...",\n    "broker_id": "0489",\n    "client_id": "P77197",\n    "hkt_datetime": "2025-10-20T10:01:20",\n    ...\n  }\n}',
                    lines=12,
                    info="Provide client/broker information and call datetime"
                )
                
                gr.Markdown("#### ⚙️ Settings")
                
                trades_file_box = gr.Textbox(
                    label="Trades CSV File Path",
                    value="trades.csv",
                    placeholder="trades.csv",
                    info="Path to the trades CSV file"
                )
                
                time_window_slider = gr.Slider(
                    minimum=0.5,
                    maximum=24.0,
                    value=1.0,
                    step=0.5,
                    label="Time Window (hours)",
                    info="Search for trades within ±X hours of the call time"
                )
                
                verify_btn = gr.Button(
                    "🔍 Verify Transactions",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("#### 📊 Verification Results")
                
                results_box = gr.Textbox(
                    label="Matching Results (Formatted Text)",
                    lines=20,
                    interactive=False,
                    show_copy_button=True,
                )
                
                json_analysis_box = gr.Textbox(
                    label="Analysis Result (JSON Format)",
                    lines=20,
                    interactive=False,
                    show_copy_button=True,
                    info="Complete analysis with all matching details in JSON format"
                )
                
                csv_records_box = gr.Textbox(
                    label="CSV Data Records (JSON Format)",
                    lines=20,
                    interactive=False,
                    show_copy_button=True,
                    info="All matched CSV records with complete data from trades.csv"
                )
        
        # Connect the button
        verify_btn.click(
            fn=verify_transactions,
            inputs=[
                transaction_json_box,
                metadata_json_box,
                trades_file_box,
                time_window_slider,
            ],
            outputs=[
                results_box,
                json_analysis_box,
                csv_records_box,
            ],
        )

