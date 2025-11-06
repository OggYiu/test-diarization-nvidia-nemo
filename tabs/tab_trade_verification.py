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
import logging

# Import embedding functionality for stock name similarity
try:
    from langchain_ollama import OllamaEmbeddings
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Embeddings not available. Stock name similarity will be disabled.")


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


# Global embedding model instance (lazy initialization)
_embeddings_model = None


def get_embeddings_model():
    """Get or initialize the embeddings model"""
    global _embeddings_model
    if _embeddings_model is None and EMBEDDINGS_AVAILABLE:
        try:
            _embeddings_model = OllamaEmbeddings(model="qwen3-embedding:8b")
            logging.info("✅ Embeddings model initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize embeddings model: {e}")
            return None
    return _embeddings_model


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two text strings using embeddings.
    
    Args:
        text1: First text string
        text2: Second text string
    
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    if not EMBEDDINGS_AVAILABLE:
        return 0.0
    
    embeddings = get_embeddings_model()
    if embeddings is None:
        return 0.0
    
    try:
        # Get embeddings for both texts
        emb1 = embeddings.embed_query(text1)
        emb2 = embeddings.embed_query(text2)
        
        # Compute cosine similarity
        emb1_array = np.array(emb1)
        emb2_array = np.array(emb2)
        
        # Normalize vectors
        emb1_norm = emb1_array / np.linalg.norm(emb1_array)
        emb2_norm = emb2_array / np.linalg.norm(emb2_array)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Ensure it's between 0 and 1
        return max(0.0, min(1.0, float(similarity)))
        
    except Exception as e:
        logging.error(f"Error computing text similarity: {e}")
        return 0.0


def compute_stock_name_similarity(tx_stock_name: str, trade_stock_name: str) -> tuple[float, str]:
    """
    Compute stock name similarity and return score + description
    
    Args:
        tx_stock_name: Stock name from transaction
        trade_stock_name: Stock name from trade record
    
    Returns:
        Tuple of (similarity_score, description_text)
    """
    if not tx_stock_name or not trade_stock_name:
        return 0.0, "⚠️ Stock name missing"
    
    # Exact match (case-insensitive)
    if tx_stock_name.strip().lower() == trade_stock_name.strip().lower():
        return 1.0, f"✅ Exact match: '{tx_stock_name}'"
    
    # Check if one contains the other (partial match)
    tx_clean = tx_stock_name.strip().lower()
    trade_clean = trade_stock_name.strip().lower()
    
    if tx_clean in trade_clean or trade_clean in tx_clean:
        return 0.9, f"✅ Partial match: '{tx_stock_name}' ↔ '{trade_stock_name}'"
    
    # Compute embedding similarity
    similarity = compute_text_similarity(tx_stock_name, trade_stock_name)
    
    if similarity >= 0.8:
        return similarity, f"✅ High similarity ({similarity:.2f}): '{tx_stock_name}' ↔ '{trade_stock_name}'"
    elif similarity >= 0.6:
        return similarity, f"⚠️ Medium similarity ({similarity:.2f}): '{tx_stock_name}' ↔ '{trade_stock_name}'"
    else:
        return similarity, f"❌ Low similarity ({similarity:.2f}): '{tx_stock_name}' ↔ '{trade_stock_name}'"


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
    
    Primary Filters (Required):
    1. Client ID (ACCode) must match metadata.client_id
    2. DATE in OrderTime must match DATE in hkt_datetime (ignoring time)
    
    Secondary Filter:
    3. Time window filters the date-matched records for detailed comparison
    
    Args:
        transaction: Transaction dict with transaction_type, stock_code, quantity, price, hkt_datetime, etc.
        metadata: Metadata dict with client_id, broker_id, hkt_datetime (fallback), etc.
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
        "all_client_records": [],  # ALL records found for this client in trades.csv
    }
    
    # Extract transaction details
    tx_type = transaction.get("transaction_type", "").lower()
    tx_stock_code = transaction.get("stock_code", "")
    tx_stock_name = transaction.get("stock_name", "")
    tx_quantity = transaction.get("quantity", "")
    tx_price = transaction.get("price", "")
    
    # Extract metadata
    client_id = metadata.get("client_id", "")
    broker_id = metadata.get("broker_id", "")
    
    # Priority: Use hkt_datetime from transaction itself, fall back to metadata
    hkt_datetime_str = transaction.get("hkt_datetime") or metadata.get("hkt_datetime", "")
    
    # Parse datetime
    if not hkt_datetime_str or hkt_datetime_str == "N/A":
        result["analysis"]["error"] = "No HKT datetime provided in transaction or metadata"
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
    
    # First pass: find ALL records with this client ID AND matching date
    all_client_records = []
    client_records_in_window = []
    
    # Extract target date (date only, no time)
    target_date = target_dt.date()
    
    try:
        with open(trades_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Check client ID first
                trade_client_id = row.get("ACCode", "")
                if trade_client_id != client_id:
                    continue
                
                # Parse trade datetime
                trade_dt_str = row.get("OrderTime", "")
                trade_dt = parse_datetime(trade_dt_str)
                
                if not trade_dt:
                    continue
                
                # Check if DATE matches (not time, just date)
                trade_date = trade_dt.date()
                if trade_date != target_date:
                    continue
                
                # This record belongs to the client AND has matching date
                all_client_records.append(row.copy())
                
                # Check if within time window (for further filtering)
                if not (time_start <= trade_dt <= time_end):
                    continue
                
                # This record is within the time window
                client_records_in_window.append((row, trade_dt))
        
        # Store statistics about client records
        result["analysis"]["total_client_records"] = len(all_client_records)
        result["analysis"]["client_records_in_time_window"] = len(client_records_in_window)
        
        # Store ALL client records found
        result["all_client_records"] = all_client_records
        
        # Now process ALL records that are in the time window
        for row, trade_dt in client_records_in_window:
                # Extract trade details
                trade_broker_code = row.get("AECode", "")
                trade_stock_code = row.get("SCTYCode", "")
                trade_stock_normalized = trade_stock_code.lstrip('0') if trade_stock_code else ""
                trade_stock_name = row.get("stock_name", "")
                trade_order_side = row.get("OrderSide", "")
                trade_qty = row.get("OrderQty", "")
                trade_price = row.get("OrderPrice", "")
                
                # Calculate match details
                match_details = {
                    "trade_record": row,
                    "matches": {},
                    "mismatches": {},
                    "partial_matches": {},
                    "stock_name_similarity": 0.0,  # Store similarity score
                }
                
                # Check broker ID
                if broker_id and trade_broker_code:
                    if parse_broker_code(broker_id, trade_broker_code):
                        match_details["matches"]["broker_id"] = f"✅ Broker ID {broker_id} matches AECode {trade_broker_code}"
                    else:
                        match_details["mismatches"]["broker_id"] = f"❌ Broker ID {broker_id} does NOT match AECode {trade_broker_code}"
                
                # Check stock code
                stock_code_matches = False
                if tx_stock_normalized and trade_stock_normalized:
                    if tx_stock_normalized == trade_stock_normalized:
                        match_details["matches"]["stock_code"] = f"✅ {tx_stock_code} matches {trade_stock_code}"
                        stock_code_matches = True
                    else:
                        match_details["mismatches"]["stock_code"] = f"❌ {tx_stock_code} does NOT match {trade_stock_code}"
                
                # Check stock name similarity (especially useful when stock code doesn't match)
                if tx_stock_name and trade_stock_name:
                    name_similarity, name_desc = compute_stock_name_similarity(tx_stock_name, trade_stock_name)
                    match_details["stock_name_similarity"] = name_similarity
                    
                    # If stock code matches, name similarity is confirmatory
                    # If stock code doesn't match, high name similarity can suggest the code was mis-transcribed
                    if name_similarity >= 0.8:
                        match_details["matches"]["stock_name"] = name_desc
                        if not stock_code_matches:
                            match_details["partial_matches"]["stock_code_override"] = (
                                f"⚠️ Stock codes don't match, but names are very similar "
                                f"(similarity: {name_similarity:.2f}) - possible STT error in code"
                            )
                    elif name_similarity >= 0.6:
                        match_details["partial_matches"]["stock_name"] = name_desc
                    else:
                        match_details["mismatches"]["stock_name"] = name_desc
                elif tx_stock_name or trade_stock_name:
                    match_details["partial_matches"]["stock_name"] = "⚠️ Stock name missing in one of the records"
                
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
                    "broker_id": 20.0,
                    "stock_code": 20.0,
                    "stock_name": 20.0,  # NEW: Stock name similarity
                    "order_side": 12.0,
                    "quantity": 12.0,
                    "price": 8.0,
                    "time": 8.0,
                }
                
                for field, weight in field_weights.items():
                    if field in match_details["matches"]:
                        confidence += weight
                    elif field in match_details["mismatches"]:
                        confidence -= weight * 1.5  # Penalties are stronger
                    elif field in match_details["partial_matches"]:
                        confidence += weight * 0.3
                
                # Bonus: If stock name similarity is high (>=0.8) even if code doesn't match,
                # this suggests a likely match with transcription error
                name_similarity = match_details.get("stock_name_similarity", 0.0)
                if name_similarity >= 0.8 and not stock_code_matches:
                    # Add bonus points for high name similarity when code doesn't match
                    confidence += 15.0
                    logging.info(f"High name similarity ({name_similarity:.2f}) with code mismatch - adding bonus confidence")
                
                # Cap confidence between 0 and 100
                confidence = max(0.0, min(100.0, confidence))
                
                match_details["confidence"] = confidence
                
                # Add to matches list
                result["matches"].append(match_details)
        
        # Sort matches by confidence score (highest first)
        result["matches"].sort(key=lambda x: x["confidence"], reverse=True)
        
        # Overall analysis
        total_client_recs = result["analysis"].get("total_client_records", 0)
        recs_in_window = result["analysis"].get("client_records_in_time_window", 0)
        
        if len(result["matches"]) == 0:
            result["analysis"]["summary"] = f"❌ No matching trade records found in the time window\n"
            result["analysis"]["summary"] += f"   📊 Client has {total_client_recs} total records in trades.csv ({recs_in_window} within time window)"
            result["confidence_score"] = 0.0
        elif len(result["matches"]) == 1:
            best_match = result["matches"][0]
            result["confidence_score"] = best_match["confidence"]
            result["analysis"]["summary"] = f"✅ Found 1 potential match with {result['confidence_score']:.1f}% confidence\n"
            result["analysis"]["summary"] += f"   📊 Client has {total_client_recs} total records in trades.csv ({recs_in_window} within time window)"
        else:
            best_match = result["matches"][0]
            result["confidence_score"] = best_match["confidence"]
            result["analysis"]["summary"] = f"⚠️ Found {len(result['matches'])} potential matches. Best match: {result['confidence_score']:.1f}% confidence\n"
            result["analysis"]["summary"] += f"   📊 Client has {total_client_recs} total records in trades.csv ({recs_in_window} within time window)"
        
    except Exception as e:
        result["analysis"]["error"] = f"Error reading trades.csv: {str(e)}"
    
    return result


def save_to_report_csv(verification_data: Dict, metadata: Dict, report_file: str = "report.csv") -> str:
    """
    Save verification results to report.csv with UTF-8 encoding for Chinese characters.
    
    Duplicate Detection Logic:
    - Checks if existing records have the same (client_id, broker_id, hkt_datetime)
    - If found, DELETE ALL matching records first
    - Then add all new records for this client/broker/datetime combination
    
    Args:
        verification_data: The JSON analysis dict from verify_transactions
        metadata: The metadata dict with client info
        report_file: Path to the report CSV file
    
    Returns:
        Status message
    """
    try:
        # Prepare rows to write
        new_rows = []
        
        client_id = metadata.get("client_id", "N/A")
        broker_id = metadata.get("broker_id", "N/A")
        
        for tx_verification in verification_data.get("transaction_verifications", []):
            tx_details = tx_verification.get("transaction_details", {})
            tx_summary = tx_verification.get("verification_summary", {})
            
            # Extract transaction info
            hkt_datetime = tx_details.get("hkt_datetime", "N/A")
            stock_code = tx_details.get("stock_code", "N/A")
            stock_name = tx_details.get("stock_name", "N/A")
            tx_type = tx_details.get("type", "N/A")
            quantity = tx_details.get("quantity", "N/A")
            price = tx_details.get("price", "N/A")
            llm_confidence = tx_details.get("llm_confidence_score", "N/A")
            
            # Extract verification results
            total_matches = tx_summary.get("total_matches_found", 0)
            best_match_confidence = tx_summary.get("best_match_confidence", 0.0)
            status = tx_summary.get("status", "N/A")
            summary = tx_summary.get("summary", "").replace("\n", " ")
            
            # Get best match details if available
            best_match_stock_name_similarity = 0.0
            best_match_order_no = "N/A"
            best_match_order_time = "N/A"
            
            matched_records = tx_verification.get("matched_records", [])
            if matched_records:
                best_match = matched_records[0]  # First one is the best
                best_match_stock_name_similarity = best_match.get("stock_name_similarity", 0.0)
                # We don't have direct access to trade record here, so we'll leave these as N/A
            
            # Create row
            row = {
                "client_id": client_id,
                "broker_id": broker_id,
                "hkt_datetime": hkt_datetime,
                "transaction_type": tx_type,
                "stock_code": stock_code,
                "stock_name": stock_name,
                "quantity": quantity,
                "price": price,
                "llm_confidence_score": llm_confidence,
                # "total_matches_found": total_matches,
                "best_match_confidence": f"{best_match_confidence:.2f}",
            }
            new_rows.append(row)
        
        if not new_rows:
            return "⚠️ No transactions to save to report"
        
        # Define CSV columns
        fieldnames = [
            "client_id",
            "broker_id",
            "hkt_datetime",
            "transaction_type",
            "stock_code",
            "stock_name",
            "quantity",
            "price",
            "llm_confidence_score",
            "total_matches_found",
            "best_match_confidence",
        ]
        
        # Read existing records if file exists
        existing_rows = []
        file_exists = os.path.exists(report_file)
        
        if file_exists:
            try:
                with open(report_file, 'r', encoding='utf-8-sig', newline='') as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)
            except Exception as e:
                logging.warning(f"Could not read existing report.csv: {e}")
        
        # Determine which records to delete based on (client_id, broker_id, hkt_datetime)
        # Get the unique combination from new rows
        new_keys_to_delete = set()
        for new_row in new_rows:
            key = (
                new_row["client_id"],
                new_row["broker_id"],
                new_row["hkt_datetime"]
            )
            new_keys_to_delete.add(key)
        
        # Filter out existing records that match the deletion criteria
        deleted_count = 0
        filtered_existing_rows = []
        for row in existing_rows:
            key = (
                row.get("client_id", ""),
                row.get("broker_id", ""),
                row.get("hkt_datetime", "")
            )
            if key in new_keys_to_delete:
                # This record should be deleted
                deleted_count += 1
            else:
                # Keep this record
                filtered_existing_rows.append(row)
        
        # Add all new rows
        final_rows = filtered_existing_rows + new_rows
        added_count = len(new_rows)
        
        # Write all records back to file with UTF-8-BOM for better Excel compatibility with Chinese characters
        # Using utf-8-sig adds a BOM (Byte Order Mark) which helps Excel detect UTF-8 encoding
        try:
            # Try UTF-8-SIG first (best for modern Excel and international use)
            with open(report_file, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(final_rows)
        except UnicodeEncodeError:
            # Fallback to GB2312 if UTF-8 fails (for Chinese Windows Excel)
            logging.warning("UTF-8 encoding failed, trying GB2312 for Chinese Excel compatibility")
            with open(report_file, 'w', encoding='gb2312', newline='', errors='ignore') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(final_rows)
        
        # Build status message
        status_msg = f"✅ Report saved to {report_file} (UTF-8 encoding)\n"
        status_msg += f"   📝 Added: {added_count} new record(s)\n"
        status_msg += f"   🗑️ Deleted: {deleted_count} old record(s) with same client_id + broker_id + hkt_datetime\n"
        status_msg += f"   📊 Total records in report: {len(final_rows)}"
        
        return status_msg
        
    except Exception as e:
        error_msg = f"❌ Error saving to report.csv: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return error_msg


def verify_transactions(
    transaction_json: str,
    trades_file_path: str,
    time_window: float,
) -> tuple[str, str, str, str, str]:
    """
    Main function to verify transactions against trade records
    
    Returns:
        tuple: (formatted_text_result, json_analysis_result, csv_records_json, all_client_records_json, report_status)
    """
    try:
        # Parse transaction JSON
        if not transaction_json.strip():
            error_msg = "❌ Error: Please provide transaction analysis JSON"
            return (error_msg, "", "", "", "")
        
        try:
            transaction_data = json.loads(transaction_json)
        except json.JSONDecodeError as e:
            error_msg = f"❌ Error: Cannot parse transaction JSON: {str(e)}"
            return (error_msg, "", "", "", "")
        
        # Extract transactions list
        transactions = transaction_data.get("transactions", [])
        
        if len(transactions) == 0:
            error_msg = "❌ No transactions found in the transaction analysis JSON"
            return (error_msg, "", "", "", "")
        
        # Extract metadata from first transaction
        first_transaction = transactions[0]
        metadata = {
            "client_id": first_transaction.get("client_id", "N/A"),
            "client_name": first_transaction.get("client_name", "N/A"),
            "broker_id": first_transaction.get("broker_id", "N/A"),
            "broker_name": first_transaction.get("broker_name", "N/A"),
            "hkt_datetime": first_transaction.get("hkt_datetime", "N/A"),
            "conversation_number": first_transaction.get("conversation_number", "N/A")
        }
        
        # Verify trades file exists
        if not os.path.exists(trades_file_path):
            error_msg = f"❌ Error: Trades file not found: {trades_file_path}"
            return (error_msg, "", "", "", "")
        
        # Build results structures
        results_text = f"""{'='*80}
📋 TRADE VERIFICATION RESULTS
{'='*80}

📂 Trades File: {trades_file_path}
⏰ Time Window: ±{time_window} hours
👤 Client ID: {metadata.get('client_id', 'N/A')}
👤 Client Name: {metadata.get('client_name', 'N/A')}
👔 Broker ID: {metadata.get('broker_id', 'N/A')}
👔 Broker Name: {metadata.get('broker_name', 'N/A')}
📅 HKT DateTime: {metadata.get('hkt_datetime', 'N/A')}
🔢 Conversation Number: {metadata.get('conversation_number', 'N/A')}

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
                "client_name": metadata.get('client_name', 'N/A'),
                "broker_id": metadata.get('broker_id', 'N/A'),
                "broker_name": metadata.get('broker_name', 'N/A'),
                "hkt_datetime": metadata.get('hkt_datetime', 'N/A'),
                "conversation_number": metadata.get('conversation_number', 'N/A'),
                "total_transactions": len(transactions)
            },
            "transaction_verifications": []
        }
        
        # CSV records structure
        csv_records = {
            "total_matched_records": 0,
            "matched_csv_records": []
        }
        
        # All client records structure (to collect ALL records found in trades.csv)
        all_client_records_data = {
            "client_id": metadata.get('client_id', 'N/A'),
            "total_records_found": 0,
            "all_records": []
        }
        
        # Process each transaction
        for idx, transaction in enumerate(transactions, 1):
            results_text += f"\n{'─'*80}\n"
            results_text += f"🔍 TRANSACTION #{idx}\n"
            results_text += f"{'─'*80}\n"
            
            # Show transaction details with datetime
            tx_datetime = transaction.get('hkt_datetime') or metadata.get('hkt_datetime', 'N/A')
            results_text += f"📅 DateTime (HKT): {tx_datetime}\n"
            results_text += f"Type: {transaction.get('transaction_type', 'N/A')}\n"
            results_text += f"Stock Code: {transaction.get('stock_code', 'N/A')}\n"
            results_text += f"Stock Name: {transaction.get('stock_name', 'N/A')}\n"
            results_text += f"Quantity: {transaction.get('quantity', 'N/A')}\n"
            results_text += f"Price: {transaction.get('price', 'N/A')}\n"
            results_text += f"LLM Confidence: {transaction.get('confidence_score', 'N/A')}/2.0\n"
            results_text += f"\n🔎 Searching trades.csv for client {metadata.get('client_id', 'N/A')}...\n"
            results_text += f"   Using time window: ±{time_window} hours from {tx_datetime}\n"
            
            # Match against trades
            match_result = match_transaction_to_trades(
                transaction, 
                metadata, 
                trades_file_path,
                time_window
            )
            
            # Collect all client records (only do this once, on first transaction)
            if idx == 1 and match_result.get("all_client_records"):
                all_client_records_data["all_records"] = match_result["all_client_records"]
                all_client_records_data["total_records_found"] = len(match_result["all_client_records"])
            
            # Build JSON verification for this transaction
            tx_datetime = transaction.get('hkt_datetime') or metadata.get('hkt_datetime', 'N/A')
            verification_json = {
                "transaction_index": idx,
                "transaction_details": {
                    "hkt_datetime": tx_datetime,
                    "type": transaction.get('transaction_type', 'N/A'),
                    "stock_code": transaction.get('stock_code', 'N/A'),
                    "stock_name": transaction.get('stock_name', 'N/A'),
                    "quantity": transaction.get('quantity', 'N/A'),
                    "price": transaction.get('price', 'N/A'),
                    "llm_confidence_score": transaction.get('confidence_score', 'N/A'),
                    "explanation": transaction.get('explanation', '')
                },
                "client_record_statistics": {
                    "total_client_records_in_csv": match_result["analysis"].get("total_client_records", 0),
                    "records_within_time_window": match_result["analysis"].get("client_records_in_time_window", 0),
                    "records_matching_all_criteria": len(match_result["matches"])
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
                total_client_recs = match_result["analysis"].get("total_client_records", 0)
                recs_in_window = match_result["analysis"].get("client_records_in_time_window", 0)
                
                results_text += "❌ No matching trade records found.\n\n"
                results_text += f"📊 Client Record Statistics:\n"
                results_text += f"  - Total records for this client in trades.csv: {total_client_recs}\n"
                results_text += f"  - Records within time window (±{time_window}h): {recs_in_window}\n"
                results_text += f"  - Records with sufficient confidence score: 0\n\n"
                results_text += "Possible reasons for no match:\n"
                results_text += "  - Trade was not executed (pending/cancelled)\n"
                if recs_in_window == 0 and total_client_recs > 0:
                    results_text += f"  - ⚠️ All {total_client_recs} client records are OUTSIDE the time window\n"
                    results_text += f"    → Consider increasing the time window or checking the call datetime\n"
                elif recs_in_window > 0:
                    results_text += f"  - ⚠️ Found {recs_in_window} record(s) in time window but none passed matching criteria\n"
                    results_text += f"    → All records failed one or more checks (broker ID, stock code, quantity, price, order type)\n"
                    results_text += f"    → Try checking the 'All Client Records' below for details\n"
                else:
                    results_text += "  - Client ID may be incorrect\n"
                    results_text += "  - Trade may be in a different system\n"
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
                    results_text += f"  Stock Name: {trade.get('stock_name', 'N/A')}\n"
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
                        "stock_name_similarity": match.get("stock_name_similarity", 0.0),
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
        all_client_records_str = json.dumps(all_client_records_data, indent=2, ensure_ascii=False)
        
        # Save to report.csv
        report_status = save_to_report_csv(json_analysis, metadata, "report.csv")
        
        return (results_text, json_analysis_str, csv_records_str, all_client_records_str, report_status)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
        error_json = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        error_json_str = json.dumps(error_json, indent=2, ensure_ascii=False)
        return (error_msg, error_json_str, "", "", "")


def create_trade_verification_tab():
    """Create and return the Trade Verification tab"""
    with gr.Tab("🔍 Trade Verification"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📥 Input Data")
                
                transaction_json_box = gr.Textbox(
                    label="Transaction Analysis JSON",
                    placeholder='{\n  "transactions": [\n    {\n      "transaction_type": "buy",\n      "confidence_score": 0.95,\n      "conversation_number": 1,\n      "hkt_datetime": "2025-10-20T10:01:20",\n      "broker_id": "0489",\n      "broker_name": "Dickson Lau",\n      "client_id": "P77197",\n      "client_name": "CHENG SUK HING",\n      "stock_code": "18138",\n      "stock_name": "腾讯认购证",\n      "quantity": "20000",\n      "price": "0.38",\n      "explanation": "..."\n    }\n  ]\n}',
                    lines=20,
                    info="Paste the JSON output from Transaction Analysis tab (metadata is extracted from each transaction)"
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
                
                all_client_records_box = gr.Textbox(
                    label="All Client Records Found in trades.csv (JSON Format)",
                    lines=20,
                    interactive=False,
                    show_copy_button=True,
                    info="ALL records found for this client in trades.csv (regardless of time window or matching criteria)"
                )
                
                report_status_box = gr.Textbox(
                    label="📊 Report.csv Save Status",
                    lines=5,
                    interactive=False,
                    show_copy_button=False,
                    info="Status of saving verification results to report.csv"
                )
        
        # Connect the button
        verify_btn.click(
            fn=verify_transactions,
            inputs=[
                transaction_json_box,
                trades_file_box,
                time_window_slider,
            ],
            outputs=[
                results_box,
                json_analysis_box,
                csv_records_box,
                all_client_records_box,
                report_status_box,
            ],
        )

