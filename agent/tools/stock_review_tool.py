"""
Stock Review Tool for Agent
Reviews and confirms stocks and transactions from a stock list file against conversation transcriptions using dspy.
"""

import os
import json
import time
import csv
from typing import Annotated, Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from langchain.tools import tool
import dspy

# Import path normalization utilities
from .path_utils import normalize_path_for_llm, normalize_path_from_llm


def load_conversation_transcription(file_path: str) -> Optional[str]:
    """
    Load conversation transcription from file.
    
    Args:
        file_path: Path to transcription file
        
    Returns:
        Transcription text, or None if error
    """
    try:
        file_path = normalize_path_from_llm(file_path)
        
        if not os.path.exists(file_path):
            return None
        
        # Try multiple encodings
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        return None
    except Exception as e:
        print(f"⚠️  Error loading transcription file: {str(e)}")
        return None


def load_stock_list(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load stock list from JSON or CSV file.
    
    Args:
        file_path: Path to stock list file (JSON or CSV)
        
    Returns:
        List of stock/transaction dictionaries, or None if error
    """
    try:
        file_path = normalize_path_from_llm(file_path)
        
        if not os.path.exists(file_path):
            return None
        
        # Determine file type
        if file_path.lower().endswith('.json'):
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Check for common keys
                if 'transactions' in data:
                    return data['transactions']
                elif 'stocks' in data:
                    return data['stocks']
                else:
                    # Return as single-item list
                    return [data]
            else:
                return None
                
        elif file_path.lower().endswith('.csv'):
            # Load CSV file
            transactions = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    transactions.append(dict(row))
            return transactions
        else:
            # Try to load as JSON anyway
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    if 'transactions' in data:
                        return data['transactions']
                    elif 'stocks' in data:
                        return data['stocks']
                    else:
                        return [data]
                else:
                    return None
                    
    except Exception as e:
        print(f"⚠️  Error loading stock list file: {str(e)}")
        return None


@dataclass
class TransactionConfirmation:
    """Structured confirmation result for a stock transaction."""
    confirmed: bool
    confidence_score: int  # 0-100
    explanation: str
    verified_stock_code: str
    verified_stock_name: str
    verified_quantity: str
    verified_price: str
    verified_transaction_type: str


def confirm_transaction_with_dspy(
    conversation_text: str,
    transaction: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use dspy to confirm a transaction against the conversation and assign confidence score.
    
    Args:
        conversation_text: Full conversation transcription
        transaction: Transaction dictionary to verify
        
    Returns:
        Dictionary with confirmation results including confidence score (0-100)
    """
    # Extract transaction details
    stock_code = transaction.get('stock_code') or transaction.get('stock_number') or transaction.get('code', '')
    stock_name = transaction.get('stock_name') or transaction.get('name', '')
    quantity = transaction.get('quantity', '')
    price = transaction.get('price', '')
    transaction_type = transaction.get('transaction_type') or transaction.get('order_type') or transaction.get('type', '')
    
    # Use full conversation text, but limit to reasonable length to avoid token limits
    conversation_snippet = conversation_text
    if len(conversation_text) > 10000:
        # For very long conversations, include beginning and end
        conversation_snippet = conversation_text[:5000] + "\n\n[...中間部分省略...]\n\n" + conversation_text[-5000:]
    
    # Define dspy signature for transaction verification
    class VerifyTransaction(dspy.Signature):
        """
        你是一位專業的股票交易分析師。根據對話內容，確認股票交易是否真實發生，並給出置信度評分（0-100分）。
        
        評分標準：
        - 90-100分：交易明確確認，券商重複確認了所有細節（股票代號、數量、價格），客戶明確同意
        - 70-89分：交易基本確認，有明確的交易意圖和確認，但可能缺少部分細節確認
        - 50-69分：交易意圖明確，但確認流程不完整，或缺少關鍵細節
        - 30-49分：僅有交易討論，但未明確下單或確認
        - 0-29分：僅提到股票，但無交易意圖或確認
        """
        conversation: str = dspy.InputField(desc="對話內容")
        stock_code: str = dspy.InputField(desc="待確認的股票代號")
        stock_name: str = dspy.InputField(desc="待確認的股票名稱")
        quantity: str = dspy.InputField(desc="待確認的數量")
        price: str = dspy.InputField(desc="待確認的價格")
        transaction_type: str = dspy.InputField(desc="待確認的交易類型")
        
        result: TransactionConfirmation = dspy.OutputField(
            desc="包含確認狀態、置信度評分(0-100)、詳細解釋，以及確認後的股票代號、名稱、數量、價格和交易類型"
        )
    
    try:
        # Use dspy to verify transaction
        module = dspy.Predict(VerifyTransaction)
        response = module(
            conversation=conversation_snippet,
            stock_code=stock_code,
            stock_name=stock_name,
            quantity=str(quantity),
            price=str(price),
            transaction_type=transaction_type
        )
        
        # Convert dataclass to dictionary
        confirmation = response.result
        if isinstance(confirmation, TransactionConfirmation):
            result = asdict(confirmation)
        else:
            # If dspy returns a dict-like object, convert it
            result = {
                "confirmed": bool(getattr(confirmation, 'confirmed', False)),
                "confidence_score": int(getattr(confirmation, 'confidence_score', 50)),
                "explanation": str(getattr(confirmation, 'explanation', 'No explanation provided')),
                "verified_stock_code": str(getattr(confirmation, 'verified_stock_code', stock_code)),
                "verified_stock_name": str(getattr(confirmation, 'verified_stock_name', stock_name)),
                "verified_quantity": str(getattr(confirmation, 'verified_quantity', quantity)),
                "verified_price": str(getattr(confirmation, 'verified_price', price)),
                "verified_transaction_type": str(getattr(confirmation, 'verified_transaction_type', transaction_type))
            }
        
        # Ensure confidence_score is between 0 and 100
        confidence = int(result.get('confidence_score', 50))
        confidence = max(0, min(100, confidence))
        result['confidence_score'] = confidence
        
        return result
        
    except Exception as e:
        print(f"⚠️  Error using dspy: {str(e)}")
        # Return default result on error
        return {
            "confirmed": False,
            "confidence_score": 0,
            "explanation": f"dspy處理錯誤: {str(e)}",
            "verified_stock_code": stock_code,
            "verified_stock_name": stock_name,
            "verified_quantity": quantity,
            "verified_price": price,
            "verified_transaction_type": transaction_type
        }


def generate_review_report(
    conversation_file: str,
    stock_list_file: str,
    transactions: List[Dict[str, Any]],
    confirmations: List[Dict[str, Any]]
) -> str:
    """
    Generate a comprehensive review report.
    
    Args:
        conversation_file: Path to conversation transcription file
        stock_list_file: Path to stock list file
        transactions: List of original transactions
        confirmations: List of LLM confirmation results
        
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("STOCK TRANSACTION REVIEW REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Conversation File: {conversation_file}")
    report_lines.append(f"Stock List File: {stock_list_file}")
    report_lines.append(f"Review Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Statistics
    total_transactions = len(transactions)
    confirmed_count = sum(1 for c in confirmations if c.get('confirmed', False))
    high_confidence_count = sum(1 for c in confirmations if c.get('confidence_score', 0) >= 85)
    medium_confidence_count = sum(1 for c in confirmations if 50 <= c.get('confidence_score', 0) < 85)
    low_confidence_count = sum(1 for c in confirmations if c.get('confidence_score', 0) < 50)
    
    avg_confidence = sum(c.get('confidence_score', 0) for c in confirmations) / len(confirmations) if confirmations else 0
    
    # Executive Summary
    report_lines.append("=" * 80)
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Total Transactions Reviewed: {total_transactions}")
    report_lines.append(f"Confirmed Transactions: {confirmed_count} ({confirmed_count/total_transactions*100:.1f}%)" if total_transactions > 0 else "Confirmed Transactions: 0")
    report_lines.append(f"Average Confidence Score: {avg_confidence:.1f}/100")
    report_lines.append("")
    
    report_lines.append("Confidence Distribution:")
    report_lines.append(f"  🟢 High Confidence (≥85): {high_confidence_count}")
    report_lines.append(f"  🟡 Medium Confidence (50-84): {medium_confidence_count}")
    report_lines.append(f"  🔴 Low Confidence (<50): {low_confidence_count}")
    report_lines.append("")
    
    # Transaction Details
    report_lines.append("=" * 80)
    report_lines.append("TRANSACTION DETAILS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for i, (transaction, confirmation) in enumerate(zip(transactions, confirmations), 1):
        report_lines.append(f"Transaction {i}")
        report_lines.append("-" * 80)
        
        # Original transaction info
        stock_code = transaction.get('stock_code') or transaction.get('stock_number') or transaction.get('code', 'N/A')
        stock_name = transaction.get('stock_name') or transaction.get('name', 'N/A')
        quantity = transaction.get('quantity', 'N/A')
        price = transaction.get('price', 'N/A')
        transaction_type = transaction.get('transaction_type') or transaction.get('order_type') or transaction.get('type', 'N/A')
        
        report_lines.append(f"Original Information:")
        report_lines.append(f"  Stock Code: {stock_code}")
        report_lines.append(f"  Stock Name: {stock_name}")
        report_lines.append(f"  Quantity: {quantity}")
        report_lines.append(f"  Price: {price}")
        report_lines.append(f"  Transaction Type: {transaction_type}")
        report_lines.append("")
        
        # Confirmation results
        confirmed = confirmation.get('confirmed', False)
        confidence = confirmation.get('confidence_score', 0)
        explanation = confirmation.get('explanation', 'No explanation provided')
        
        report_lines.append(f"Confirmation Results:")
        report_lines.append(f"  Status: {'✅ CONFIRMED' if confirmed else '❌ NOT CONFIRMED'}")
        report_lines.append(f"  Confidence Score: {confidence}/100")
        report_lines.append(f"  Explanation: {explanation}")
        report_lines.append("")
        
        # Verified information (if different from original)
        verified_code = confirmation.get('verified_stock_code', stock_code)
        verified_name = confirmation.get('verified_stock_name', stock_name)
        verified_quantity = confirmation.get('verified_quantity', quantity)
        verified_price = confirmation.get('verified_price', price)
        verified_type = confirmation.get('verified_transaction_type', transaction_type)
        
        if (verified_code != stock_code or verified_name != stock_name or 
            verified_quantity != quantity or verified_price != price or verified_type != transaction_type):
            report_lines.append(f"Verified Information (Corrected):")
            report_lines.append(f"  Stock Code: {verified_code}")
            report_lines.append(f"  Stock Name: {verified_name}")
            report_lines.append(f"  Quantity: {verified_quantity}")
            report_lines.append(f"  Price: {verified_price}")
            report_lines.append(f"  Transaction Type: {verified_type}")
            report_lines.append("")
        
        report_lines.append("")
    
    # Recommendations
    report_lines.append("=" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    if low_confidence_count > 0:
        report_lines.append(f"⚠️  {low_confidence_count} transaction(s) have low confidence scores. Please review manually.")
        report_lines.append("")
    
    if confirmed_count < total_transactions:
        unconfirmed = total_transactions - confirmed_count
        report_lines.append(f"⚠️  {unconfirmed} transaction(s) were not confirmed. Please verify these transactions.")
        report_lines.append("")
    
    if confirmed_count == total_transactions and high_confidence_count == total_transactions:
        report_lines.append("✅ All transactions confirmed with high confidence. No action required.")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


@tool
def generate_transaction_report(
    conversation_transcription_file: Annotated[str, "Path to the conversation transcription file"],
    stock_list_file: Annotated[str, "Path to the stock list file (JSON or CSV)"]
) -> str:
    """
    Generate a comprehensive transaction report by analyzing the conversation and verified stocks.
    
    This tool reviews and confirms stocks and transactions from a stock list file against conversation transcriptions using dspy.
    
    This tool uses dspy with structured output to verify each transaction in the stock list against the conversation transcription,
    assigns a confidence score (0-100) to each transaction, and generates a comprehensive report.
    
    Confidence Score Scale:
    - 90-100: Transaction clearly confirmed with all details verified by broker and client agreement
    - 70-89: Transaction basically confirmed with clear intent and confirmation, may lack some details
    - 50-69: Clear transaction intent but incomplete confirmation process or missing key details
    - 30-49: Only discussion about transaction, no clear order placement or confirmation
    - 0-29: Stock mentioned only, no transaction intent or confirmation
    
    The report is saved to agent/output/reports/ directory.
    
    Args:
        conversation_transcription_file: Path to the file containing conversation transcription
        stock_list_file: Path to the stock list file (JSON or CSV format)
        
    Returns:
        str: Summary message with report location and key statistics
    """
    try:
        # Normalize paths
        conversation_file = normalize_path_from_llm(conversation_transcription_file)
        stock_list_file_path = normalize_path_from_llm(stock_list_file)
        
        # Load conversation transcription
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Loading conversation transcription...")
        conversation_text = load_conversation_transcription(conversation_file)
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Transcription loaded in {trace_elapsed:.4f}s")
        
        if not conversation_text:
            return f"❌ Error: Could not load conversation transcription from: {conversation_file}"
        
        # Load stock list
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Loading stock list...")
        transactions = load_stock_list(stock_list_file_path)
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Stock list loaded in {trace_elapsed:.4f}s")
        
        if not transactions:
            return f"❌ Error: Could not load stock list from: {stock_list_file_path}"
        
        if len(transactions) == 0:
            return "❌ No transactions found in the stock list file."
        
        # Note: dspy is configured in app.py to avoid threading issues
        # Just use the configured dspy instance here
        
        # Confirm each transaction with dspy
        confirmations = []
        total_transactions = len(transactions)
        
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Confirming {total_transactions} transaction(s) with dspy...")
        
        for i, transaction in enumerate(transactions, 1):
            print(f"[TRACE {time.strftime('%H:%M:%S')}] Processing transaction {i}/{total_transactions}...")
            trace_start = time.time()
            
            confirmation = confirm_transaction_with_dspy(conversation_text, transaction)
            confirmations.append(confirmation)
            
            trace_elapsed = time.time() - trace_start
            print(f"[TRACE {time.strftime('%H:%M:%S')}] Transaction {i} confirmed in {trace_elapsed:.4f}s (Confidence: {confirmation.get('confidence_score', 0)}/100)")
        
        # Generate report
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Generating report...")
        report_content = generate_review_report(
            conversation_file,
            stock_list_file_path,
            transactions,
            confirmations
        )
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Report generation completed in {trace_elapsed:.4f}s")
        
        # Save report
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Saving report to file...")
        
        # Determine output directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        agent_dir = os.path.dirname(current_dir)
        reports_dir = os.path.join(agent_dir, 'output', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate report filename
        base_name = os.path.splitext(os.path.basename(stock_list_file_path))[0]
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_filename = f"{base_name}_review_{timestamp}.txt"
        report_path = os.path.join(reports_dir, report_filename)
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Also save JSON results
        json_filename = f"{base_name}_review_{timestamp}.json"
        json_path = os.path.join(reports_dir, json_filename)
        
        results_data = {
            "conversation_file": conversation_file,
            "stock_list_file": stock_list_file_path,
            "review_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "transactions": transactions,
            "confirmations": confirmations
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] File save completed in {trace_elapsed:.4f}s")
        
        # Normalize output path for LLM
        report_path_for_llm = normalize_path_for_llm(report_path)
        json_path_for_llm = normalize_path_for_llm(json_path)
        
        # Calculate summary statistics
        confirmed_count = sum(1 for c in confirmations if c.get('confirmed', False))
        avg_confidence = sum(c.get('confidence_score', 0) for c in confirmations) / len(confirmations) if confirmations else 0
        high_confidence_count = sum(1 for c in confirmations if c.get('confidence_score', 0) >= 85)
        
        summary = f"\n{'='*80}\n"
        summary += f"✅ Stock Transaction Review Complete\n"
        summary += f"{'='*80}\n\n"
        summary += f"📊 Total Transactions: {total_transactions}\n"
        summary += f"✅ Confirmed: {confirmed_count}\n"
        summary += f"📈 Average Confidence: {avg_confidence:.1f}/100\n"
        summary += f"🟢 High Confidence (≥85): {high_confidence_count}\n"
        summary += f"💾 Report saved to: {report_path_for_llm}\n"
        summary += f"💾 JSON results saved to: {json_path_for_llm}\n"
        summary += f"{'='*80}\n\n"
        
        # Mark completion - this is the final step
        summary += f"{'='*80}\n"
        summary += "🎉 TRANSACTION REPORT GENERATED - PIPELINE COMPLETE!\n"
        summary += "{'='*80}\n"
        summary += "All transaction analysis is complete. The comprehensive report has been generated.\n"
        summary += f"{'='*80}\n"
        
        return summary
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error during stock review: {str(e)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        return error_msg
