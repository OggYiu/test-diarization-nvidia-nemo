from langchain.tools import tool
from dataclasses import dataclass, asdict
from typing import Literal
import time
import dspy
import os
import json

@tool
def identify_stocks_in_conversation(file_path: str) -> str:
    """
    Analyze conversation text using dspy to identify stocks that were discussed.
    
    This tool takes a file path containing a conversation transcript and extracts structured stock information including:
    - Stock codes and names mentioned
    - Prices and quantities
    - Order types (ask/bid/none)
    - Confidence score (0-100) indicating how sure the model is about the extraction
    
    The identified stocks are saved to agent/output/stocks/ as a JSON file for later reference.
    
    Args:
        file_path: Path to the file containing the conversation text to analyze
        
    Returns:
        A formatted string with identified stocks and analysis, or an error message
    """
    # Read the conversation text from the file
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        
        # Try UTF-8 first, fall back to other encodings if needed
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5']
        conversation_text = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    conversation_text = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if conversation_text is None:
            return f"Error: Could not read file {file_path} with any supported encoding"
        
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"
    
    # dspy is configured in app.py to avoid threading issues
    # Just use the configured dspy instance here

    @dataclass
    class StockInfo():
        stock_name: str
        stock_number: str
        price: float
        quantity: int
        order_type: Literal["ask", "bid", "unknown"]
        confidence: int  # Confidence score from 0 to 100

    class ExtractInfo(dspy.Signature):
        """Extract structured information from text."""
        text: str = dspy.InputField()
        entities: list[StockInfo] = dspy.OutputField(desc="a list of stock name, stock number, price, quantity, order type (ask/bid/unknown), and confidence score (0-100 indicating how sure you are about the extraction)")

    try:
        module = dspy.Predict(ExtractInfo)
        response = module(text=conversation_text)
        
        # Format the output
        if not response.entities:
            return "No stocks identified in the conversation."
        
        # Convert dataclass objects to dictionaries for JSON serialization
        stocks_data = [asdict(stock) for stock in response.entities]
        
        # Save stocks to output/stocks/ directory
        try:
            # Determine the output directory - use agent/output/stocks/
            # current_dir is agent/tools/, so agent_dir is agent/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            agent_dir = os.path.dirname(current_dir)
            stocks_output_dir = os.path.join(agent_dir, "output", "stocks")
            
            # Create directory if it doesn't exist
            os.makedirs(stocks_output_dir, exist_ok=True)
            
            # Extract audio filename from the transcription file path
            # file_path is like: agent/output/transcriptions/[audio_filename]/transcriptions_text.txt
            # We want to get [audio_filename]
            file_path_normalized = os.path.normpath(file_path)
            path_parts = file_path_normalized.split(os.sep)
            
            # Find the transcription directory name (should be the audio filename)
            audio_filename = None
            if "transcriptions" in path_parts:
                trans_idx = path_parts.index("transcriptions")
                if trans_idx + 1 < len(path_parts):
                    audio_filename = path_parts[trans_idx + 1]
            
            # Fallback: use the parent directory name if we couldn't find it
            if not audio_filename:
                audio_filename = os.path.basename(os.path.dirname(file_path_normalized))
            
            # Create JSON filename based on audio filename
            json_filename = f"{audio_filename}.json"
            json_filepath = os.path.join(stocks_output_dir, json_filename)
            
            # Prepare data structure for saving
            stocks_output = {
                "source_file": file_path,
                "audio_filename": audio_filename,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "stocks": stocks_data
            }
            
            # Save stocks as JSON
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(stocks_output, f, indent=2, ensure_ascii=False)
            
            # Format the output string
            output = f"Identified {len(response.entities)} stock(s) in the conversation:\n\n"
            for i, stock in enumerate(response.entities, 1):
                output += f"Stock {i}:\n"
                output += f"  Name: {stock.stock_name}\n"
                output += f"  Code: {stock.stock_number}\n"
                output += f"  Price: {stock.price}\n"
                output += f"  Quantity: {stock.quantity}\n"
                output += f"  Order Type: {stock.order_type}\n"
                output += f"  Confidence: {stock.confidence:.1f}/100\n\n"
            
            # Add save confirmation to output
            output += f"💾 Stocks information saved to: {json_filepath}\n\n"
            
            # Add explicit instruction to continue to next step
            output += f"{'='*80}\n"
            output += "✅ Stock identification complete. Continue with the next step in the pipeline.\n"
            output += f"   Use stock_list_file: {json_filepath}\n"
            output += f"{'='*80}\n"
            
            return output
        except Exception as e:
            # If saving fails, log but don't fail the tool
            output = f"Identified {len(response.entities)} stock(s) in the conversation:\n\n"
            for i, stock in enumerate(response.entities, 1):
                output += f"Stock {i}:\n"
                output += f"  Name: {stock.stock_name}\n"
                output += f"  Code: {stock.stock_number}\n"
                output += f"  Price: {stock.price}\n"
                output += f"  Quantity: {stock.quantity}\n"
                output += f"  Order Type: {stock.order_type}\n"
                output += f"  Confidence: {stock.confidence:.1f}/100\n\n"
            
            output += f"\n⚠️  Warning: Could not save stocks to file: {str(e)}"
            return output
        
    except Exception as e:
        return f"Error analyzing conversation: {str(e)}"