from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import Dict
import json
import os


def analyze_conversation_for_stocks(conversation_text: str) -> dict:
    """
    Use LLM to analyze conversation text and identify stocks mentioned.
    
    Args:
        conversation_text: The conversation text to analyze
        
    Returns:
        dict: Dictionary with 'status' (success/error), 'stocks' (list of identified stocks),
              and 'analysis' (detailed analysis)
    """
    try:
        # Initialize LLM (same configuration as main agent)
        llm = ChatOpenAI(
            api_key="ollama",
            model="qwen3:8b",
            base_url="http://localhost:11434/v1",
            temperature=0.0
        )
        
        # Create analysis prompt
        analysis_prompt = f"""Analyze the following conversation and identify all stocks that are mentioned.
For each stock, provide:
1. Stock name or code (if mentioned)
2. Stock symbol (if mentioned)
3. Context (what was said about the stock - e.g., buy, sell, price discussion)
4. Any quantities or prices mentioned

Return your response in JSON format with this structure:
{{
    "stocks": [
        {{
            "name": "stock name",
            "symbol": "stock symbol or code",
            "context": "what was discussed",
            "quantity": "quantity mentioned (if any)",
            "price": "price mentioned (if any)"
        }}
    ],
    "summary": "brief summary of stock-related discussion"
}}

If no stocks are mentioned, return an empty stocks array.

Conversation:
{conversation_text}
"""
        
        # Call LLM
        response = llm.invoke(analysis_prompt)
        
        # Parse response
        response_text = response.content.strip()
        
        # Try to extract JSON from response (in case LLM adds extra text)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            analysis_result = json.loads(json_text)
        else:
            # If no JSON found, treat entire response as summary
            analysis_result = {
                "stocks": [],
                "summary": response_text
            }
        
        return {
            'status': 'success',
            'stocks': analysis_result.get('stocks', []),
            'summary': analysis_result.get('summary', ''),
            'raw_response': response_text
        }
        
    except json.JSONDecodeError as e:
        return {
            'status': 'error',
            'error': f'Failed to parse LLM response as JSON: {str(e)}',
            'raw_response': response_text if 'response_text' in locals() else None
        }
    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


@tool
def identify_stocks_in_conversation(conversation_text: str) -> str:
    """
    Analyze conversation text using LLM to identify stocks that were discussed.
    
    This tool takes a conversation transcript and uses an LLM to identify:
    - Stock names and symbols mentioned
    - Context (buy/sell/price discussion)
    - Quantities and prices mentioned
    
    Args:
        conversation_text: The full conversation text to analyze
        
    Returns:
        A formatted string with identified stocks and analysis, or an error message
    """
    result = analyze_conversation_for_stocks(conversation_text)
    
    if result['status'] == 'error':
        return f"Error analyzing conversation: {result.get('error', 'Unknown error')}"
    
    # Format output
    stocks = result.get('stocks', [])
    summary = result.get('summary', '')
    
    if not stocks:
        output = "No stocks identified in the conversation.\n\n"
        if summary:
            output += f"Summary: {summary}"
        return output
    
    output = f"Identified {len(stocks)} stock(s) in the conversation:\n\n"
    
    for i, stock in enumerate(stocks, 1):
        output += f"Stock {i}:\n"
        output += f"  Name: {stock.get('name', 'N/A')}\n"
        output += f"  Symbol/Code: {stock.get('symbol', 'N/A')}\n"
        output += f"  Context: {stock.get('context', 'N/A')}\n"
        if stock.get('quantity'):
            output += f"  Quantity: {stock.get('quantity')}\n"
        if stock.get('price'):
            output += f"  Price: {stock.get('price')}\n"
        output += "\n"
    
    if summary:
        output += f"Summary: {summary}\n"
    
    return output

