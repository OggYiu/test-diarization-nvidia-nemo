# from langchain.tools import tool
# from langchain_openai import ChatOpenAI
# from typing import Dict
# import json
# import os


# def analyze_conversation_for_stocks(conversation_text: str) -> dict:
#     """
#     Use LLM to analyze conversation text and identify stocks mentioned.
    
#     Args:
#         conversation_text: The conversation text to analyze
        
#     Returns:
#         dict: Dictionary with 'status' (success/error), 'stocks' (list of identified stocks),
#               and 'analysis' (detailed analysis)
#     """
#     try:
#         # Initialize LLM (same configuration as main agent)
#         llm = ChatOpenAI(
#             api_key="ollama",
#             model="qwen3:32b",
#             base_url="http://localhost:11434/v1",
#             temperature=1.0
#         )
        
#         # Create analysis prompt
#         analysis_prompt = f"""Analyze the following conversation and identify all stocks that are mentioned.
# For each stock, provide:
# 1. Stock code
# 2. Stock name
# 3. Action: MUST be one of "buy", "sell", "hold", "discuss", or "unknown" based on the conversation context
#    - "buy": ONLY if the conversation indicates buying/purchasing the stock AND BOTH price and quantity are explicitly mentioned
#    - "sell": ONLY if the conversation indicates selling/disposing the stock AND BOTH price and quantity are explicitly mentioned
#    - "hold": if the conversation indicates holding/keeping the stock
#    - "discuss": if the stock is just discussed without clear buy/sell intent, or if buy/sell is mentioned but price/quantity is missing
#    - "unknown": if the action cannot be determined
   
#    IMPORTANT: If you identify a potential "buy" or "sell" action but cannot find BOTH a specific price AND a specific quantity in the conversation, you MUST classify it as "discuss" instead. It is very unlikely to be an actual buy/sell transaction without both price and quantity information.

# 4. Context (what was said about the stock - e.g., buy, sell, price discussion)
# 5. Any quantities or prices mentioned (extract these explicitly if present)
# 6. Confidence score (0.0 to 1.0)
# 7. Reasoning (why you think the stock was mentioned and what action was indicated, including whether price/quantity were found)

# Return your response in JSON format with this structure:
# {{
#     "stocks": [
#         {{
#             "stock_code": "stock code",
#             "stock_name": "stock name",
#             "action": "buy|sell|hold|discuss|unknown",
#             "context": "what was discussed",
#             "quantity": "quantity mentioned (if any)",
#             "price": "price mentioned (if any)",
#             "confidence": "confidence score (0.0 to 1.0)",
#             "reasoning": "why you think the stock was mentioned and what action was indicated"
#         }}
#     ],
#     "summary": "brief summary of stock-related discussion"
# }}

# If no stocks are mentioned, return an empty stocks array. PLEASE REPLY IN TRADITIONAL CHINESE.

# Conversation:
# {conversation_text}
# """
        
#         # Call LLM
#         response = llm.invoke(analysis_prompt)
        
#         # Parse response
#         response_text = response.content.strip()
        
#         # Try to extract JSON from response (in case LLM adds extra text)
#         json_start = response_text.find('{')
#         json_end = response_text.rfind('}') + 1
        
#         if json_start >= 0 and json_end > json_start:
#             json_text = response_text[json_start:json_end]
#             analysis_result = json.loads(json_text)
#         else:
#             # If no JSON found, treat entire response as summary
#             analysis_result = {
#                 "stocks": [],
#                 "summary": response_text
#             }
        
#         # Post-process: Validate that buy/sell actions have both price and quantity
#         stocks = analysis_result.get('stocks', [])
#         for stock in stocks:
#             action = stock.get('action', '').lower()
#             if action in ['buy', 'sell']:
#                 price = stock.get('price', '').strip()
#                 quantity = stock.get('quantity', '').strip()
                
#                 # If either price or quantity is missing, downgrade to "discuss"
#                 if not price or not quantity:
#                     stock['action'] = 'discuss'
#                     # Update reasoning to explain why it was downgraded
#                     original_reasoning = stock.get('reasoning', '')
#                     stock['reasoning'] = f"{original_reasoning} [Downgraded from {action} to discuss: missing {'price' if not price else 'quantity'}]"
        
#         return {
#             'status': 'success',
#             'stocks': stocks,
#             'summary': analysis_result.get('summary', ''),
#             'raw_response': response_text
#         }
        
#     except json.JSONDecodeError as e:
#         return {
#             'status': 'error',
#             'error': f'Failed to parse LLM response as JSON: {str(e)}',
#             'raw_response': response_text if 'response_text' in locals() else None
#         }
#     except Exception as e:
#         import traceback
#         return {
#             'status': 'error',
#             'error': str(e),
#             'traceback': traceback.format_exc()
#         }


# @tool
# def identify_stocks_in_conversation(conversation_text: str) -> str:
#     """
#     Analyze conversation text using LLM to identify stocks that were discussed.
    
#     This tool takes a conversation transcript and uses an LLM to identify:
#     - Stock codes and names mentioned
#     - Action (buy/sell/hold/discuss/unknown)
#     - Context (buy/sell/price discussion)
#     - Quantities and prices mentioned
    
#     Args:
#         conversation_text: The full conversation text to analyze
        
#     Returns:
#         A formatted string with identified stocks and analysis, or an error message
#     """
#     result = analyze_conversation_for_stocks(conversation_text)
    
#     if result['status'] == 'error':
#         return f"Error analyzing conversation: {result.get('error', 'Unknown error')}"
    
#     # Format output
#     stocks = result.get('stocks', [])
#     summary = result.get('summary', '')
    
#     if not stocks:
#         output = "No stocks identified in the conversation.\n\n"
#         if summary:
#             output += f"Summary: {summary}"
#         return output
    
#     output = f"Identified {len(stocks)} stock(s) in the conversation:\n\n"
    
#     for i, stock in enumerate(stocks, 1):
#         output += f"Stock {i}:\n"
#         # Try both naming conventions (stock_name/stock_code from LLM, or name/symbol for compatibility)
#         output += f"  Name: {stock.get('stock_name', 'N/A')}\n"
#         output += f"  Code: {stock.get('stock_code', 'N/A')}\n"
#         # Prominently display the action
#         action = stock.get('action', 'unknown')
#         action_display = action.upper() if action in ['buy', 'sell'] else action
#         output += f"  Action: {action_display}\n"
#         output += f"  Context: {stock.get('context', 'N/A')}\n"
#         if stock.get('quantity'):
#             output += f"  Quantity: {stock.get('quantity')}\n"
#         if stock.get('price'):
#             output += f"  Price: {stock.get('price')}\n"
#         output += "\n"
    
#     if summary:
#         output += f"Summary: {summary}\n"
    
#     return output

from langchain.tools import tool
from dataclasses import dataclass
import time
import dspy

@tool
def identify_stocks_in_conversation(conversation_text: str) -> str:
    """
    Analyze conversation text using dspy to identify stocks that were discussed.
    
    This tool takes a conversation transcript and extracts structured stock information including:
    - Stock codes and names mentioned
    - Prices and quantities
    - Order types (ask/bid/none)
    
    Args:
        conversation_text: The full conversation text to analyze
        
    Returns:
        A formatted string with identified stocks and analysis, or an error message
    """
    lm = dspy.LM("ollama_chat/qwen3:32b", api_base="http://localhost:11434", api_key="")

    dspy.configure(lm=lm)

    @dataclass
    class StockInfo():
        stock_name: str
        stock_number: str
        price: float
        quantity: int
        order_type: str

    class ExtractInfo(dspy.Signature):
        """Extract structured information from text."""
        text: str = dspy.InputField()
        entities: list[StockInfo] = dspy.OutputField(desc="a list of stock name, stock number, price, quantity, order type (ask/bid/none)")

    module = dspy.Predict(ExtractInfo)
    response = module(text=conversation_text)
    print(response.entities)