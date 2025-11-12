"""
Test script for the stock identifier tool.
This demonstrates how to use the identify_stocks_in_conversation tool.
"""

from tools.stock_identifier_tool import identify_stocks_in_conversation

# Example conversation texts
example_conversations = [
    # Example 1: Simple stock discussion
    """
    Broker: Good morning, Mr. Chan. How can I help you today?
    Client: I want to buy 1000 shares of Tencent at HK$350.
    Broker: Okay, that's 1000 shares of Tencent Holdings, stock code 0700, at HK$350 per share.
    Client: Yes, please proceed.
    Broker: Done. The order has been placed.
    """,
    
    # Example 2: Multiple stocks
    """
    Broker: What stocks are you interested in?
    Client: I'm looking at HSBC and Bank of China. 
    Broker: HSBC is currently at HK$62, and Bank of China is at HK$3.50.
    Client: Let me buy 500 shares of HSBC and 2000 shares of Bank of China.
    Broker: Understood. 500 shares of HSBC (0005) and 2000 shares of BOC (3988).
    """,
    
    # Example 3: No stocks mentioned
    """
    Broker: Hello, how are you doing today?
    Client: I'm good, thanks. Just checking my account balance.
    Broker: Your current balance is HK$150,000.
    Client: Great, thank you!
    """
]

def test_stock_identifier():
    """Test the stock identifier tool with example conversations."""
    
    print("="*80)
    print("Testing Stock Identifier Tool")
    print("="*80)
    
    for i, conversation in enumerate(example_conversations, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}:")
        print(f"{'='*80}")
        print(f"\nConversation:")
        print(conversation.strip())
        print(f"\n{'-'*80}")
        print("Analysis Result:")
        print(f"{'-'*80}")
        
        # Call the tool (it's a StructuredTool, so use .invoke())
        result = identify_stocks_in_conversation.invoke({"conversation_text": conversation})
        print(result)
        print()

if __name__ == "__main__":
    test_stock_identifier()

