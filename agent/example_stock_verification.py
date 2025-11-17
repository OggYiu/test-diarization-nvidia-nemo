"""
Simple example demonstrating stock verification tool usage.
This shows the minimal code needed to verify stocks.
"""

import os
import json

# Example 1: Create a sample stocks JSON file
def create_sample_stock_file():
    """Create a sample stocks JSON file for demonstration."""
    
    sample_data = {
        "source_file": "example_conversation.txt",
        "audio_filename": "example_audio",
        "timestamp": "2025-11-17 15:00:00",
        "stocks": [
            {
                "stock_name": "騰訊控股",
                "stock_number": "0700",
                "price": 350.0,
                "quantity": 1000,
                "order_type": "buy"
            },
            {
                "stock_name": "匯豐銀行",  # This will be corrected to 滙豐控股
                "stock_number": "0005",
                "price": 55.0,
                "quantity": 500,
                "order_type": "sell"
            }
        ]
    }
    
    # Save to output/stocks directory
    output_dir = "output/stocks"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "example_audio.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created sample stocks file: {output_file}")
    return output_file


# Example 2: Verify stocks using the tool
def verify_sample_stocks():
    """Verify the sample stocks using the stock verifier tool."""
    
    # Import the tool
    from tools.stock_verifier_tool import verify_stocks
    
    # Create sample file
    stocks_file = create_sample_stock_file()
    
    # Verify stocks
    print("\n" + "="*80)
    print("Verifying stocks...")
    print("="*80 + "\n")
    
    # Call the tool (use .func to call the actual function)
    result = verify_stocks.func(stocks_file)
    
    # Print result
    print(result)
    
    # Check the verified file
    verified_file = stocks_file.replace('.json', '_verified.json')
    if os.path.exists(verified_file):
        with open(verified_file, 'r', encoding='utf-8') as f:
            verified_data = json.load(f)
        
        print("\n" + "="*80)
        print("Verified Data:")
        print("="*80)
        print(json.dumps(verified_data, indent=2, ensure_ascii=False))


# Example 3: Direct function usage (without LangChain tool wrapper)
def direct_verification_example():
    """Example of using verification functions directly."""
    
    from tools.stock_verifier_tool import (
        initialize_stock_database,
        verify_stock_with_excel
    )
    
    print("\n" + "="*80)
    print("Direct Verification Example")
    print("="*80 + "\n")
    
    # Initialize database
    df = initialize_stock_database()
    
    # Verify a single stock
    stock_name = "騰訊控股"
    stock_code = "0700"
    
    print(f"Verifying: {stock_name} ({stock_code})")
    result = verify_stock_with_excel(stock_name, stock_code, df)
    
    print(f"\nResult:")
    print(f"  Verified: {result['verified']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    if result['verified']:
        print(f"  Verified Name: {result['verified_name']}")
        print(f"  Verified Code: {result['verified_code']}")
        print(f"  Match Type: {result['match_type']}")
        print(f"  Corrected: {result['corrected']}")


if __name__ == "__main__":
    print("="*80)
    print("Stock Verification Tool - Simple Examples")
    print("="*80)
    
    # Run examples
    try:
        # Example 1 & 2: Create and verify sample file
        verify_sample_stocks()
        
        # Example 3: Direct function usage
        direct_verification_example()
        
        print("\n" + "="*80)
        print("✅ Examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

