"""
Test script for Stock Verifier Tool
Demonstrates the verification and correction capabilities of the tool.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools.stock_verifier_tool import (
    initialize_stock_database,
    initialize_vector_store,
    verify_stock_with_excel,
    verify_stock_with_vector_store
)


def create_test_stocks_json(output_dir: str = "output/stocks"):
    """Create a test stocks JSON file with some intentional errors."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Test data with various scenarios
    test_stocks = {
        "source_file": "test/transcriptions/test_audio/transcriptions_text.txt",
        "audio_filename": "test_audio",
        "timestamp": "2025-11-17 14:30:00",
        "stocks": [
            {
                "stock_name": "騰訊控股",  # Correct name
                "stock_number": "0700",     # Correct code
                "price": 350.0,
                "quantity": 1000,
                "order_type": "buy"
            },
            {
                "stock_name": "匯豐銀行",  # Slight variation (should be 滙豐控股)
                "stock_number": "0005",     # Correct code
                "price": 55.0,
                "quantity": 500,
                "order_type": "sell"
            },
            {
                "stock_name": "中國移動",  # Common variation
                "stock_number": "0941",     # Correct code
                "price": 45.0,
                "quantity": 2000,
                "order_type": "buy"
            },
            {
                "stock_name": "阿里巴巴",  # Correct name
                "stock_number": "9988",     # Correct code
                "price": 85.0,
                "quantity": 1500,
                "order_type": "buy"
            },
            {
                "stock_name": "美圖公司",  # Test name
                "stock_number": "1357",     # May or may not be in DB
                "price": 2.5,
                "quantity": 10000,
                "order_type": "buy"
            }
        ]
    }
    
    output_path = os.path.join(output_dir, "test_audio.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_stocks, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created test stocks JSON: {output_path}")
    return output_path


def test_excel_verification():
    """Test Excel database verification."""
    print("\n" + "="*80)
    print("Testing Excel Database Verification")
    print("="*80 + "\n")
    
    # Initialize database
    df = initialize_stock_database()
    
    # Test cases
    test_cases = [
        {"name": "騰訊控股", "code": "0700"},  # Perfect match
        {"name": "匯豐銀行", "code": "0005"},  # Name variation
        {"name": "中國移動", "code": "0941"},  # Common name
        {"name": "Unknown", "code": "9999"},   # No match
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']} ({test['code']})")
        result = verify_stock_with_excel(test['name'], test['code'], df)
        
        if result['verified']:
            print(f"  ✅ Verified: {result['verified_name']} ({result['verified_code']})")
            print(f"  📈 Confidence: {result['confidence']:.2%}")
            print(f"  🔍 Match Type: {result['match_type']}")
            if result['corrected']:
                print(f"  🔧 Corrected from original")
        else:
            print(f"  ❌ Not verified")
            if result['candidates']:
                print(f"  📋 Candidates:")
                for j, candidate in enumerate(result['candidates'][:3], 1):
                    print(f"     {j}. {candidate['name']} ({candidate['code']}) - {candidate['similarity']:.2%}")
        print()


def test_vector_store_verification():
    """Test vector store verification (requires Ollama and Milvus)."""
    print("\n" + "="*80)
    print("Testing Vector Store Verification")
    print("="*80 + "\n")
    
    try:
        vs = initialize_vector_store()
        
        if not vs:
            print("⚠️  Vector store not available, skipping test")
            return
        
        # Test cases with STT-like errors
        test_cases = [
            {"name": "騰訊", "code": ""},           # Partial name
            {"name": "匯豐", "code": ""},           # Partial name
            {"name": "阿里", "code": ""},           # Partial name
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"Test {i}: {test['name']} ({test['code']})")
            result = verify_stock_with_vector_store(test['name'], test['code'], vs)
            
            if result['verified']:
                print(f"  ✅ Verified: {result['verified_name']} ({result['verified_code']})")
                print(f"  📈 Confidence: {result['confidence']:.2%}")
                print(f"  🔍 Match Type: {result['match_type']}")
            else:
                print(f"  ❌ Not verified")
            
            if result['candidates']:
                print(f"  📋 Candidates:")
                for j, candidate in enumerate(result['candidates'][:3], 1):
                    print(f"     {j}. {candidate['name']} ({candidate['code']}) - {candidate['similarity']:.2%}")
            print()
            
    except Exception as e:
        print(f"⚠️  Vector store test failed: {str(e)}")
        print("   (This is expected if Ollama or Milvus is not running)")


def test_full_verification():
    """Test the complete verification workflow using the tool."""
    print("\n" + "="*80)
    print("Testing Full Verification Workflow")
    print("="*80 + "\n")
    
    # Create test JSON
    test_json_path = create_test_stocks_json()
    
    # Import the tool function
    from agent.tools.stock_verifier_tool import verify_stocks
    
    # Run verification
    result = verify_stocks.func(test_json_path)  # Use .func to call the actual function
    
    print("\n" + "="*80)
    print("Verification Result:")
    print("="*80)
    print(result)
    
    # Check if verified file was created
    verified_path = test_json_path.replace('.json', '_verified.json')
    if os.path.exists(verified_path):
        print(f"\n✅ Verified JSON file created: {verified_path}")
        
        # Load and display the verified data
        with open(verified_path, 'r', encoding='utf-8') as f:
            verified_data = json.load(f)
        
        print(f"\n📊 Verified Stocks Summary:")
        for i, stock in enumerate(verified_data['stocks'], 1):
            verification = stock.get('verification', {})
            print(f"\nStock {i}:")
            print(f"  Original: {stock['stock_name']} ({stock['stock_number']})")
            if verification.get('verified'):
                print(f"  Verified: {verification['verified_name']} ({verification['verified_code']})")
                print(f"  Confidence: {verification['confidence']:.2%}")
                print(f"  Corrected: {'Yes' if verification['corrected'] else 'No'}")
            else:
                print(f"  Status: Unverified")
    else:
        print(f"\n❌ Verified JSON file not created")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Stock Verifier Tool Test Suite")
    print("="*80)
    
    try:
        # Test 1: Excel verification
        test_excel_verification()
        
        # Test 2: Vector store verification (optional)
        test_vector_store_verification()
        
        # Test 3: Full workflow
        test_full_verification()
        
        print("\n" + "="*80)
        print("✅ All Tests Complete")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

