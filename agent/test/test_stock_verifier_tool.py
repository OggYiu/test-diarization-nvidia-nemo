"""
Test script for stock_verifier_tool.py
Tests various scenarios including exact matches, fuzzy matches, corrections, and edge cases.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add project root to path to import the tool
# From agent/test/ we need to go up to project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.tools.stock_verifier_tool import verify_stocks


def create_test_stocks_file(test_name: str, stocks_data: list, output_dir: str = None) -> str:
    """Create a test stocks JSON file."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "test_stocks")
    
    os.makedirs(output_dir, exist_ok=True)
    
    test_data = {
        "source_file": f"test_transcription_{test_name}.txt",
        "audio_filename": f"test_audio_{test_name}",
        "timestamp": "2025-01-20 10:00:00",
        "stocks": stocks_data
    }
    
    output_file = os.path.join(output_dir, f"{test_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    return output_file


def test_exact_match():
    """Test 1: Exact code and name match - should verify without correction."""
    print("\n" + "="*80)
    print("TEST 1: Exact Match (Code and Name)")
    print("="*80)
    
    stocks = [
        {
            "stock_name": "騰訊控股",
            "stock_number": "00700",
            "price": 350.0,
            "quantity": 1000,
            "order_type": "bid"
        }
    ]
    
    test_file = create_test_stocks_file("test_exact_match", stocks)
    print(f"Created test file: {test_file}")
    
    result = verify_stocks.func(test_file)
    print("\nResult:")
    print(result)
    
    # Check verified file
    verified_file = test_file.replace('.json', '_verified.json')
    if os.path.exists(verified_file):
        with open(verified_file, 'r', encoding='utf-8') as f:
            verified_data = json.load(f)
        print("\nVerified JSON:")
        print(json.dumps(verified_data, indent=2, ensure_ascii=False))
    
    return result


def test_fuzzy_name_match():
    """Test 2: Fuzzy name match - STT error in name, should correct."""
    print("\n" + "="*80)
    print("TEST 2: Fuzzy Name Match (STT Error)")
    print("="*80)
    
    stocks = [
        {
            "stock_name": "騰訊控",  # Missing character (STT error)
            "stock_number": "00700",
            "price": 350.0,
            "quantity": 1000,
            "order_type": "bid"
        }
    ]
    
    test_file = create_test_stocks_file("test_fuzzy_name", stocks)
    print(f"Created test file: {test_file}")
    
    result = verify_stocks.func(test_file)
    print("\nResult:")
    print(result)
    
    return result


def test_code_only_match():
    """Test 3: Code only match - missing or incorrect name."""
    print("\n" + "="*80)
    print("TEST 3: Code Only Match")
    print("="*80)
    
    stocks = [
        {
            "stock_name": "",  # Missing name
            "stock_number": "00005",  # HSBC
            "price": 55.0,
            "quantity": 500,
            "order_type": "ask"
        }
    ]
    
    test_file = create_test_stocks_file("test_code_only", stocks)
    print(f"Created test file: {test_file}")
    
    result = verify_stocks.func(test_file)
    print("\nResult:")
    print(result)
    
    return result


def test_name_only_match():
    """Test 4: Name only match - missing or incorrect code."""
    print("\n" + "="*80)
    print("TEST 4: Name Only Match")
    print("="*80)
    
    stocks = [
        {
            "stock_name": "中國移動",
            "stock_number": "",  # Missing code
            "price": 50.0,
            "quantity": 2000,
            "order_type": "bid"
        }
    ]
    
    test_file = create_test_stocks_file("test_name_only", stocks)
    print(f"Created test file: {test_file}")
    
    result = verify_stocks.func(test_file)
    print("\nResult:")
    print(result)
    
    return result


def test_multiple_stocks():
    """Test 5: Multiple stocks with various match types."""
    print("\n" + "="*80)
    print("TEST 5: Multiple Stocks (Mixed Scenarios)")
    print("="*80)
    
    stocks = [
        {
            "stock_name": "騰訊控股",
            "stock_number": "00700",
            "price": 350.0,
            "quantity": 1000,
            "order_type": "bid"
        },
        {
            "stock_name": "快手",  # Should match
            "stock_number": "01680",
            "price": 89.5,
            "quantity": 400,
            "order_type": "bid"
        },
        {
            "stock_name": "阿里巴巴",  # Might not match exactly
            "stock_number": "09988",
            "price": 100.0,
            "quantity": 500,
            "order_type": "ask"
        },
        {
            "stock_name": "未知股票",  # Unlikely to match
            "stock_number": "99999",
            "price": 10.0,
            "quantity": 100,
            "order_type": "bid"
        }
    ]
    
    test_file = create_test_stocks_file("test_multiple", stocks)
    print(f"Created test file: {test_file}")
    
    result = verify_stocks.func(test_file)
    print("\nResult:")
    print(result)
    
    return result


def test_fuzzy_code_match():
    """Test 6: Fuzzy code match - STT error in code."""
    print("\n" + "="*80)
    print("TEST 6: Fuzzy Code Match (STT Error)")
    print("="*80)
    
    stocks = [
        {
            "stock_name": "騰訊控股",
            "stock_number": "0070",  # Missing last digit
            "price": 350.0,
            "quantity": 1000,
            "order_type": "bid"
        }
    ]
    
    test_file = create_test_stocks_file("test_fuzzy_code", stocks)
    print(f"Created test file: {test_file}")
    
    result = verify_stocks.func(test_file)
    print("\nResult:")
    print(result)
    
    return result


def test_unverified_stock():
    """Test 7: Stock that cannot be verified."""
    print("\n" + "="*80)
    print("TEST 7: Unverified Stock")
    print("="*80)
    
    stocks = [
        {
            "stock_name": "不存在的股票",
            "stock_number": "99999",
            "price": 1.0,
            "quantity": 1,
            "order_type": "bid"
        }
    ]
    
    test_file = create_test_stocks_file("test_unverified", stocks)
    print(f"Created test file: {test_file}")
    
    result = verify_stocks.func(test_file)
    print("\nResult:")
    print(result)
    
    return result


def test_empty_stocks():
    """Test 8: Empty stocks array."""
    print("\n" + "="*80)
    print("TEST 8: Empty Stocks Array")
    print("="*80)
    
    stocks = []
    
    test_file = create_test_stocks_file("test_empty", stocks)
    print(f"Created test file: {test_file}")
    
    result = verify_stocks.func(test_file)
    print("\nResult:")
    print(result)
    
    return result


def test_missing_fields():
    """Test 9: Stocks with missing optional fields."""
    print("\n" + "="*80)
    print("TEST 9: Missing Optional Fields")
    print("="*80)
    
    stocks = [
        {
            "stock_name": "騰訊控股",
            "stock_number": "00700"
            # Missing price, quantity, order_type
        }
    ]
    
    test_file = create_test_stocks_file("test_missing_fields", stocks)
    print(f"Created test file: {test_file}")
    
    result = verify_stocks.func(test_file)
    print("\nResult:")
    print(result)
    
    return result


def test_real_world_example():
    """Test 10: Real-world example similar to actual usage."""
    print("\n" + "="*80)
    print("TEST 10: Real-World Example")
    print("="*80)
    
    # Based on the actual example file structure
    stocks = [
        {
            "stock_name": "快手",
            "stock_number": "01680",
            "price": 89.5,
            "quantity": 400,
            "order_type": "bid"
        },
        {
            "stock_name": "快手",
            "stock_number": "01680",
            "price": 83.1,
            "quantity": 400,
            "order_type": "bid"
        },
        {
            "stock_name": "快手",
            "stock_number": "01680",
            "price": 100.0,
            "quantity": 400,
            "order_type": "ask"
        }
    ]
    
    test_file = create_test_stocks_file("test_real_world", stocks)
    print(f"Created test file: {test_file}")
    
    result = verify_stocks.func(test_file)
    print("\nResult:")
    print(result)
    
    # Check verified file
    verified_file = test_file.replace('.json', '_verified.json')
    if os.path.exists(verified_file):
        with open(verified_file, 'r', encoding='utf-8') as f:
            verified_data = json.load(f)
        print("\nVerified JSON Structure:")
        print(f"  Source file: {verified_data.get('source_file')}")
        print(f"  Audio filename: {verified_data.get('audio_filename')}")
        print(f"  Original timestamp: {verified_data.get('original_timestamp')}")
        print(f"  Verification timestamp: {verified_data.get('verification_timestamp')}")
        print(f"  Number of stocks: {len(verified_data.get('stocks', []))}")
        
        for i, stock in enumerate(verified_data.get('stocks', []), 1):
            print(f"\n  Stock {i}:")
            print(f"    Name: {stock.get('stock_name')}")
            print(f"    Code: {stock.get('stock_number')}")
            verification = stock.get('verification', {})
            print(f"    Verified: {verification.get('verified')}")
            print(f"    Corrected: {verification.get('corrected')}")
            print(f"    Confidence: {verification.get('confidence', 0):.2%}")
            print(f"    Match Type: {verification.get('match_type')}")
    
    return result


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*80)
    print("STOCK VERIFIER TOOL - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    test_results = []
    
    tests = [
        ("Exact Match", test_exact_match),
        ("Fuzzy Name Match", test_fuzzy_name_match),
        ("Code Only Match", test_code_only_match),
        ("Name Only Match", test_name_only_match),
        ("Multiple Stocks", test_multiple_stocks),
        ("Fuzzy Code Match", test_fuzzy_code_match),
        ("Unverified Stock", test_unverified_stock),
        ("Empty Stocks", test_empty_stocks),
        ("Missing Fields", test_missing_fields),
        ("Real-World Example", test_real_world_example),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'#'*80}")
            print(f"Running: {test_name}")
            print(f"{'#'*80}")
            result = test_func()
            test_results.append((test_name, "PASSED", result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            test_results.append((test_name, "FAILED", str(e)))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, status, _ in test_results if status == "PASSED")
    failed = sum(1 for _, status, _ in test_results if status == "FAILED")
    
    for test_name, status, _ in test_results:
        status_symbol = "✅" if status == "PASSED" else "❌"
        print(f"{status_symbol} {test_name}: {status}")
    
    print(f"\nTotal: {len(test_results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("="*80)
    
    # Print test files location
    test_dir = os.path.join(os.path.dirname(__file__), "test_stocks")
    print(f"\nTest files created in: {test_dir}")
    print(f"Verified files will be in the same directory with '_verified.json' suffix")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test stock verifier tool")
    parser.add_argument(
        "--test",
        type=str,
        help="Run specific test (exact, fuzzy_name, code_only, name_only, multiple, fuzzy_code, unverified, empty, missing_fields, real_world)",
        choices=["exact", "fuzzy_name", "code_only", "name_only", "multiple", "fuzzy_code", "unverified", "empty", "missing_fields", "real_world", "all"]
    )
    
    args = parser.parse_args()
    
    if args.test == "exact":
        test_exact_match()
    elif args.test == "fuzzy_name":
        test_fuzzy_name_match()
    elif args.test == "code_only":
        test_code_only_match()
    elif args.test == "name_only":
        test_name_only_match()
    elif args.test == "multiple":
        test_multiple_stocks()
    elif args.test == "fuzzy_code":
        test_fuzzy_code_match()
    elif args.test == "unverified":
        test_unverified_stock()
    elif args.test == "empty":
        test_empty_stocks()
    elif args.test == "missing_fields":
        test_missing_fields()
    elif args.test == "real_world":
        test_real_world_example()
    elif args.test == "all" or args.test is None:
        run_all_tests()
    else:
        print(f"Unknown test: {args.test}")
        parser.print_help()

