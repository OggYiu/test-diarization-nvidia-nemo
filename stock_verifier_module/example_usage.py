"""
Example usage of the Stock Verifier Module

This script demonstrates various ways to use the stock verifier.
"""

import logging
from stock_verifier_improved import (
    verify_and_correct_stock,
    batch_verify_stocks,
    SearchStrategy,
    format_correction_summary,
    get_vector_store,
)


def example_1_basic_verification():
    """Example 1: Basic stock verification"""
    print("\n" + "=" * 80)
    print("Example 1: Basic Stock Verification")
    print("=" * 80)
    
    # Test the problematic case mentioned by user
    result = verify_and_correct_stock(
        stock_name="騰訊升認購證",
        stock_code="18138"
    )
    
    print(f"\nInput:")
    print(f"  Code: {result.original_stock_code}")
    print(f"  Name: {result.original_stock_name}")
    
    print(f"\nOutput:")
    print(f"  Corrected Code: {result.corrected_stock_code}")
    print(f"  Corrected Name: {result.corrected_stock_name}")
    print(f"  Confidence: {result.confidence:.2%} ({result.confidence_level})")
    print(f"  Correction Applied: {result.correction_applied}")
    print(f"  Reasoning: {result.reasoning}")


def example_2_different_strategies():
    """Example 2: Comparing different search strategies"""
    print("\n" + "=" * 80)
    print("Example 2: Comparing Search Strategies")
    print("=" * 80)
    
    test_case = {
        "stock_name": "騰訊升認購證",
        "stock_code": "18138"
    }
    
    strategies = [
        ("Optimized (Default)", SearchStrategy.OPTIMIZED),
        ("Semantic Only", SearchStrategy.SEMANTIC_ONLY),
        ("Exact Only", SearchStrategy.EXACT_ONLY),
    ]
    
    for strategy_name, strategy in strategies:
        print(f"\n{strategy_name}:")
        print("-" * 40)
        
        result = verify_and_correct_stock(
            stock_name=test_case["stock_name"],
            stock_code=test_case["stock_code"],
            strategy=strategy
        )
        
        print(f"  Result: {result.corrected_stock_name or result.original_stock_name}")
        print(f"  Code: {result.corrected_stock_code or result.original_stock_code}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Reasoning: {result.reasoning}")


def example_3_batch_processing():
    """Example 3: Batch processing multiple stocks"""
    print("\n" + "=" * 80)
    print("Example 3: Batch Processing")
    print("=" * 80)
    
    # Multiple stocks to verify
    stocks = [
        {"stock_name": "騰訊升認購證", "stock_code": "18138"},
        {"stock_name": "騰訊", "stock_code": "700"},
        {"stock_name": "小米", "stock_code": "1810"},
        {"stock_name": "泡泡沬特", "stock_code": None},  # STT error
    ]
    
    print(f"\nVerifying {len(stocks)} stocks...")
    results = batch_verify_stocks(stocks)
    
    print("\nResults:")
    print("-" * 80)
    
    for i, (stock, result) in enumerate(zip(stocks, results), 1):
        print(f"\n{i}. {stock.get('stock_name')} ({stock.get('stock_code') or 'N/A'})")
        print(f"   → {result.corrected_stock_name or result.original_stock_name}")
        print(f"   → Code: {result.corrected_stock_code or result.original_stock_code}")
        print(f"   → Confidence: {result.confidence:.2%}")
        if result.correction_applied:
            print(f"   → ✅ Correction applied")


def example_4_code_only_name_only():
    """Example 4: Code-only and name-only lookups"""
    print("\n" + "=" * 80)
    print("Example 4: Code-Only and Name-Only Lookups")
    print("=" * 80)
    
    # Code only
    print("\nCode-only lookup (700):")
    result = verify_and_correct_stock(stock_code="700")
    print(f"  Found: {result.corrected_stock_name or result.original_stock_name}")
    print(f"  Confidence: {result.confidence:.2%}")
    
    # Name only
    print("\nName-only lookup (騰訊控股):")
    result = verify_and_correct_stock(stock_name="騰訊控股")
    print(f"  Found: {result.corrected_stock_code or result.original_stock_code}")
    print(f"  Confidence: {result.confidence:.2%}")


def example_5_confidence_thresholds():
    """Example 5: Using different confidence thresholds"""
    print("\n" + "=" * 80)
    print("Example 5: Confidence Thresholds")
    print("=" * 80)
    
    test_stock = {"stock_name": "泡泡沬特", "stock_code": None}
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        print(f"\nThreshold: {threshold:.1f}")
        print("-" * 40)
        
        result = verify_and_correct_stock(
            stock_name=test_stock["stock_name"],
            stock_code=test_stock["stock_code"],
            confidence_threshold=threshold
        )
        
        print(f"  Correction Applied: {result.correction_applied}")
        print(f"  Confidence: {result.confidence:.2%}")
        if result.corrected_stock_name:
            print(f"  Suggested: {result.corrected_stock_name}")


def example_6_reusing_vector_store():
    """Example 6: Reusing vector store for better performance"""
    print("\n" + "=" * 80)
    print("Example 6: Reusing Vector Store (Performance)")
    print("=" * 80)
    
    import time
    
    stocks = [
        {"stock_name": "騰訊", "stock_code": "700"},
        {"stock_name": "小米", "stock_code": "1810"},
        {"stock_name": "阿里巴巴", "stock_code": "9988"},
    ]
    
    # Method 1: Create new connection each time (slower)
    print("\nMethod 1: New connection each time")
    start = time.time()
    for stock in stocks:
        result = verify_and_correct_stock(
            stock_name=stock["stock_name"],
            stock_code=stock["stock_code"]
        )
    method1_time = time.time() - start
    print(f"  Time: {method1_time:.2f}s")
    
    # Method 2: Reuse vector store (faster)
    print("\nMethod 2: Reuse vector store")
    vector_store = get_vector_store()
    vector_store.initialize()
    
    start = time.time()
    for stock in stocks:
        result = verify_and_correct_stock(
            stock_name=stock["stock_name"],
            stock_code=stock["stock_code"],
            vector_store=vector_store  # Reuse
        )
    method2_time = time.time() - start
    print(f"  Time: {method2_time:.2f}s")
    
    if method1_time > 0:
        speedup = method1_time / method2_time
        print(f"\n  Speedup: {speedup:.2f}x faster")


def example_7_format_summary():
    """Example 7: Using format_correction_summary"""
    print("\n" + "=" * 80)
    print("Example 7: Formatted Summaries")
    print("=" * 80)
    
    stocks = [
        {"stock_name": "騰訊升認購證", "stock_code": "18138"},
        {"stock_name": "騰訊控股", "stock_code": "00700"},  # Already correct
    ]
    
    for stock in stocks:
        result = verify_and_correct_stock(
            stock_name=stock["stock_name"],
            stock_code=stock["stock_code"]
        )
        
        print(f"\n{stock['stock_name']}:")
        summary = format_correction_summary(result)
        print(summary)


def example_8_debugging_candidates():
    """Example 8: Examining all candidates for debugging"""
    print("\n" + "=" * 80)
    print("Example 8: Debugging with All Candidates")
    print("=" * 80)
    
    result = verify_and_correct_stock(
        stock_name="騰訊升認購證",
        stock_code="18138"
    )
    
    print("\nTop candidates:")
    print("-" * 80)
    
    for i, candidate in enumerate(result.all_candidates[:5], 1):
        print(f"\n{i}. Confidence: {candidate['confidence']:.4f}")
        print(f"   Match Type: {candidate.get('match_type', 'N/A')}")
        print(f"   Query: {candidate.get('query', 'N/A')}")
        
        doc = candidate.get('doc')
        if doc and hasattr(doc, 'metadata'):
            print(f"   Code: {doc.metadata.get('InstrumentCd', 'N/A')}")
            print(f"   Name: {doc.metadata.get('AliasName', 'N/A')}")


def main():
    """Run all examples"""
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Set to INFO to see detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 80)
    print("STOCK VERIFIER - EXAMPLE USAGE")
    print("=" * 80)
    print("\nThis script demonstrates various ways to use the stock verifier.")
    print("Each example shows a different feature or use case.")
    
    # Run examples
    examples = [
        example_1_basic_verification,
        example_2_different_strategies,
        example_3_batch_processing,
        example_4_code_only_name_only,
        example_5_confidence_thresholds,
        example_6_reusing_vector_store,
        example_7_format_summary,
        example_8_debugging_candidates,
    ]
    
    for example in examples:
        try:
            example()
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {str(e)}")
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nFor more information:")
    print("  - See README.md for overview and quick start")
    print("  - See USAGE.md for detailed API documentation")
    print("  - Run 'python test_runner.py' to run test suite")
    print()


if __name__ == "__main__":
    main()

