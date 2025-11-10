"""
Stock Verifier Module

An improved stock name and code verifier using Milvus vector store
with enhanced search strategies for better accuracy.

Main Features:
- Prioritized exact matching for stock codes
- Multiple search strategies (optimized, semantic-only, exact-only)
- Flexible test framework with JSON-based test cases
- Detailed reporting and confidence scoring

Quick Start:
    >>> from stock_verifier_module import verify_and_correct_stock
    >>> result = verify_and_correct_stock(
    ...     stock_name="騰訊升認購證",
    ...     stock_code="18138"
    ... )
    >>> print(f"Corrected: {result.corrected_stock_name}")

For more examples, see example_usage.py
For detailed documentation, see README.md and USAGE.md
"""

from .stock_verifier_improved import (
    # Main functions
    verify_and_correct_stock,
    batch_verify_stocks,
    
    # Enums
    SearchStrategy,
    
    # Data classes
    StockCorrectionResult,
    
    # Vector store
    StockVectorStore,
    get_vector_store,
    
    # Utilities
    normalize_stock_code,
    is_valid_stock_code,
    format_correction_summary,
    
    # Constants
    CONFIDENCE_THRESHOLD_HIGH,
    CONFIDENCE_THRESHOLD_MEDIUM,
    CONFIDENCE_THRESHOLD_LOW,
)

__version__ = "1.0.0"
__author__ = "Stock Verifier Team"
__all__ = [
    # Main functions
    "verify_and_correct_stock",
    "batch_verify_stocks",
    
    # Enums
    "SearchStrategy",
    
    # Data classes
    "StockCorrectionResult",
    
    # Vector store
    "StockVectorStore",
    "get_vector_store",
    
    # Utilities
    "normalize_stock_code",
    "is_valid_stock_code",
    "format_correction_summary",
    
    # Constants
    "CONFIDENCE_THRESHOLD_HIGH",
    "CONFIDENCE_THRESHOLD_MEDIUM",
    "CONFIDENCE_THRESHOLD_LOW",
]






