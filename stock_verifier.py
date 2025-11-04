"""
Stock Name and Code Verifier using Milvus Vector Store

This module provides functionality to verify and correct stock names and codes
that may have been incorrectly transcribed by Speech-to-Text systems.
Uses semantic similarity search powered by Milvus and Ollama embeddings.
"""

import logging
import re
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus


# ============================================================================
# Configuration
# ============================================================================

MILVUS_CLUSTER_ENDPOINT = "https://in03-5fb5696b79d8b56.serverless.aws-eu-central-1.cloud.zilliz.com"
MILVUS_TOKEN = "9052d0067c0dd76fc12de51d2cc7a456dcd6caf58e72e344a2c372c85d6f7b486f39d1f2fd15916a7a9234127760e622c3145c36"
MILVUS_DB_NAME = "stocks"
MILVUS_COLLECTION_NAME = "phone_calls"
EMBEDDING_MODEL = "qwen3-embedding:8b"

# Thresholds for correction confidence
CONFIDENCE_THRESHOLD_HIGH = 0.8  # Very confident correction
CONFIDENCE_THRESHOLD_MEDIUM = 0.6  # Moderately confident correction
CONFIDENCE_THRESHOLD_LOW = 0.4  # Low confidence, but worth suggesting

# Search parameters
TOP_K_RESULTS = 5  # Number of results to retrieve per query
MAX_SEARCH_ATTEMPTS = 3  # Maximum search variations to try


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StockCorrectionResult:
    """Result of stock verification/correction"""
    original_stock_name: Optional[str] = None
    original_stock_code: Optional[str] = None
    corrected_stock_name: Optional[str] = None
    corrected_stock_code: Optional[str] = None
    confidence: float = 0.0
    correction_applied: bool = False
    confidence_level: str = "none"  # none, low, medium, high
    reasoning: str = ""
    matched_content: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Determine confidence level
        if self.confidence >= CONFIDENCE_THRESHOLD_HIGH:
            self.confidence_level = "high"
        elif self.confidence >= CONFIDENCE_THRESHOLD_MEDIUM:
            self.confidence_level = "medium"
        elif self.confidence >= CONFIDENCE_THRESHOLD_LOW:
            self.confidence_level = "low"
        else:
            self.confidence_level = "none"


# ============================================================================
# Vector Store Management
# ============================================================================

class StockVectorStore:
    """Manages connection to Milvus vector store for stock data"""
    
    def __init__(
        self,
        endpoint: str = MILVUS_CLUSTER_ENDPOINT,
        token: str = MILVUS_TOKEN,
        db_name: str = MILVUS_DB_NAME,
        collection_name: str = MILVUS_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.endpoint = endpoint
        self.token = token
        self.db_name = db_name
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._vector_store = None
        self._is_initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the Milvus vector store connection.
        Returns True if successful, False otherwise.
        """
        if self._is_initialized and self._vector_store is not None:
            return True
        
        try:
            logging.info(f"Initializing Milvus vector store (embedding: {self.embedding_model})...")
            
            embeddings = OllamaEmbeddings(model=self.embedding_model)
            
            self._vector_store = Milvus(
                embedding_function=embeddings,
                collection_name=self.collection_name,
                connection_args={
                    "uri": self.endpoint,
                    "token": self.token,
                    "db_name": self.db_name
                },
                consistency_level="Strong",
                drop_old=False,
            )
            
            self._is_initialized = True
            logging.info("✅ Vector store initialized successfully!")
            return True
        
        except Exception as e:
            logging.error(f"❌ Failed to initialize vector store: {str(e)}")
            self._is_initialized = False
            return False
    
    def search(self, query: str, k: int = TOP_K_RESULTS) -> List[Tuple[Any, float]]:
        """
        Search the vector store with similarity scores.
        Returns list of (document, score) tuples.
        """
        if not self._is_initialized:
            if not self.initialize():
                return []
        
        try:
            results = self._vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logging.error(f"Error searching vector store for '{query}': {str(e)}")
            return []
    
    @property
    def is_available(self) -> bool:
        """Check if vector store is available"""
        return self._is_initialized and self._vector_store is not None


# Global instance (singleton pattern)
_global_vector_store = None


def get_vector_store() -> StockVectorStore:
    """Get or create the global vector store instance"""
    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = StockVectorStore()
    return _global_vector_store


# ============================================================================
# Stock Code Normalization
# ============================================================================

def normalize_stock_code(code: str) -> str:
    """
    Normalize stock code to standard format.
    Examples: '700' -> '00700', '1810' -> '01810'
    """
    if not code:
        return ""
    
    # Remove whitespace and special characters
    code = re.sub(r'[^\d]', '', code)
    
    # Pad with zeros to 5 digits for HK stocks
    if code.isdigit() and len(code) < 5:
        code = code.zfill(5)
    
    return code


def is_valid_stock_code(code: str) -> bool:
    """Check if a string looks like a valid stock code"""
    if not code:
        return False
    
    normalized = normalize_stock_code(code)
    return normalized.isdigit() and 1 <= len(normalized) <= 6


# ============================================================================
# Data Parsing Utilities
# ============================================================================

def parse_stock_from_content(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse stock code and name from content string.
    Supports multiple formats:
    - CSV: "00700,騰訊控股"
    - Space: "00700 騰訊控股"
    - Tab: "00700\t騰訊控股"
    - Mixed: "股票代號:00700 名稱:騰訊控股"
    
    Returns: (stock_code, stock_name)
    """
    if not content:
        return None, None
    
    content = content.strip()
    stock_code = None
    stock_name = None
    
    # Strategy 1: Look for explicit markers
    code_patterns = [
        r'(?:股票代號|代號|code|stock_code)[:：\s]+([0-9]+)',
        r'(?:編號|number)[:：\s]+([0-9]+)',
    ]
    
    name_patterns = [
        r'(?:股票名稱|名稱|name|stock_name)[:：\s]+([^,\n\r]+)',
        r'(?:公司|company)[:：\s]+([^,\n\r]+)',
    ]
    
    for pattern in code_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            stock_code = normalize_stock_code(match.group(1))
            break
    
    for pattern in name_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            stock_name = match.group(1).strip()
            break
    
    # Strategy 2: Try delimiter-based parsing
    if not stock_code or not stock_name:
        # Try comma separation
        if ',' in content:
            parts = content.split(',', 1)
            if len(parts) >= 1 and not stock_code:
                candidate = parts[0].strip()
                if is_valid_stock_code(candidate):
                    stock_code = normalize_stock_code(candidate)
            if len(parts) >= 2 and not stock_name:
                stock_name = parts[1].strip()
        
        # Try tab separation
        elif '\t' in content:
            parts = content.split('\t', 1)
            if len(parts) >= 1 and not stock_code:
                candidate = parts[0].strip()
                if is_valid_stock_code(candidate):
                    stock_code = normalize_stock_code(candidate)
            if len(parts) >= 2 and not stock_name:
                stock_name = parts[1].strip()
        
        # Try space separation (be careful with Chinese names)
        elif ' ' in content:
            # Look for number at the start
            match = re.match(r'^([0-9]+)\s+(.+)$', content)
            if match:
                if not stock_code:
                    stock_code = normalize_stock_code(match.group(1))
                if not stock_name:
                    stock_name = match.group(2).strip()
    
    # Strategy 3: If only one value, determine what it is
    if not stock_code and not stock_name:
        if is_valid_stock_code(content):
            stock_code = normalize_stock_code(content)
        elif content and not content.isdigit():
            stock_name = content
    
    return stock_code, stock_name


def extract_from_metadata(metadata: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract stock code and name from metadata dictionary.
    Checks multiple possible field names.
    
    Returns: (stock_code, stock_name)
    """
    if not metadata:
        return None, None
    
    # Possible field names for stock code
    code_fields = ['stock_code', 'InstrumentCd', 'code', 'stock_number', 'instrument_cd', 'symbol']
    # Possible field names for stock name
    name_fields = ['stock_name', 'AliasName', 'name', 'company_name', 'alias_name', 'company']
    
    stock_code = None
    stock_name = None
    
    # Try to find stock code
    for field in code_fields:
        if field in metadata and metadata[field]:
            candidate = str(metadata[field])
            if is_valid_stock_code(candidate):
                stock_code = normalize_stock_code(candidate)
                break
    
    # Try to find stock name
    for field in name_fields:
        if field in metadata and metadata[field]:
            stock_name = str(metadata[field]).strip()
            break
    
    return stock_code, stock_name


# ============================================================================
# Core Verification and Correction Logic
# ============================================================================

def verify_and_correct_stock(
    stock_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    vector_store: Optional[StockVectorStore] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_LOW,
    top_k: int = TOP_K_RESULTS,
) -> StockCorrectionResult:
    """
    Verify and correct stock name/code using vector store.
    
    Args:
        stock_name: Original stock name from STT/LLM
        stock_code: Original stock code from STT/LLM
        vector_store: Vector store instance (creates new if None)
        confidence_threshold: Minimum confidence to suggest correction
        top_k: Number of results to retrieve
    
    Returns:
        StockCorrectionResult with correction details
    """
    # Initialize result
    result = StockCorrectionResult(
        original_stock_name=stock_name,
        original_stock_code=normalize_stock_code(stock_code) if stock_code else None,
    )
    
    # Get vector store
    if vector_store is None:
        vector_store = get_vector_store()
    
    if not vector_store.is_available:
        if not vector_store.initialize():
            result.reasoning = "Vector store not available"
            return result
    
    # Prepare search queries
    queries = []
    query_weights = []  # Weight for each query type
    
    if stock_name:
        queries.append(stock_name)
        query_weights.append(1.0)  # Name search is most reliable
        
        # Add variations for better matching
        # Remove common suffixes that might be transcription errors
        name_variations = generate_name_variations(stock_name)
        for variation in name_variations:
            queries.append(variation)
            query_weights.append(0.8)  # Slightly lower weight for variations
    
    if stock_code:
        normalized_code = normalize_stock_code(stock_code)
        queries.append(normalized_code)
        query_weights.append(0.9)  # Code search is quite reliable
        
        # Also try without padding
        if normalized_code != stock_code:
            queries.append(stock_code)
            query_weights.append(0.7)
    
    if not queries:
        result.reasoning = "No stock name or code provided"
        return result
    
    # Search vector store with all queries
    all_matches = []
    
    for query, weight in zip(queries, query_weights):
        try:
            search_results = vector_store.search(query, k=top_k)
            
            for doc, score in search_results:
                # Convert score to confidence (lower distance = higher confidence)
                # Milvus returns L2 distance, lower is better
                confidence = 1.0 / (1.0 + score) if score > 0 else 1.0
                weighted_confidence = confidence * weight
                
                all_matches.append({
                    'doc': doc,
                    'score': score,
                    'confidence': weighted_confidence,
                    'query': query,
                    'weight': weight,
                })
        
        except Exception as e:
            logging.warning(f"Error searching for '{query}': {str(e)}")
            continue
    
    if not all_matches:
        result.reasoning = "No matches found in vector store"
        return result
    
    # Sort by confidence (highest first)
    all_matches.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Get best match
    best_match = all_matches[0]
    best_doc = best_match['doc']
    best_confidence = best_match['confidence']
    
    # Extract stock information from best match
    content = best_doc.page_content if hasattr(best_doc, 'page_content') else str(best_doc)
    metadata = best_doc.metadata if hasattr(best_doc, 'metadata') else {}
    
    # Try metadata first, then content
    corrected_code, corrected_name = extract_from_metadata(metadata)
    
    if not corrected_code or not corrected_name:
        parsed_code, parsed_name = parse_stock_from_content(content)
        corrected_code = corrected_code or parsed_code
        corrected_name = corrected_name or parsed_name
    
    # Determine if correction should be applied
    result.confidence = round(best_confidence, 4)
    result.matched_content = content[:200]  # Store preview
    result.metadata = metadata
    
    # Check if there's a meaningful difference
    has_correction = False
    correction_details = []
    
    if corrected_name and corrected_name != stock_name:
        result.corrected_stock_name = corrected_name
        has_correction = True
        correction_details.append(f"名稱: {stock_name} → {corrected_name}")
    
    if corrected_code and corrected_code != result.original_stock_code:
        result.corrected_stock_code = corrected_code
        has_correction = True
        correction_details.append(f"代號: {result.original_stock_code or 'N/A'} → {corrected_code}")
    
    # Apply correction if confidence is high enough
    if has_correction and best_confidence >= confidence_threshold:
        result.correction_applied = True
        result.reasoning = f"Vector store correction ({best_confidence:.1%} confidence): {'; '.join(correction_details)}"
    elif has_correction:
        result.reasoning = f"Possible correction found but confidence too low ({best_confidence:.1%}): {'; '.join(correction_details)}"
    else:
        result.reasoning = f"Verified correct ({best_confidence:.1%} confidence)"
    
    # Log for debugging
    logging.info(f"Stock verification: {stock_name or stock_code} -> {result.reasoning}")
    
    return result


def generate_name_variations(name: str) -> List[str]:
    """
    Generate variations of stock name for better matching.
    Handles common STT transcription errors.
    """
    if not name:
        return []
    
    variations = []
    
    # Common character substitutions (STT errors)
    substitutions = {
        '百': ['八', '伯'],
        '八': ['百'],
        '孤': ['沽', '股'],
        '沽': ['孤', '股'],
        '轮': ['輪'],
        '星': ['升'],
        '號': ['毫', '豪'],
        '毫': ['號', '豪'],
        '特': ['得', '德'],
        '沬': ['瑪', '馬'],
    }
    
    # Generate variations by replacing one character at a time
    for i, char in enumerate(name):
        if char in substitutions:
            for replacement in substitutions[char]:
                variation = name[:i] + replacement + name[i+1:]
                if variation != name:
                    variations.append(variation)
    
    # Limit to avoid too many variations
    return variations[:3]


def batch_verify_stocks(
    stocks: List[Dict[str, Optional[str]]],
    vector_store: Optional[StockVectorStore] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_LOW,
) -> List[StockCorrectionResult]:
    """
    Verify and correct multiple stocks in batch.
    
    Args:
        stocks: List of dicts with 'stock_name' and/or 'stock_code' keys
        vector_store: Vector store instance (creates new if None)
        confidence_threshold: Minimum confidence to suggest correction
    
    Returns:
        List of StockCorrectionResult objects
    """
    if vector_store is None:
        vector_store = get_vector_store()
    
    # Initialize once for all stocks
    if not vector_store.is_available:
        vector_store.initialize()
    
    results = []
    for stock in stocks:
        result = verify_and_correct_stock(
            stock_name=stock.get('stock_name'),
            stock_code=stock.get('stock_code'),
            vector_store=vector_store,
            confidence_threshold=confidence_threshold,
        )
        results.append(result)
    
    return results


# ============================================================================
# Utility Functions
# ============================================================================

def format_correction_summary(result: StockCorrectionResult) -> str:
    """Format correction result as a readable summary"""
    if not result.correction_applied:
        return f"✓ 已驗證: {result.original_stock_name or result.original_stock_code}"
    
    lines = []
    lines.append(f"🔧 修正建議 ({result.confidence_level.upper()}, {result.confidence:.1%}):")
    
    if result.corrected_stock_name:
        lines.append(f"  名稱: {result.original_stock_name} → {result.corrected_stock_name}")
    
    if result.corrected_stock_code:
        lines.append(f"  代號: {result.original_stock_code or 'N/A'} → {result.corrected_stock_code}")
    
    return "\n".join(lines)


# ============================================================================
# Testing / Example Usage
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    print("=" * 80)
    print("Stock Verifier - Example Usage")
    print("=" * 80)
    
    # Test cases with common STT errors
    test_stocks = [
        {"stock_name": "騰訊", "stock_code": "700"},
        {"stock_name": "泡泡沬特", "stock_code": None},  # STT error: 沬 should be 瑪
        {"stock_name": "小米", "stock_code": "1810"},
        {"stock_name": None, "stock_code": "00700"},
    ]
    
    print("\nVerifying stocks...\n")
    
    for i, stock in enumerate(test_stocks, 1):
        print(f"Test {i}: {stock}")
        result = verify_and_correct_stock(
            stock_name=stock.get("stock_name"),
            stock_code=stock.get("stock_code"),
        )
        
        print(format_correction_summary(result))
        print(f"Reasoning: {result.reasoning}")
        print("-" * 80)

