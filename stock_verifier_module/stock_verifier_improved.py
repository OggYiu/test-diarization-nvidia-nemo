"""
Stock Name and Code Verifier using Milvus Vector Store (Improved)

This module provides enhanced functionality to verify and correct stock names and codes
that may have been incorrectly transcribed by Speech-to-Text systems.

Key Improvements:
- Prioritizes exact stock code matches over semantic similarity
- Multiple search strategies: optimized (default), semantic-only, exact-only
- Better handling of warrant and derivative securities
- Improved confidence scoring
"""

import logging
import re
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
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
TOP_K_RESULTS = 10  # Increased to get more candidates for exact matching
MAX_SEARCH_ATTEMPTS = 3


# ============================================================================
# Enums
# ============================================================================

class SearchStrategy(Enum):
    """Search strategy for stock verification"""
    OPTIMIZED = "optimized"  # Default: Exact code match + semantic (recommended)
    SEMANTIC_ONLY = "semantic_only"  # Pure semantic similarity search
    EXACT_ONLY = "exact_only"  # Only exact code/name matches


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    search_strategy: str = "optimized"
    all_candidates: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
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
    
    def search_by_metadata(self, stock_code: Optional[str] = None, k: int = TOP_K_RESULTS) -> List[Tuple[Any, float]]:
        """
        Search the vector store by metadata filter (stock code).
        Returns list of (document, score) tuples.
        
        Args:
            stock_code: Stock code to filter by (normalized format)
            k: Number of results to retrieve
        """
        if not self._is_initialized:
            if not self.initialize():
                return []
        
        if not stock_code:
            return []
        
        # Try both normalized and non-normalized versions
        # The database may store codes with or without leading zeros
        codes_to_try = set()
        
        # Add the normalized version (padded to 5 digits)
        codes_to_try.add(stock_code)
        
        # Add the non-padded version (strip leading zeros)
        non_padded = stock_code.lstrip('0')
        if non_padded:
            codes_to_try.add(non_padded)
        
        # If input was short, also add longer padded versions
        if len(stock_code) < 5:
            codes_to_try.add(stock_code.zfill(5))
        
        codes_to_try = list(codes_to_try)
        
        try:
            # Use Milvus expr filter to find exact stock code matches
            # Build filter for all code variations
            code_filters = ' or '.join([f'stock_code == "{code}"' for code in codes_to_try])
            
            # For metadata filtering, we need a dummy query but with filter
            # We'll search with a generic query and filter by metadata
            results = self._vector_store.similarity_search_with_score(
                query=stock_code,  # Use the code as query
                k=k * 10,  # Get more results since filter will reduce count
                expr=code_filters
            )
            return results
        except Exception as e:
            logging.warning(f"Metadata search not supported or error: {str(e)}")
            # Fall back to manual filtering from a broader search
            try:
                # Search broadly and manually filter
                # Try multiple search queries to cast a wider net
                all_results = []
                for search_query in codes_to_try + [f"stock {stock_code}", f"code {stock_code}"]:
                    try:
                        broad_results = self._vector_store.similarity_search_with_score(search_query, k=k * 20)
                        all_results.extend(broad_results)
                    except:
                        continue
                
                # Manual filtering
                filtered_results = []
                seen_pks = set()  # Deduplicate by primary key
                
                for doc, score in all_results:
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    
                    # Get primary key for deduplication
                    pk = metadata.get('pk', id(doc))
                    if pk in seen_pks:
                        continue
                    
                    doc_code = metadata.get('stock_code', '')
                    
                    # Check if this document matches the target code (handles padding)
                    if doc_code and codes_match(doc_code, stock_code):
                        filtered_results.append((doc, score))
                        seen_pks.add(pk)
                    
                    if len(filtered_results) >= k:
                        break
                
                logging.info(f"Manual filtering found {len(filtered_results)} matches for code {stock_code}")
                return filtered_results
            except Exception as e2:
                logging.error(f"Error in fallback metadata search: {str(e2)}")
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
    Examples: '700' -> '00700', '1810' -> '01810', '18138' -> '18138'
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


def codes_match(code1: Optional[str], code2: Optional[str]) -> bool:
    """
    Check if two stock codes match, handling both padded and non-padded versions.
    
    Examples:
        codes_match('9992', '09992') -> True
        codes_match('00700', '700') -> True
        codes_match('18138', '18138') -> True
        codes_match('700', '1810') -> False
    
    Args:
        code1: First stock code
        code2: Second stock code
    
    Returns:
        True if codes match (considering padding), False otherwise
    """
    if not code1 or not code2:
        return False
    
    # Remove whitespace and special characters
    code1 = re.sub(r'[^\d]', '', str(code1))
    code2 = re.sub(r'[^\d]', '', str(code2))
    
    # Convert to integers for comparison (removes leading zeros)
    try:
        int_code1 = int(code1)
        int_code2 = int(code2)
        return int_code1 == int_code2
    except (ValueError, TypeError):
        # If conversion fails, fall back to string comparison
        return code1 == code2


def calculate_name_similarity(name1: Optional[str], name2: Optional[str]) -> float:
    """
    Calculate similarity between two stock names.
    Returns a score from 0.0 to 1.0 where 1.0 is exact match.
    
    This uses simple character overlap and common prefix matching.
    """
    if not name1 or not name2:
        return 0.0
    
    # Normalize names
    name1 = name1.strip()
    name2 = name2.strip()
    
    # Exact match
    if name1 == name2:
        return 1.0
    
    # Check if one contains the other
    if name1 in name2 or name2 in name1:
        return 0.8
    
    # Calculate character overlap
    set1 = set(name1)
    set2 = set(name2)
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    jaccard_similarity = intersection / union if union > 0 else 0.0
    
    # Calculate common prefix length (important for Chinese names)
    common_prefix_len = 0
    for c1, c2 in zip(name1, name2):
        if c1 == c2:
            common_prefix_len += 1
        else:
            break
    
    prefix_similarity = common_prefix_len / max(len(name1), len(name2))
    
    # Weighted combination: prefix is more important for Chinese stock names
    similarity = 0.6 * prefix_similarity + 0.4 * jaccard_similarity
    
    return similarity


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
        
        # Try space separation
        elif ' ' in content:
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
    
    Returns: (stock_code, stock_name)
    """
    if not metadata:
        return None, None
    
    # Possible field names for stock code
    code_fields = ['stock_code', 'InstrumentCd', 'code', 'stock_number', 'instrument_cd', 'symbol']
    # Possible field names for stock name (Chinese name preferred)
    name_fields = ['stock_name_c', 'stock_name', 'AliasName', 'name', 'company_name', 'alias_name', 'company']
    
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
# Search Strategy Implementation
# ============================================================================

def exact_code_search(
    search_results: List[Tuple[Any, float]],
    target_code: str
) -> Optional[Dict[str, Any]]:
    """
    Find exact code match from search results.
    
    Args:
        search_results: List of (document, score) tuples from vector search
        target_code: Normalized stock code to match
    
    Returns:
        Match dict with doc, score, and confidence, or None if not found
    """
    if not target_code or not search_results:
        return None
    
    for doc, score in search_results:
        # Extract code from metadata
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        doc_code, _ = extract_from_metadata(metadata)
        
        # If not in metadata, try content
        if not doc_code:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            doc_code, _ = parse_stock_from_content(content)
        
        # Check for exact match (handles both padded and non-padded versions)
        if doc_code and codes_match(doc_code, target_code):
            # Exact code match gets very high confidence
            confidence = 0.95
            return {
                'doc': doc,
                'score': score,
                'confidence': confidence,
                'match_type': 'exact_code',
                'query': target_code,
            }
    
    return None


def semantic_search_strategy(
    stock_name: Optional[str],
    stock_code: Optional[str],
    vector_store: StockVectorStore,
    top_k: int = TOP_K_RESULTS,
) -> List[Dict[str, Any]]:
    """
    Pure semantic similarity search.
    
    Returns:
        List of match dictionaries sorted by confidence
    """
    queries = []
    query_weights = []
    
    if stock_name:
        queries.append(stock_name)
        query_weights.append(1.0)
        
        # Add variations
        name_variations = generate_name_variations(stock_name)
        for variation in name_variations:
            queries.append(variation)
            query_weights.append(0.8)
    
    if stock_code:
        normalized_code = normalize_stock_code(stock_code)
        queries.append(normalized_code)
        query_weights.append(0.9)
    
    if not queries:
        return []
    
    all_matches = []
    
    for query, weight in zip(queries, query_weights):
        try:
            search_results = vector_store.search(query, k=top_k)
            
            for doc, score in search_results:
                # Convert score to confidence
                confidence = 1.0 / (1.0 + score) if score > 0 else 1.0
                weighted_confidence = confidence * weight
                
                all_matches.append({
                    'doc': doc,
                    'score': score,
                    'confidence': weighted_confidence,
                    'match_type': 'semantic',
                    'query': query,
                    'weight': weight,
                })
        
        except Exception as e:
            logging.warning(f"Error searching for '{query}': {str(e)}")
            continue
    
    # Sort by confidence
    all_matches.sort(key=lambda x: x['confidence'], reverse=True)
    return all_matches


def optimized_search_strategy(
    stock_name: Optional[str],
    stock_code: Optional[str],
    vector_store: StockVectorStore,
    top_k: int = TOP_K_RESULTS,
) -> List[Dict[str, Any]]:
    """
    Optimized search with two-step verification approach.
    
    This is the default and recommended strategy.
    
    Strategy (User recommended approach):
    1. If stock code is provided:
       a. First search metadata with stock code to get exact matches
       b. Analyze if the stock name is similar
       c. If name is similar enough (>0.5), return that match with high confidence
       d. If not similar, perform name search for further verification
    2. If only name is provided, fall back to semantic search
    
    Returns:
        List of match dictionaries sorted by confidence
    """
    if stock_code:
        normalized_code = normalize_stock_code(stock_code)
        
        # STEP 1: Search metadata directly by stock code for exact matches
        logging.info(f"Step 1: Searching metadata for exact code match: {normalized_code}")
        metadata_results = vector_store.search_by_metadata(stock_code=normalized_code, k=top_k)
        
        if metadata_results:
            logging.info(f"Found {len(metadata_results)} exact code matches in metadata")
            
            # STEP 2: Analyze name similarity for each code match
            best_match = None
            best_similarity = 0.0
            
            for doc, score in metadata_results:
                # Extract name from document
                doc_name = None
                if hasattr(doc, 'metadata'):
                    doc_name = doc.metadata.get('stock_name_c') or doc.metadata.get('stock_name') or doc.metadata.get('AliasName')
                if not doc_name and hasattr(doc, 'page_content'):
                    _, doc_name = parse_stock_from_content(doc.page_content)
                
                if stock_name and doc_name:
                    # Calculate name similarity
                    name_similarity = calculate_name_similarity(stock_name, doc_name)
                    logging.info(f"  Code match: {normalized_code}, Name: {doc_name}, Similarity: {name_similarity:.2f}")
                    
                    if name_similarity > best_similarity:
                        best_similarity = name_similarity
                        best_match = (doc, score, name_similarity)
                else:
                    # No name to compare, but we have exact code match
                    logging.info(f"  Code match: {normalized_code} (no name comparison)")
                    if not best_match:
                        best_match = (doc, score, 0.5)  # Medium confidence without name verification
            
            # If we have a match with good name similarity, return it
            NAME_SIMILARITY_THRESHOLD = 0.5
            if best_match:
                doc, score, name_similarity = best_match
                
                if name_similarity >= NAME_SIMILARITY_THRESHOLD:
                    # High confidence: exact code + similar name
                    confidence = 0.95  # Very high confidence
                    logging.info(f"✓ Step 2: Name similarity {name_similarity:.2f} >= threshold {NAME_SIMILARITY_THRESHOLD}, returning match with confidence {confidence}")
                    
                    return [{
                        'doc': doc,
                        'score': score,
                        'confidence': confidence,
                        'match_type': 'exact_code_similar_name',
                        'query': f"code:{normalized_code}",
                        'weight': 1.0,
                        'name_similarity': name_similarity,
                    }]
                else:
                    # Names don't match well, proceed to Step 3
                    logging.info(f"Step 2: Name similarity {name_similarity:.2f} < threshold {NAME_SIMILARITY_THRESHOLD}, proceeding to name verification")
            else:
                logging.info(f"Step 2: No name comparison possible, proceeding to name verification")
        else:
            logging.info(f"Step 1: No exact code matches found in metadata")
        
        # STEP 3: Name verification - search by name to verify
        if stock_name:
            logging.info(f"Step 3: Searching by name for verification: '{stock_name}'")
            name_results = vector_store.search(stock_name, k=top_k * 3)
            
            # Check if any name search results have the same code
            for doc, score in name_results:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                doc_code = metadata.get('stock_code', '')
                
                if doc_code and codes_match(doc_code, normalized_code):
                    # Found a match via name search with same code!
                    confidence = 0.85  # High confidence: code match via name search
                    logging.info(f"✓ Step 3: Found code match {normalized_code} via name search, confidence {confidence}")
                    
                    return [{
                        'doc': doc,
                        'score': score,
                        'confidence': confidence,
                        'match_type': 'name_search_code_verified',
                        'query': stock_name,
                        'weight': 1.0,
                    }]
            
            # If we had metadata matches but name didn't verify, still return the best code match
            # but with lower confidence
            if metadata_results and best_match:
                doc, score, name_similarity = best_match
                confidence = 0.7  # Medium-high confidence: exact code but name doesn't match well
                logging.info(f"Step 3: Name search didn't verify, returning code match with reduced confidence {confidence}")
                
                return [{
                    'doc': doc,
                    'score': score,
                    'confidence': confidence,
                    'match_type': 'exact_code_unverified_name',
                    'query': f"code:{normalized_code}",
                    'weight': 1.0,
                    'name_similarity': name_similarity,
                }]
            
            logging.info(f"Step 3: Name search didn't find code match")
        
        # STEP 4: No metadata matches found, try broad semantic search with code
        logging.info(f"Step 4: Trying broad semantic search")
        semantic_results = vector_store.search(f"{stock_name or ''} {normalized_code}", k=top_k * 5)
        exact_match = exact_code_search(semantic_results, normalized_code)
        
        if exact_match:
            logging.info(f"✓ Step 4: Found code match via semantic search")
            exact_match['match_type'] = 'semantic_code_match'
            return [exact_match]
    
    # STEP 5: Fall back to pure semantic search (no code or code not found)
    logging.info("Step 5: Falling back to pure semantic search")
    return semantic_search_strategy(stock_name, stock_code, vector_store, top_k)


def exact_only_search_strategy(
    stock_name: Optional[str],
    stock_code: Optional[str],
    vector_store: StockVectorStore,
    top_k: int = TOP_K_RESULTS,
) -> List[Dict[str, Any]]:
    """
    Exact match only strategy.
    Only returns results if exact code or name match is found.
    
    Returns:
        List of match dictionaries (may be empty)
    """
    if not stock_code:
        return []
    
    normalized_code = normalize_stock_code(stock_code)
    code_search_results = vector_store.search(normalized_code, k=top_k * 2)
    
    exact_match = exact_code_search(code_search_results, normalized_code)
    
    if exact_match:
        return [exact_match]
    
    return []


# ============================================================================
# Core Verification and Correction Logic
# ============================================================================

def verify_and_correct_stock(
    stock_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    vector_store: Optional[StockVectorStore] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_LOW,
    top_k: int = TOP_K_RESULTS,
    strategy: SearchStrategy = SearchStrategy.OPTIMIZED,
) -> StockCorrectionResult:
    """
    Verify and correct stock name/code using vector store.
    
    Args:
        stock_name: Original stock name from STT/LLM
        stock_code: Original stock code from STT/LLM
        vector_store: Vector store instance (creates new if None)
        confidence_threshold: Minimum confidence to suggest correction
        top_k: Number of results to retrieve
        strategy: Search strategy to use (default: OPTIMIZED)
    
    Returns:
        StockCorrectionResult with correction details
    """
    # Initialize result
    result = StockCorrectionResult(
        original_stock_name=stock_name,
        original_stock_code=normalize_stock_code(stock_code) if stock_code else None,
        search_strategy=strategy.value,
    )
    
    # Get vector store
    if vector_store is None:
        vector_store = get_vector_store()
    
    if not vector_store.is_available:
        if not vector_store.initialize():
            result.reasoning = "Vector store not available"
            return result
    
    # Select search strategy
    if strategy == SearchStrategy.OPTIMIZED:
        all_matches = optimized_search_strategy(stock_name, stock_code, vector_store, top_k)
    elif strategy == SearchStrategy.SEMANTIC_ONLY:
        all_matches = semantic_search_strategy(stock_name, stock_code, vector_store, top_k)
    elif strategy == SearchStrategy.EXACT_ONLY:
        all_matches = exact_only_search_strategy(stock_name, stock_code, vector_store, top_k)
    else:
        result.reasoning = f"Unknown search strategy: {strategy}"
        return result
    
    if not all_matches:
        result.reasoning = "No matches found in vector store"
        return result
    
    # Store all candidates for debugging
    result.all_candidates = all_matches[:5]  # Keep top 5
    
    # Get best match
    best_match = all_matches[0]
    best_doc = best_match['doc']
    best_confidence = best_match['confidence']
    match_type = best_match.get('match_type', 'unknown')
    
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
    result.matched_content = content[:200]
    result.metadata = metadata
    
    # Check if there's a meaningful difference
    has_correction = False
    correction_details = []
    
    if corrected_name and corrected_name != stock_name:
        result.corrected_stock_name = corrected_name
        has_correction = True
        correction_details.append(f"名稱: {stock_name} → {corrected_name}")
    
    # Use codes_match to handle both padded and non-padded versions (e.g., '9992' == '09992')
    if corrected_code and not codes_match(corrected_code, result.original_stock_code):
        result.corrected_stock_code = corrected_code
        has_correction = True
        correction_details.append(f"代號: {result.original_stock_code or 'N/A'} → {corrected_code}")
    
    # Apply correction if confidence is high enough
    if has_correction and best_confidence >= confidence_threshold:
        result.correction_applied = True
        result.reasoning = f"{match_type} match ({best_confidence:.1%} confidence): {'; '.join(correction_details)}"
    elif has_correction:
        result.reasoning = f"Possible correction found but confidence too low ({best_confidence:.1%}): {'; '.join(correction_details)}"
    else:
        result.reasoning = f"Verified correct ({best_confidence:.1%} confidence, {match_type})"
    
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
        '升': ['星', '昇'],
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
    strategy: SearchStrategy = SearchStrategy.OPTIMIZED,
) -> List[StockCorrectionResult]:
    """
    Verify and correct multiple stocks in batch.
    
    Args:
        stocks: List of dicts with 'stock_name' and/or 'stock_code' keys
        vector_store: Vector store instance (creates new if None)
        confidence_threshold: Minimum confidence to suggest correction
        strategy: Search strategy to use
    
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
            strategy=strategy,
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
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("Stock Verifier (Improved) - Example Usage")
    print("=" * 80)
    
    # Test with the problematic case
    test_stocks = [
        {"stock_name": "騰訊升認購證", "stock_code": "18138"},
        {"stock_name": "騰訊", "stock_code": "700"},
        {"stock_name": "小米", "stock_code": "1810"},
    ]
    
    print("\n使用優化策略 (OPTIMIZED - 默認):")
    print("-" * 80)
    
    for stock in test_stocks:
        print(f"\n測試: {stock}")
        result = verify_and_correct_stock(
            stock_name=stock.get("stock_name"),
            stock_code=stock.get("stock_code"),
            strategy=SearchStrategy.OPTIMIZED,
        )
        
        print(format_correction_summary(result))
        print(f"Reasoning: {result.reasoning}")
        print(f"Strategy: {result.search_strategy}")

