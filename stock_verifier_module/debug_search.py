"""Debug script to understand why exact code match isn't working"""

import logging
import sys
import io
from stock_verifier_improved import get_vector_store, normalize_stock_code, extract_from_metadata, parse_stock_from_content

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO)

# Initialize vector store
vector_store = get_vector_store()
vector_store.initialize()

# Search for stock code 18138
stock_code = "18138"
normalized_code = normalize_stock_code(stock_code)

print(f"\nSearching for stock code: {stock_code} (normalized: {normalized_code})")
print("=" * 80)

# Get search results using code
print("\n1. Searching by CODE only:")
print("-" * 80)
search_results = vector_store.search(normalized_code, k=20)

found_exact = False
for i, (doc, score) in enumerate(search_results, 1):
    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
    
    # Extract code and name
    meta_code, meta_name = extract_from_metadata(metadata)
    content_code, content_name = parse_stock_from_content(content)
    
    final_code = meta_code or content_code
    final_name = meta_name or content_name
    
    # Check if this is an exact match
    if final_code == normalized_code:
        found_exact = True
        print(f"\n{i}. Score: {score:.4f} *** EXACT MATCH ***")
        print(f"   Code: {final_code}")
        print(f"   Name: {final_name}")
        print(f"   Content: {content}")
        break

if not found_exact:
    print("\nNo exact match found in top 20 results!")

# Now try searching with name
print("\n\n2. Searching by NAME ('騰訊升認購證'):")
print("-" * 80)
search_results2 = vector_store.search("騰訊升認購證", k=20)

for i, (doc, score) in enumerate(search_results2, 1):
    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
    
    meta_code, meta_name = extract_from_metadata(metadata)
    content_code, content_name = parse_stock_from_content(content)
    
    final_code = meta_code or content_code
    final_name = meta_name or content_name
    
    print(f"\n{i}. Score: {score:.4f}")
    print(f"   Code: {final_code}")
    print(f"   Name: {final_name}")
    
    if final_code == normalized_code:
        print(f"   *** FOUND 18138! ***")

# Try searching with "騰訊" only
print("\n\n3. Searching by '騰訊' and looking for warrants:")
print("-" * 80)
search_results3 = vector_store.search("騰訊", k=50)

count_18138 = 0
for i, (doc, score) in enumerate(search_results3, 1):
    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
    
    meta_code, meta_name = extract_from_metadata(metadata)
    content_code, content_name = parse_stock_from_content(content)
    
    final_code = meta_code or content_code
    final_name = meta_name or content_name
    
    if final_code == normalized_code:
        count_18138 += 1
        print(f"\n{i}. Score: {score:.4f} *** FOUND 18138! ***")
        print(f"   Code: {final_code}")
        print(f"   Name: {final_name}")

if count_18138 == 0:
    print(f"\nStock 18138 not found in top 50 '騰訊' results")

