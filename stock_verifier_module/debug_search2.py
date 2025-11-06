"""Search for the correct stock name to see if it exists"""

import sys
import io
from stock_verifier_improved import get_vector_store, normalize_stock_code, extract_from_metadata

# Set UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Initialize
vector_store = get_vector_store()
vector_store.initialize()

# Search for the correct name
correct_name = "騰訊摩通六一購Ｂ"
print(f"\nSearching for correct name: '{correct_name}'")
print("=" * 80)

results = vector_store.search(correct_name, k=10)

print(f"\nFound {len(results)} results:\n")

for i, (doc, score) in enumerate(results, 1):
    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
    
    code, name = extract_from_metadata(metadata)
    
    print(f"{i}. Score: {score:.4f}")
    print(f"   Code: {code}")
    print(f"   Name: {name}")
    print(f"   Content: {content}")
    
    if code == "18138":
        print(f"   *** THIS IS STOCK 18138! ***")
    print()




