from langchain.tools import tool
from dataclasses import dataclass, asdict
from typing import Literal, Dict
import time
import dspy
import os
import json
import pandas as pd
from difflib import SequenceMatcher

# Global variables for caching
stock_df = None
vector_store = None
embeddings = None


def initialize_stock_database():
    """Initialize the stock database from CSV file."""
    global stock_df
    
    if stock_df is not None:
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Stock database already initialized, skipping")
        return stock_df
    
    # Find the CSV file
    agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(agent_dir, "assets", "ListOfSecurities_c.csv")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Stock database file not found: {csv_file}")
    
    trace_start = time.time()
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Loading stock database from CSV...")
    # Skip metadata rows (0, 1, 3, 4, 5, 6) and use row 2 (0-indexed) as column names
    # Row 2 contains the actual column headers
    stock_df = pd.read_csv(csv_file, skiprows=[0, 1, 3, 4, 5, 6], encoding='utf-8')
    # Filter out any rows where stock code is not numeric (to remove any remaining metadata)
    code_col = None
    for col in ['股份代號', 'Stock Code', 'stock_code', 'Code', 'code', '股票代號', '股票代码']:
        if col in stock_df.columns:
            code_col = col
            break
    if code_col:
        # Keep only rows where stock code is numeric (starts with digits)
        stock_df = stock_df[stock_df[code_col].astype(str).str.strip().str.match(r'^\d+', na=False)]
    trace_elapsed = time.time() - trace_start
    print(f"[TRACE {time.strftime('%H:%M:%S')}] Stock database loaded in {trace_elapsed:.4f}s - {len(stock_df)} stocks")
    print(f"✅ Stock database initialized with {len(stock_df)} stocks")
    
    # Print column names for debugging
    print(f"[DEBUG] CSV columns: {stock_df.columns.tolist()}")
    
    return stock_df


def initialize_vector_store():
    """Initialize the Milvus vector store for fuzzy matching."""
    global vector_store, embeddings
    
    if vector_store is not None:
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Vector store already initialized, skipping")
        return vector_store
    
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_milvus import Milvus
        
        trace_start = time.time()
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Initializing vector store...")
        
        # Milvus configuration
        CLUSTER_ENDPOINT = "https://in03-5fb5696b79d8b56.serverless.aws-eu-central-1.cloud.zilliz.com"
        TOKEN = "9052d0067c0dd76fc12de51d2cc7a456dcd6caf58e72e344a2c372c85d6f7b486f39d1f2fd15916a7a9234127760e622c3145c36"
        
        embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
        
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name="phone_calls",
            connection_args={
                "uri": CLUSTER_ENDPOINT,
                "token": TOKEN,
                "db_name": "stocks"
            },
            consistency_level="Strong",
            drop_old=False,
        )
        
        trace_elapsed = time.time() - trace_start
        print(f"[TRACE {time.strftime('%H:%M:%S')}] Vector store initialized in {trace_elapsed:.4f}s")
        print(f"✅ Vector store initialized successfully")
        
        return vector_store
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize vector store: {str(e)}")
        print(f"   Will use CSV database only for verification")
        return None


def similarity_ratio(str1: str, str2: str) -> float:
    """Calculate similarity ratio between two strings."""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def verify_stock_with_excel(stock_name: str, stock_code: str, df: pd.DataFrame) -> Dict:
    """
    Verify stock information against the CSV database.
    
    Args:
        stock_name: Stock name to verify
        stock_code: Stock code to verify
        df: Stock database dataframe
        
    Returns:
        Dict with verification results and corrections
    """
    result = {
        'verified': False,
        'corrected': False,
        'confidence': 0,  # Confidence score 0-100
        'original_name': stock_name,
        'original_code': stock_code,
        'verified_name': None,
        'verified_code': None,
        'match_type': None,
        'candidates': []
    }
    
    # Normalize inputs
    stock_code_normalized = str(stock_code).strip() if stock_code else ""
    stock_name_normalized = str(stock_name).strip() if stock_name else ""
    
    # Try to identify the correct columns (common variations)
    code_columns = ['Stock Code', 'stock_code', 'Code', 'code', '股票代號', '股票代码']
    name_columns = ['Stock Name', 'stock_name', 'Name', 'name', '股票名稱', '股票名称', 'Stock Name(English)', 'Stock Name(Traditional Chinese)']
    
    code_col = None
    name_col = None
    
    for col in code_columns:
        if col in df.columns:
            code_col = col
            break
    
    for col in name_columns:
        if col in df.columns:
            name_col = col
            break
    
    if not code_col or not name_col:
        print(f"⚠️  Warning: Could not identify code/name columns in CSV. Columns: {df.columns.tolist()}")
        return result
    
    # Strategy 1: Exact code match
    if stock_code_normalized:
        exact_code_matches = df[df[code_col].astype(str).str.strip() == stock_code_normalized]
        if len(exact_code_matches) > 0:
            match = exact_code_matches.iloc[0]
            result['verified'] = True
            result['confidence'] = 100  # Exact match = 100% confidence
            result['verified_code'] = str(match[code_col]).strip()
            result['verified_name'] = str(match[name_col]).strip()
            result['match_type'] = 'exact_code'
            
            # Check if name also matches
            if stock_name_normalized and stock_name_normalized == result['verified_name']:
                result['corrected'] = False
            else:
                result['corrected'] = True
            
            return result
    
    # Strategy 2: Exact name match
    if stock_name_normalized:
        exact_name_matches = df[df[name_col].astype(str).str.strip() == stock_name_normalized]
        if len(exact_name_matches) > 0:
            match = exact_name_matches.iloc[0]
            result['verified'] = True
            result['confidence'] = 100  # Exact match = 100% confidence
            result['verified_code'] = str(match[code_col]).strip()
            result['verified_name'] = str(match[name_col]).strip()
            result['match_type'] = 'exact_name'
            
            # Check if code also matches
            if stock_code_normalized and stock_code_normalized == result['verified_code']:
                result['corrected'] = False
            else:
                result['corrected'] = True
            
            return result
    
    # Strategy 3: Fuzzy matching on name
    if stock_name_normalized:
        similarities = []
        for idx, row in df.iterrows():
            db_name = str(row[name_col]).strip()
            sim = similarity_ratio(stock_name_normalized, db_name)
            if sim > 0.6:  # Threshold for considering a match
                similarities.append({
                    'name': db_name,
                    'code': str(row[code_col]).strip(),
                    'similarity': sim
                })
        
        if similarities:
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            result['candidates'] = similarities[:5]  # Keep top 5 candidates
            
            # If best match is very good (>0.85), use it
            if similarities[0]['similarity'] > 0.85:
                result['verified'] = True
                result['corrected'] = True
                # Convert similarity (0.0-1.0) to confidence score (0-100)
                result['confidence'] = int(similarities[0]['similarity'] * 100)
                result['verified_name'] = similarities[0]['name']
                result['verified_code'] = similarities[0]['code']
                result['match_type'] = 'fuzzy_name'
                return result
    
    # Strategy 4: Fuzzy matching on code (with lower threshold)
    if stock_code_normalized:
        similarities = []
        for idx, row in df.iterrows():
            db_code = str(row[code_col]).strip()
            sim = similarity_ratio(stock_code_normalized, db_code)
            if sim > 0.7:  # Higher threshold for codes
                similarities.append({
                    'name': str(row[name_col]).strip(),
                    'code': db_code,
                    'similarity': sim
                })
        
        if similarities:
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Add to candidates if not already there
            for sim_match in similarities[:3]:
                if sim_match not in result['candidates']:
                    result['candidates'].append(sim_match)
            
            # If best match is very good (>0.9), use it
            if similarities[0]['similarity'] > 0.9:
                result['verified'] = True
                result['corrected'] = True
                # Convert similarity (0.0-1.0) to confidence score (0-100)
                result['confidence'] = int(similarities[0]['similarity'] * 100)
                result['verified_name'] = similarities[0]['name']
                result['verified_code'] = similarities[0]['code']
                result['match_type'] = 'fuzzy_code'
                return result
    
    return result


def verify_stock_with_vector_store(stock_name: str, stock_code: str, vs) -> Dict:
    """
    Verify stock information using Milvus vector store for semantic similarity.
    
    Args:
        stock_name: Stock name to verify
        stock_code: Stock code to verify
        vs: Vector store instance
        
    Returns:
        Dict with verification results
    """
    result = {
        'verified': False,
        'corrected': False,
        'confidence': 0,  # Confidence score 0-100
        'verified_name': None,
        'verified_code': None,
        'match_type': None,
        'candidates': []
    }
    
    if not vs:
        return result
    
    try:
        # Combine name and code for better semantic search
        query = f"{stock_name} {stock_code}".strip()
        if not query:
            return result
        
        # Search for similar stocks
        docs = vs.similarity_search_with_score(query, k=5)
        
        candidates = []
        for doc, score in docs:
            # Extract stock info from document
            metadata = doc.metadata
            candidates.append({
                'name': metadata.get('stock_name', ''),
                'code': metadata.get('stock_code', ''),
                'similarity': 1.0 - score,  # Convert distance to similarity
                'metadata': metadata
            })
        
        result['candidates'] = candidates
        
        # If best match has high confidence (>0.85), use it
        if candidates and candidates[0]['similarity'] > 0.85:
            result['verified'] = True
            result['corrected'] = True
            # Convert similarity (0.0-1.0) to confidence score (0-100)
            result['confidence'] = int(candidates[0]['similarity'] * 100)
            result['verified_name'] = candidates[0]['name']
            result['verified_code'] = candidates[0]['code']
            result['match_type'] = 'vector_semantic'
        
    except Exception as e:
        print(f"⚠️  Warning: Vector store search failed: {str(e)}")
    
    return result


@tool
def identify_stocks_in_conversation(file_path: str) -> str:
    """
    Analyze conversation text using dspy to identify stocks AND automatically verify them against the stock database.
    
    This integrated tool performs two operations in sequence:
    
    1. IDENTIFICATION: Extracts structured stock information from conversation text including:
       - Stock codes and names mentioned
       - Prices and quantities
       - Order types (ask/bid/none)
       - Confidence score (0-100) indicating how sure the model is about the extraction
    
    2. VERIFICATION: Validates and corrects the identified stocks using:
       - CSV database for exact and fuzzy matching
       - Milvus vector store for semantic similarity search
       - Multi-strategy verification (exact match, fuzzy match, semantic search)
       - Confidence scoring for each verification (0-100)
    
    The tool saves two JSON files to agent/output/stocks/:
    - [filename].json: Originally identified stocks
    - [filename]_verified.json: Verified and corrected stocks with confidence scores
    
    Args:
        file_path: Path to the file containing the conversation text to analyze
        
    Returns:
        A formatted string with both identification and verification results, including:
        - List of identified stocks
        - Verification summary (verified/corrected/unverified counts)
        - Detailed verification results for each stock
        - File paths to saved JSON files
    """
    # Read the conversation text from the file
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        
        # Try UTF-8 first, fall back to other encodings if needed
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5']
        conversation_text = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    conversation_text = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if conversation_text is None:
            return f"Error: Could not read file {file_path} with any supported encoding"
        
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"
    
    # dspy is configured in app.py to avoid threading issues
    # Just use the configured dspy instance here

    @dataclass
    class StockInfo():
        stock_name: str
        stock_number: str
        price: float
        quantity: int
        order_type: Literal["ask", "bid", "unknown"]
        confidence: int  # Confidence score from 0 to 100

    class ExtractInfo(dspy.Signature):
        """Extract structured information from text."""
        text: str = dspy.InputField()
        entities: list[StockInfo] = dspy.OutputField(desc="a list of stock name, stock number, price, quantity, order type (ask/bid/unknown), and confidence score (0-100 indicating how sure you are about the extraction)")

    try:
        module = dspy.Predict(ExtractInfo)
        response = module(text=conversation_text)
        
        # Format the output
        if not response.entities:
            return "No stocks identified in the conversation."
        
        # Convert dataclass objects to dictionaries for JSON serialization
        stocks_data = [asdict(stock) for stock in response.entities]
        
        # Save stocks to output/stocks/ directory
        try:
            # Determine the output directory - use agent/output/stocks/
            # current_dir is agent/tools/, so agent_dir is agent/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            agent_dir = os.path.dirname(current_dir)
            stocks_output_dir = os.path.join(agent_dir, "output", "stocks")
            
            # Create directory if it doesn't exist
            os.makedirs(stocks_output_dir, exist_ok=True)
            
            # Extract audio filename from the transcription file path
            # file_path is like: agent/output/transcriptions/[audio_filename]/transcriptions_text.txt
            # We want to get [audio_filename]
            file_path_normalized = os.path.normpath(file_path)
            path_parts = file_path_normalized.split(os.sep)
            
            # Find the transcription directory name (should be the audio filename)
            audio_filename = None
            if "transcriptions" in path_parts:
                trans_idx = path_parts.index("transcriptions")
                if trans_idx + 1 < len(path_parts):
                    audio_filename = path_parts[trans_idx + 1]
            
            # Fallback: use the parent directory name if we couldn't find it
            if not audio_filename:
                audio_filename = os.path.basename(os.path.dirname(file_path_normalized))
            
            # Create JSON filename based on audio filename
            json_filename = f"{audio_filename}.json"
            json_filepath = os.path.join(stocks_output_dir, json_filename)
            
            # Prepare data structure for saving
            stocks_output = {
                "source_file": file_path,
                "audio_filename": audio_filename,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "stocks": stocks_data
            }
            
            # Save stocks as JSON
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(stocks_output, f, indent=2, ensure_ascii=False)
            
            # Format the output string
            output = f"\n{'='*80}\n"
            output += f"📊 STOCK IDENTIFICATION RESULTS\n"
            output += f"{'='*80}\n\n"
            output += f"Identified {len(response.entities)} stock(s) in the conversation:\n\n"
            for i, stock in enumerate(response.entities, 1):
                output += f"Stock {i}:\n"
                output += f"  Name: {stock.stock_name}\n"
                output += f"  Code: {stock.stock_number}\n"
                output += f"  Price: {stock.price}\n"
                output += f"  Quantity: {stock.quantity}\n"
                output += f"  Order Type: {stock.order_type}\n"
                output += f"  Confidence: {stock.confidence}/100\n\n"
            
            output += f"💾 Identified stocks saved to: {json_filepath}\n\n"
            
            # === VERIFICATION STEP ===
            print(f"\n{'='*80}")
            print(f"🔍 Starting stock verification...")
            print(f"{'='*80}\n")
            
            # Initialize verification databases
            df = initialize_stock_database()
            vs = initialize_vector_store()
            
            # Verify each stock
            verified_stocks = []
            verification_summary = []
            
            for i, stock in enumerate(response.entities, 1):
                stock_name = stock.stock_name
                stock_code = stock.stock_number
                
                print(f"📊 Verifying stock {i}: {stock_name} ({stock_code})")
                
                # Try CSV database first
                excel_result = verify_stock_with_excel(stock_name, stock_code, df)
                
                # If CSV gives us a confident match, use it (confidence >= 85 out of 100)
                if excel_result['verified'] and excel_result['confidence'] >= 85:
                    verified_stock = asdict(stock)
                    verified_stock['verification'] = excel_result
                    verified_stocks.append(verified_stock)
                    
                    if excel_result['corrected']:
                        print(f"   ✅ Corrected: {excel_result['verified_name']} ({excel_result['verified_code']}) - Confidence: {excel_result['confidence']}/100")
                    else:
                        print(f"   ✅ Verified: {excel_result['verified_name']} ({excel_result['verified_code']}) - Confidence: {excel_result['confidence']}/100")
                    
                    verification_summary.append({
                        'stock_num': i,
                        'original_name': stock_name,
                        'original_code': stock_code,
                        'verified_name': excel_result['verified_name'],
                        'verified_code': excel_result['verified_code'],
                        'corrected': excel_result['corrected'],
                        'confidence': excel_result['confidence'],
                        'match_type': excel_result['match_type']
                    })
                    continue
                
                # If CSV doesn't give confident match, try vector store
                if vs:
                    vector_result = verify_stock_with_vector_store(stock_name, stock_code, vs)
                    
                    # Compare results and use the one with higher confidence
                    if vector_result['verified'] and vector_result['confidence'] > excel_result.get('confidence', 0):
                        verified_stock = asdict(stock)
                        verified_stock['verification'] = vector_result
                        verified_stocks.append(verified_stock)
                        
                        print(f"   ✅ Corrected (vector): {vector_result['verified_name']} ({vector_result['verified_code']}) - Confidence: {vector_result['confidence']}/100")
                        
                        verification_summary.append({
                            'stock_num': i,
                            'original_name': stock_name,
                            'original_code': stock_code,
                            'verified_name': vector_result['verified_name'],
                            'verified_code': vector_result['verified_code'],
                            'corrected': vector_result['corrected'],
                            'confidence': vector_result['confidence'],
                            'match_type': vector_result['match_type']
                        })
                        continue
                
                # If neither method gave confident result, keep original but include candidates
                verified_stock = asdict(stock)
                verified_stock['verification'] = excel_result
                verified_stock['verification']['verified'] = False
                verified_stocks.append(verified_stock)
                
                print(f"   ⚠️  Could not verify with high confidence")
                if excel_result['candidates']:
                    print(f"   📋 Candidates:")
                    for j, candidate in enumerate(excel_result['candidates'][:3], 1):
                        candidate_confidence = int(candidate['similarity'] * 100)
                        print(f"      {j}. {candidate['name']} ({candidate['code']}) - Confidence: {candidate_confidence}/100")
                
                verification_summary.append({
                    'stock_num': i,
                    'original_name': stock_name,
                    'original_code': stock_code,
                    'verified_name': None,
                    'verified_code': None,
                    'corrected': False,
                    'confidence': 0,
                    'match_type': 'unverified',
                    'candidates': excel_result['candidates'][:3]
                })
            
            # Save verified stocks to new file
            verified_filename = f"{audio_filename}_verified.json"
            verified_filepath = os.path.join(stocks_output_dir, verified_filename)
            
            verified_data = {
                'source_file': file_path,
                'audio_filename': audio_filename,
                'original_timestamp': stocks_output['timestamp'],
                'verification_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stocks': verified_stocks
            }
            
            with open(verified_filepath, 'w', encoding='utf-8') as f:
                json.dump(verified_data, f, indent=2, ensure_ascii=False)
            
            # Add verification results to output
            output += f"\n{'='*80}\n"
            output += f"✅ STOCK VERIFICATION COMPLETE\n"
            output += f"{'='*80}\n\n"
            
            verified_count = sum(1 for s in verification_summary if s['verified_name'] is not None)
            corrected_count = sum(1 for s in verification_summary if s['corrected'])
            unverified_count = len(verification_summary) - verified_count
            
            output += f"📊 Verification Summary:\n"
            output += f"   Total stocks: {len(response.entities)}\n"
            output += f"   ✅ Verified: {verified_count}\n"
            output += f"   🔧 Corrected: {corrected_count}\n"
            output += f"   ⚠️  Unverified: {unverified_count}\n\n"
            output += f"💾 Verified stocks saved to: {verified_filepath}\n\n"
            
            if verification_summary:
                output += "📋 Verification Details:\n"
                output += "=" * 80 + "\n\n"
                
                for item in verification_summary:
                    output += f"📊 Stock {item['stock_num']}:\n"
                    output += f"   Original: {item['original_name']} ({item['original_code']})\n"
                    
                    if item['verified_name']:
                        if item['corrected']:
                            output += f"   ✅ Corrected to: {item['verified_name']} ({item['verified_code']})\n"
                        else:
                            output += f"   ✅ Verified: {item['verified_name']} ({item['verified_code']})\n"
                        output += f"   📈 Confidence: {item['confidence']}/100\n"
                        output += f"   🔍 Match Type: {item['match_type']}\n"
                    else:
                        output += f"   ⚠️  Could not verify with high confidence\n"
                        if item.get('candidates'):
                            output += f"   📋 Top candidates:\n"
                            for j, candidate in enumerate(item['candidates'][:3], 1):
                                candidate_confidence = int(candidate['similarity'] * 100)
                                output += f"      {j}. {candidate['name']} ({candidate['code']}) - Confidence: {candidate_confidence}/100\n"
                    
                    output += "-" * 80 + "\n\n"
            
            # Add final instruction
            output += f"\n{'='*80}\n"
            output += "✅ Stock identification and verification complete.\n"
            output += f"   📄 Identified stocks file: {json_filepath}\n"
            output += f"   ✅ Verified stocks file: {verified_filepath}\n"
            output += f"{'='*80}\n"
            
            return output
        except Exception as e:
            # If saving or verification fails, log but don't fail the tool
            import traceback
            output = f"Identified {len(response.entities)} stock(s) in the conversation:\n\n"
            for i, stock in enumerate(response.entities, 1):
                output += f"Stock {i}:\n"
                output += f"  Name: {stock.stock_name}\n"
                output += f"  Code: {stock.stock_number}\n"
                output += f"  Price: {stock.price}\n"
                output += f"  Quantity: {stock.quantity}\n"
                output += f"  Order Type: {stock.order_type}\n"
                output += f"  Confidence: {stock.confidence}/100\n\n"
            
            output += f"\n⚠️  Warning: Could not complete stock verification: {str(e)}\n"
            output += f"Traceback:\n{traceback.format_exc()}"
            return output
        
    except Exception as e:
        import traceback
        return f"❌ Error analyzing conversation: {str(e)}\nTraceback:\n{traceback.format_exc()}"