"""
Tab: Transaction Analysis with JSON Stock Reference
Analyze two transcriptions to identify stock transactions using merged JSON stock data
"""

import json
import traceback
from typing import Literal, Optional
import gradio as gr

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from opencc import OpenCC

# Import centralized model configuration
from model_config import MODEL_OPTIONS, DEFAULT_MODEL, DEFAULT_OLLAMA_URL

# Import from the stock comparison tab for conversation analysis
from tabs.tab_stt_stock_comparison import (
    extract_stocks_with_single_llm,
    DEFAULT_SYSTEM_MESSAGE as STOCK_EXTRACTION_SYSTEM_MESSAGE,
    LLM_OPTIONS,
)


# ============================================================================
# OpenCC Translation Setup
# ============================================================================

# Initialize OpenCC converter (Simplified to Traditional Chinese)
opencc_converter = OpenCC('s2t')  # s2t = Simplified to Traditional

def translate_to_traditional_chinese(text: str) -> str:
    """
    Convert Simplified Chinese text to Traditional Chinese using OpenCC.
    
    Args:
        text: Input text (may contain Simplified Chinese)
        
    Returns:
        str: Text with Simplified Chinese converted to Traditional Chinese
    """
    if not text or not text.strip():
        return text
    
    try:
        return opencc_converter.convert(text)
    except Exception as e:
        print(f"OpenCC translation failed: {e}")
        return text  # Return original text if translation fails


# ============================================================================
# Pydantic Models
# ============================================================================

# Pydantic models for structured transaction output
class Transaction(BaseModel):
    """Represents a single transaction"""
    
    transaction_type: Literal["buy", "sell", "queue"] = Field(
        description="The type of transaction identified: buy, sell, or queue"
    )
    
    confidence_score: float = Field(
        ge=0.0, 
        le=1.0,
        description="Confidence score from 0 to 1. 0=not sure at all, 0.5=moderately confident, 1.0=very confident"
    )
    
    conversation_number: Optional[int] = Field(
        default=None,
        description="The conversation number this transaction came from"
    )
    
    hkt_datetime: Optional[str] = Field(
        default=None,
        description="The Hong Kong datetime when the conversation/transaction occurred"
    )
    
    broker_id: Optional[str] = Field(
        default=None,
        description="The broker ID from the conversation metadata"
    )
    
    broker_name: Optional[str] = Field(
        default=None,
        description="The broker name from the conversation metadata"
    )
    
    client_id: Optional[str] = Field(
        default=None,
        description="The client ID from the conversation metadata"
    )
    
    client_name: Optional[str] = Field(
        default=None,
        description="The client name from the conversation metadata"
    )
    
    stock_code: Optional[str] = Field(
        default=None,
        description="The stock code/number identified in the conversation"
    )
    
    stock_name: Optional[str] = Field(
        default=None,
        description="The stock name identified in the conversation"
    )
    
    quantity: Optional[str] = Field(
        default=None,
        description="The quantity/amount of stocks in the transaction"
    )
    
    price: Optional[str] = Field(
        default=None,
        description="The price mentioned in the transaction"
    )
    
    explanation: str = Field(
        description="Detailed explanation of why this transaction type and confidence score were assigned"
    )


class TransactionAnalysisResult(BaseModel):
    """Complete analysis result with multiple transactions"""
    
    transactions: list[Transaction] = Field(
        default_factory=list,
        description="List of all transactions identified in the conversation. Empty list if no transactions found."
    )
    
    transcription_comparison: str = Field(
        description="Comparison of the two transcriptions and how they differ"
    )
    
    overall_summary: str = Field(
        description="Overall summary of the conversation and all transactions identified"
    )


def process_conversation_json_to_merged(
    conversation_json_input: str,
    selected_llms: list[str],
    ollama_url: str,
    temperature: float,
    use_vector_correction: bool = True,
) -> tuple[str, str]:
    """
    Process conversation JSON input and convert to merged JSON format
    
    Args:
        conversation_json_input: JSON string with conversation data
        selected_llms: List of LLM models to use for stock extraction
        ollama_url: Ollama server URL
        temperature: Temperature parameter
        use_vector_correction: Whether to use vector store correction
        
    Returns:
        tuple: (status_message, merged_json_output)
    """
    try:
        # Parse conversation JSON
        try:
            conversations = json.loads(conversation_json_input)
        except json.JSONDecodeError as e:
            return (f"❌ 錯誤：無效的JSON格式\n\n{str(e)}", "")
        
        # Validate it's a list
        if not isinstance(conversations, list):
            return ("❌ 錯誤：JSON必須是對話對象的數組", "")
        
        if len(conversations) == 0:
            return ("❌ 錯誤：JSON數組為空", "")
        
        # Collect all stocks from all conversations
        all_stocks = []
        status_parts = []
        
        status_parts.append(f"🔄 處理 {len(conversations)} 個對話...")
        status_parts.append(f"📊 使用 {len(selected_llms)} 個LLM模型: {', '.join(selected_llms)}")
        status_parts.append("")
        
        for conv_idx, conversation in enumerate(conversations, 1):
            conv_number = conversation.get("conversation_number", conv_idx)
            transcriptions = conversation.get("transcriptions", {})
            
            # Get transcription text
            transcription_text = None
            transcription_source = None
            
            if isinstance(transcriptions, dict):
                for source_name, text in transcriptions.items():
                    if text and text.strip():
                        transcription_text = text
                        transcription_source = source_name
                        break
            elif isinstance(transcriptions, str):
                transcription_text = transcriptions
                transcription_source = "default"
            
            if not transcription_text or not transcription_text.strip():
                status_parts.append(f"⚠️ 跳過對話 #{conv_number} - 無轉錄文字")
                continue
            
            status_parts.append(f"📞 處理對話 #{conv_number}...")
            
            # Extract stocks using each LLM
            for llm_model in selected_llms:
                result_model, formatted_result, raw_json = extract_stocks_with_single_llm(
                    model=llm_model,
                    conversation_text=transcription_text,
                    system_message=STOCK_EXTRACTION_SYSTEM_MESSAGE,
                    ollama_url=ollama_url,
                    temperature=temperature,
                    stt_source=transcription_source,
                    use_vector_correction=use_vector_correction
                )
                
                # Parse and collect stocks
                if raw_json and raw_json.strip():
                    try:
                        parsed = json.loads(raw_json)
                        stocks = parsed.get("stocks", [])
                        for stock in stocks:
                            stock["llm_model"] = llm_model
                            stock["conversation_number"] = conv_number
                        all_stocks.extend(stocks)
                        status_parts.append(f"  ✓ {llm_model}: 找到 {len(stocks)} 個股票")
                    except json.JSONDecodeError:
                        status_parts.append(f"  ⚠️ {llm_model}: 無法解析輸出")
            
            status_parts.append("")
        
        # Merge and deduplicate stocks (similar to create_merged_stocks_json)
        stocks_dict = {}
        total_analyses = len(conversations) * len(selected_llms)
        
        for stock in all_stocks:
            stock_number = stock.get("stock_number", "")
            if stock_number:
                if stock_number not in stocks_dict:
                    stocks_dict[stock_number] = []
                stocks_dict[stock_number].append(stock)
        
        # Create merged stocks list
        merged_stocks = []
        for stock_number, stock_list in stocks_dict.items():
            if not stock_list:
                continue
            
            # Calculate average relevance_score
            relevance_scores = [s.get("relevance_score", 0) for s in stock_list]
            total_score = sum(relevance_scores)
            avg_relevance_score = total_score / total_analyses if total_analyses > 0 else 0
            
            # Use first stock's data as base
            merged_stock = {
                "stock_number": stock_number,
                "stock_name": stock_list[0].get("stock_name", ""),
                "relevance_score": round(avg_relevance_score, 2),
            }
            
            # Include original_word if present
            original_words = [s.get("original_word", "") for s in stock_list if s.get("original_word")]
            if original_words:
                from collections import Counter
                word_counts = Counter(original_words)
                merged_stock["original_word"] = word_counts.most_common(1)[0][0]
            
            # Include quantity and price if present
            quantities = [s.get("quantity", "") for s in stock_list if s.get("quantity")]
            if quantities:
                from collections import Counter
                qty_counts = Counter(quantities)
                merged_stock["quantity"] = qty_counts.most_common(1)[0][0]
            
            prices = [s.get("price", "") for s in stock_list if s.get("price")]
            if prices:
                from collections import Counter
                price_counts = Counter(prices)
                merged_stock["price"] = price_counts.most_common(1)[0][0]
            
            # Include corrected stock information
            corrected_names = [s.get("corrected_stock_name") for s in stock_list if s.get("corrected_stock_name")]
            corrected_numbers = [s.get("corrected_stock_number") for s in stock_list if s.get("corrected_stock_number")]
            correction_confidences = [s.get("correction_confidence") for s in stock_list if s.get("correction_confidence")]
            
            merged_stock["corrected_stock_number"] = corrected_numbers[0] if corrected_numbers else stock_number
            merged_stock["corrected_stock_name"] = corrected_names[0] if corrected_names else stock_list[0].get("stock_name", "")
            merged_stock["correction_confidence"] = correction_confidences[0] if correction_confidences else 1.0
            
            # Confidence
            confidences = [s.get("confidence", "low").lower() for s in stock_list]
            confidence_priority = {"high": 3, "medium": 2, "low": 1}
            most_confident = max(confidences, key=lambda c: (confidences.count(c), confidence_priority.get(c, 0)))
            merged_stock["confidence"] = most_confident
            
            # Detection count
            merged_stock["detection_count"] = len(stock_list)
            
            # Track which LLM models detected this stock
            llm_models = [s.get("llm_model", "") for s in stock_list if s.get("llm_model")]
            if llm_models:
                unique_models = list(dict.fromkeys(llm_models))
                merged_stock["detected_by_llms"] = unique_models
            
            merged_stocks.append(merged_stock)
        
        # Sort by relevance_score
        merged_stocks.sort(key=lambda s: (-s["relevance_score"], s["stock_number"]))
        
        # Create merged data
        merged_data = {
            "stocks": merged_stocks,
            "metadata": {
                "total_conversations": len(conversations),
                "total_analyses": total_analyses,
                "unique_stocks_found": len(merged_stocks),
                "note": "從對話JSON自動提取和合併的股票數據"
            }
        }
        
        merged_json = json.dumps(merged_data, indent=2, ensure_ascii=False)
        
        status_parts.append(f"✅ 完成！找到 {len(merged_stocks)} 個唯一股票")
        status_message = "\n".join(status_parts)
        
        return (status_message, merged_json)
        
    except Exception as e:
        error_msg = f"❌ 錯誤: {str(e)}\n\n{traceback.format_exc()}"
        return (error_msg, "")


DEFAULT_SYSTEM_MESSAGE = """你是一位精通粵語的香港股市分析師，專門分析對話轉錄並識別潛在的股票交易。

你的任務是：
1. **核心任務：仔細分析對話轉錄內容（主要資料來源）**
2. **參考股票參考資料（次要資料來源）** - 注意：此資料可能不準確，需謹慎使用
3. 識別可能的股票交易類型（買入buy、賣出sell、排隊queue）
4. 為每個潛在交易評估置信度（0-1分）：
   - **0.0分：完全不確定** - 只是提及、沒有明確交易意圖
   - **0.5分：有一定證據但不完全確定** - 有交易跡象但證據不足
   - **1.0分：非常確定有交易發生** - 多項證據支持交易發生
5. 提取每個交易的細節（股票代號、股票名稱、數量、價格等）

# 分析方法
- **主要依據：對話轉錄內容**
  * 直接閱讀對話內容，理解上下文
  * 識別交易意圖的關鍵詞（買入、賣出、排隊等）
  * 提取明確提到的股票代號、名稱、數量、價格
  * 理解對話的語境和真實意圖

- **次要參考：股票參考資料（可能不準確，需謹慎使用）**
  * stock_number / stock_name: 從STT識別出的股票代號和名稱
  * corrected_stock_number / corrected_stock_name: 修正後的代號和名稱
  * original_word: STT轉錄的原始文字
  * relevance_score: 股票在對話中的相關度分數（0-1）
  * detection_count: 該股票被檢測到的次數
  * detected_by_llms: 檢測到該股票的LLM模型列表
  * confidence: 檢測置信度（high/medium/low）
  * **注意：這些信息僅供參考，優先相信對話內容本身**

# 判斷準則
- **首先**：仔細閱讀對話內容，理解真實意圖
- **然後**：參考股票參考資料作為輔助，但不要完全依賴
- 如果對話內容與參考資料衝突，優先相信對話內容
- 交叉驗證：對話內容 + 參考資料的元數據 → 提高準確性
- 謹慎判斷：參考資料中的高分數不一定代表有交易

# 粵語術語和簡稱
- 轮 = 窩輪
- 沽/孤 = 賣出
- 買入/入 = 買入
- 排隊 = 掛單等待成交

# 輸出格式
**必須**返回有效的JSON格式，嚴格遵守以下結構：

{
  "transactions": [
    {
      "transaction_type": "buy",  // 必須是 "buy", "sell", 或 "queue"
      "confidence_score": 0.85,   // 必須是數字 0.0-1.0，不能是字符串
      "conversation_number": 1,   // 必須是整數，表示該交易來自哪個對話
      "hkt_datetime": "2025-10-20T10:15:30",  // 對話的日期時間（系統自動從元數據提取）
      "broker_id": "B001",        // 經紀ID（系統自動從元數據提取）
      "broker_name": "Dickson Lau",  // 經紀姓名（系統自動從元數據提取）
      "client_id": "C123",        // 客戶ID（系統自動從元數據提取）
      "client_name": "CHENG SUK HING",  // 客戶姓名（系統自動從元數據提取）
      "stock_code": "0700",
      "stock_name": "騰訊控股",
      "quantity": "N/A",          // 如果從數據中無法確定
      "price": "N/A",             // 如果從數據中無法確定
      "explanation": "根據對話內容，客戶明確提到'買入騰訊100手'，交易意圖清晰。參考資料顯示相關度0.85，檢測次數3次，進一步確認。綜合判斷為買入交易，置信度0.85分"
    }
  ],
  "conversation_analysis": "對話內容的詳細分析...",
  "overall_summary": "整體摘要...（基於對話內容，參考資料僅作輔助）"
}

**重要提示：**
- confidence_score 必須是數字（float），不能是字符串
- conversation_number 必須是整數（int），表示該交易來自哪個對話
- hkt_datetime 會自動從對話元數據中提取並添加（系統會自動處理）
- broker_id, broker_name, client_id, client_name 會自動從對話元數據中提取並添加（系統會自動處理）
- explanation 字段必須詳細說明判斷依據，**優先引用對話內容**，然後才是參考資料
- conversation_analysis 必須詳細分析對話內容
- overall_summary 必須基於對話內容為主，參考資料為輔
"""


def analyze_transactions_with_json(
    conversation_json_input: str,
    merged_json_input: str,
    model: str,
    ollama_url: str,
    system_message: str,
    temperature: float,
) -> tuple[str, str]:
    """
    Analyze conversation JSON with merged JSON stock data as reference to identify potential transactions
    
    Args:
        conversation_json_input: JSON string with conversation data (primary source)
        merged_json_input: JSON string with merged/deduplicated stock data (reference only, may not be accurate)
        model: LLM model name
        ollama_url: Ollama server URL
        system_message: System prompt for the LLM
        temperature: Temperature parameter
    
    Returns:
        tuple: (summary_result, json_result)
    """
    try:
        # Validate inputs
        if not conversation_json_input or not conversation_json_input.strip():
            error_msg = "❌ 錯誤：請提供對話JSON數據"
            return (error_msg, "")
        
        if not model or not model.strip():
            error_msg = "❌ 錯誤：請指定模型名稱"
            return (error_msg, "")
        
        if not ollama_url or not ollama_url.strip():
            error_msg = "❌ 錯誤：請指定 Ollama URL"
            return (error_msg, "")
        
        # Parse conversation JSON to extract conversation text and metadata mapping
        conversation_text = ""
        conversation_info = ""
        conversation_datetime_map = {}  # Map conversation_number -> hkt_datetime
        conversation_broker_id_map = {}  # Map conversation_number -> broker_id
        conversation_broker_name_map = {}  # Map conversation_number -> broker_name
        conversation_client_id_map = {}  # Map conversation_number -> client_id
        conversation_client_name_map = {}  # Map conversation_number -> client_name
        
        try:
            conversations = json.loads(conversation_json_input)
            if not isinstance(conversations, list):
                conversations = [conversations]
            
            conversation_parts = []
            for idx, conv in enumerate(conversations, 1):
                conv_number = conv.get("conversation_number", idx)
                transcriptions = conv.get("transcriptions", {})
                metadata = conv.get("metadata", {})
                
                # Extract metadata fields
                hkt_datetime = metadata.get("hkt_datetime", "N/A")
                broker_id = metadata.get("broker_id", "N/A")
                broker_name = metadata.get("broker_name", "N/A")
                client_id = metadata.get("client_id", "N/A")
                client_name = metadata.get("client_name", "N/A")
                
                conversation_datetime_map[conv_number] = hkt_datetime
                conversation_broker_id_map[conv_number] = broker_id
                conversation_broker_name_map[conv_number] = broker_name
                conversation_client_id_map[conv_number] = client_id
                conversation_client_name_map[conv_number] = client_name
                
                # Extract transcription text
                transcription_text = ""
                if isinstance(transcriptions, dict):
                    for source_name, text in transcriptions.items():
                        if text and text.strip():
                            transcription_text += f"\n[來源: {source_name}]\n{text}\n"
                elif isinstance(transcriptions, str):
                    transcription_text = transcriptions
                
                # Format conversation with datetime
                conv_part = f"\n--- 對話 #{conv_number} ---\n"
                conv_part += f"日期時間: {hkt_datetime}\n"
                if metadata:
                    conv_part += f"元數據: {json.dumps(metadata, ensure_ascii=False)}\n"
                conv_part += f"內容:\n{transcription_text}\n"
                conversation_parts.append(conv_part)
            
            conversation_text = "\n".join(conversation_parts)
            conversation_info = f"共 {len(conversations)} 個對話"
            
        except json.JSONDecodeError as e:
            error_msg = f"❌ 錯誤：無法解析對話JSON格式\n\n{str(e)}"
            return (error_msg, "")
        
        # Parse merged JSON to extract stock information (as reference only)
        stock_ref_text = "（無提供）"
        stock_list_for_checking = []
        metadata_info = ""
        
        if merged_json_input and merged_json_input.strip():
            try:
                merged_data = json.loads(merged_json_input)
                stocks = merged_data.get("stocks", [])
                metadata = merged_data.get("metadata", {})
                
                # Format metadata if available
                if metadata:
                    metadata_parts = []
                    if "total_conversations" in metadata:
                        metadata_parts.append(f"總對話數：{metadata['total_conversations']}")
                    if "total_analyses" in metadata:
                        metadata_parts.append(f"總分析數：{metadata['total_analyses']}")
                    if "unique_stocks_found" in metadata:
                        metadata_parts.append(f"唯一股票數：{metadata['unique_stocks_found']}")
                    
                    if metadata_parts:
                        metadata_info = f"\n[數據來源：{' | '.join(metadata_parts)}]\n"
                
                if stocks:
                    stock_lines = []
                    for idx, stock in enumerate(stocks, 1):
                        # Extract stock information
                        stock_number = stock.get("stock_number", "")
                        stock_name = stock.get("stock_name", "")
                        corrected_number = stock.get("corrected_stock_number", "")
                        corrected_name = stock.get("corrected_stock_name", "")
                        original_word = stock.get("original_word", "")
                        quantity = stock.get("quantity", "")
                        price = stock.get("price", "")
                        relevance = stock.get("relevance_score", 0)
                        confidence = stock.get("confidence", "")
                        detection_count = stock.get("detection_count", 0)
                        detected_by = stock.get("detected_by_llms", [])
                        
                        # Store for explicit checking instruction
                        stock_info = {
                            "original_number": stock_number,
                            "original_name": stock_name,
                            "corrected_number": corrected_number,
                            "corrected_name": corrected_name,
                            "original_word": original_word,
                            "quantity": quantity,
                            "price": price,
                            "relevance": relevance,
                            "confidence": confidence,
                            "detection_count": detection_count,
                            "detected_by": detected_by
                        }
                        stock_list_for_checking.append(stock_info)
                        
                        # Format for display - show comprehensive information
                        line_parts = [f"{idx}."]
                        
                        if stock_number:
                            line_parts.append(f"股票代號：{stock_number}")
                        if stock_name:
                            line_parts.append(f"股票名稱：{stock_name}")
                        
                        # Show corrected versions if different
                        if corrected_number and corrected_number != stock_number:
                            line_parts.append(f"[修正代號：{corrected_number}]")
                        if corrected_name and corrected_name != stock_name:
                            line_parts.append(f"[修正名稱：{corrected_name}]")
                        
                        # Show original word from STT if available
                        if original_word:
                            line_parts.append(f"(原文：{original_word})")
                        
                        # Show quantity and price if available
                        if quantity:
                            line_parts.append(f"(數量：{quantity})")
                        if price:
                            line_parts.append(f"(價格：{price})")
                        
                        # Show relevance score
                        if relevance:
                            line_parts.append(f"(相關度：{relevance})")
                        
                        # Show confidence
                        if confidence:
                            line_parts.append(f"(置信度：{confidence})")
                        
                        # Show detection count and models
                        if detection_count:
                            line_parts.append(f"(檢測次數：{detection_count})")
                        if detected_by:
                            models_str = ", ".join(detected_by)
                            line_parts.append(f"(檢測模型：{models_str})")
                        
                        stock_lines.append("  ".join(line_parts))
                    
                    if stock_lines:
                        stock_ref_text = metadata_info + "\n".join(stock_lines)
            except json.JSONDecodeError as e:
                # If merged JSON is invalid, just ignore it
                stock_ref_text = "（無法解析參考資料）"
        
        # Build reference note if stock data is available
        reference_note = ""
        if stock_list_for_checking:
            reference_note = f"\n\n**注意：以下股票參考資料僅供參考，可能不準確，請優先分析對話內容：**\n"
            for idx, stock in enumerate(stock_list_for_checking, 1):
                orig_number = stock.get("original_number", "")
                orig_name = stock.get("original_name", "")
                corr_number = stock.get("corrected_number", "")
                corr_name = stock.get("corrected_name", "")
                orig_word = stock.get("original_word", "")
                quantity = stock.get("quantity", "")
                price = stock.get("price", "")
                relevance = stock.get("relevance", 0)
                detection_count = stock.get("detection_count", 0)
                confidence_level = stock.get("confidence", "")
                
                # Build reference item
                ref_items = []
                
                # Stock identification
                if orig_name or orig_number:
                    ref_items.append(f"股票：{orig_name or ''} ({orig_number or ''})")
                
                # Metadata
                metadata_parts = []
                if relevance:
                    metadata_parts.append(f"相關度={relevance}")
                if detection_count:
                    metadata_parts.append(f"檢測次數={detection_count}")
                if confidence_level:
                    metadata_parts.append(f"置信度={confidence_level}")
                if quantity:
                    metadata_parts.append(f"數量={quantity}")
                if price:
                    metadata_parts.append(f"價格={price}")
                if orig_word:
                    metadata_parts.append(f"原文={orig_word}")
                
                if metadata_parts:
                    ref_items.append(f"[{', '.join(metadata_parts)}]")
                
                if ref_items:
                    reference_note += f"{idx}. {' '.join(ref_items)}\n"
        
        prompt = f"""請仔細分析以下對話轉錄，識別潛在的股票交易。

## 📞 對話轉錄內容（主要資料來源 - 請優先分析）：
{conversation_text}

{conversation_info}

## 📊 股票參考資料（次要資料來源 - 僅供參考，可能不準確）：
{stock_ref_text}
{reference_note}

**重要任務：**
1. **首先**：仔細閱讀對話轉錄內容，理解對話的真實意圖和上下文
2. **然後**：參考股票參考資料作為輔助，但不要完全依賴
3. **注意**：如果對話內容與參考資料衝突，優先相信對話內容本身
4. 識別對話中的交易意圖（買入/賣出/排隊）
5. 提取交易細節（股票代號、股票名稱、數量、價格）
6. **必須**為每個交易指定 conversation_number（從對話編號中獲取）
7. **注意**：以下字段會自動從對話元數據中提取並添加到每個交易（系統會自動處理，不需要在返回的JSON中包含）：
   - hkt_datetime（日期時間）
   - broker_id, broker_name（經紀信息）
   - client_id, client_name（客戶信息）
8. 評估置信度分數（0.0-1.0）：
   - 基於對話內容的清晰度
   - 參考資料的元數據可作為輔助參考
   - 綜合判斷給出最終置信度
9. 在 conversation_analysis 中詳細分析對話內容
10. 在 overall_summary 中綜合對話內容和參考資料給出總結

請根據以上資料，使用結構化格式返回分析結果。
"""
        
        # Initialize the LLM with structured output
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=temperature,
            format="json",  # Request JSON format
        )
        
        # Prepare messages
        messages = [
            ("system", system_message),
            ("human", prompt),
        ]
        
        # Get response
        print(f"🔍 Analyzing transactions with {model}...")
        resp = chat_llm.invoke(messages)
        
        # Extract content
        try:
            response_content = getattr(resp, "content", str(resp))
        except Exception:
            response_content = str(resp)
        
        # Translate LLM response to Traditional Chinese
        response_content = translate_to_traditional_chinese(response_content)
        
        # Try to parse as structured output
        try:
            result_dict = json.loads(response_content)
            
            # Debug: Print raw JSON response
            print("="*60)
            print("🔍 DEBUG: Raw LLM JSON Response:")
            print(json.dumps(result_dict, indent=2, ensure_ascii=False))
            print("="*60)
            
            # Extract the structure
            transactions = result_dict.get("transactions", [])
            conversation_analysis = result_dict.get("conversation_analysis", "")
            overall_summary = result_dict.get("overall_summary", "")
            
            # Programmatically add metadata to each transaction based on conversation_number
            for tx in transactions:
                conv_num = tx.get("conversation_number", None)
                if conv_num:
                    tx["hkt_datetime"] = conversation_datetime_map.get(conv_num, "N/A")
                    tx["broker_id"] = conversation_broker_id_map.get(conv_num, "N/A")
                    tx["broker_name"] = conversation_broker_name_map.get(conv_num, "N/A")
                    tx["client_id"] = conversation_client_id_map.get(conv_num, "N/A")
                    tx["client_name"] = conversation_client_name_map.get(conv_num, "N/A")
                else:
                    tx["hkt_datetime"] = "N/A"
                    tx["broker_id"] = "N/A"
                    tx["broker_name"] = "N/A"
                    tx["client_id"] = "N/A"
                    tx["client_name"] = "N/A"
            
            # Count conversations
            try:
                conv_count = len(json.loads(conversation_json_input))
            except:
                conv_count = 1
            
            # Create formatted summary result for all transactions
            summary_result = f"""📊 交易分析結果（基於對話轉錄 + 股票參考資料）
{'='*50}

📞 分析的對話數：{conv_count}
📋 識別到的交易數：{len(transactions)}
📈 參考的股票數：{len(stock_list_for_checking)}

"""
            
            if len(transactions) == 0:
                summary_result += "ℹ️ 沒有識別到確定的交易\n\n"
            else:
                for idx, tx in enumerate(transactions, 1):
                    tx_type = tx.get("transaction_type", "unknown")
                    tx_conf = tx.get("confidence_score", 0.0)
                    tx_conv_num = tx.get("conversation_number", None)
                    tx_code = tx.get("stock_code", "") or "N/A"
                    tx_name = tx.get("stock_name", "") or "N/A"
                    tx_qty = tx.get("quantity", "") or "N/A"
                    tx_price = tx.get("price", "") or "N/A"
                    tx_exp = tx.get("explanation", "")
                    
                    # Get hkt_datetime from transaction (already added programmatically)
                    tx_datetime = tx.get("hkt_datetime", "N/A")
                    
                    # Transaction type display
                    tx_type_display = {
                        "buy": "買入 📈",
                        "sell": "賣出 📉",
                        "queue": "排隊 ⏳",
                        "unknown": "未知 ❓"
                    }.get(tx_type, tx_type)
                    
                    # Get broker and client info from transaction
                    tx_broker_id = tx.get("broker_id", "N/A")
                    tx_broker_name = tx.get("broker_name", "N/A")
                    tx_client_id = tx.get("client_id", "N/A")
                    tx_client_name = tx.get("client_name", "N/A")
                    
                    summary_result += f"""{'─'*50}
交易 #{idx}
{'─'*50}
📅 日期時間 (HKT): {tx_datetime}
💬 對話編號: {tx_conv_num if tx_conv_num else 'N/A'}
👤 經紀ID: {tx_broker_id}
👔 經紀姓名: {tx_broker_name}
🆔 客戶ID: {tx_client_id}
👥 客戶姓名: {tx_client_name}
🔖 交易類型: {tx_type_display}
⭐ 置信度分數: {tx_conf} / 1.0
📈 股票代號: {tx_code}
🏢 股票名稱: {tx_name}
🔢 數量: {tx_qty}
💰 價格: {tx_price}

📝 分析說明:
{tx_exp}

"""
            
            if conversation_analysis:
                summary_result += f"""{'='*50}
💬 對話內容分析:
{conversation_analysis}

"""
            
            summary_result += f"""{'='*50}
📄 整體摘要:
{overall_summary}
"""
            
            # Format JSON result with proper indentation
            json_result = json.dumps(result_dict, indent=2, ensure_ascii=False)
            
            return (summary_result, json_result)
            
        except json.JSONDecodeError:
            # If not valid JSON, return the raw response
            error_msg = f"⚠️ 模型返回非結構化輸出：\n\n{response_content}"
            return (error_msg, response_content)
        
    except Exception as e:
        error_msg = f"❌ 錯誤: {str(e)}\n\n詳細信息:\n{traceback.format_exc()}"
        return (error_msg, "")


def create_transaction_analysis_json_tab():
    """Create and return the Transaction Analysis with JSON tab"""
    with gr.Tab("📊 Transaction Analysis (JSON)"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📥 輸入方式 1：對話JSON輸入（主要來源）")
                
                conversation_json_box = gr.Textbox(
                    label="對話JSON數據 (Conversation JSON) - 主要分析來源",
                    placeholder='''[
  {
    "conversation_number": 1,
    "filename": "example.wav",
    "metadata": {
      "broker_id": "B001",
      "broker_name": "Dickson Lau",
      "client_id": "C123",
      "client_name": "CHENG SUK HING",
      "hkt_datetime": "2025-10-20T10:15:30"
    },
    "transcriptions": {
      "sensevoice": "經紀: 你好\\n客戶: 我想買騰訊"
    }
  }
]''',
                    lines=12,
                    show_copy_button=True,
                    info="貼上對話JSON，系統會自動提取股票信息和日期時間"
                )
                
                with gr.Row():
                    conv_llm_checkboxes = gr.CheckboxGroup(
                        choices=LLM_OPTIONS,
                        label="選擇LLM進行股票提取",
                        value=[LLM_OPTIONS[0]],
                        info="選擇一個或多個LLM來分析對話"
                    )
                
                with gr.Row():
                    extract_stocks_btn = gr.Button(
                        "🔍 從對話提取股票",
                        variant="secondary",
                        size="sm"
                    )
                    use_vector_correction_checkbox = gr.Checkbox(
                        label="🔧 啟用向量校正",
                        value=True,
                    )
                
                extraction_status_box = gr.Textbox(
                    label="提取狀態",
                    lines=4,
                    interactive=False,
                    show_copy_button=True,
                )
                
                gr.Markdown("#### 📥 輸入方式 2：合併股票JSON數據（次要參考）")
                
                merged_json_box = gr.Textbox(
                    label="合併股票JSON數據 (Merged JSON) - 僅作參考，可能不準確",
                    placeholder='''{
  "stocks": [
    {
      "stock_number": "00700",
      "stock_name": "騰訊控股",
      "relevance_score": 0.85,
      "original_word": "買入騰訊",
      "corrected_stock_number": "00700",
      "corrected_stock_name": "騰訊控股",
      "correction_confidence": 1.0,
      "confidence": "high",
      "detection_count": 3,
      "detected_by_llms": ["qwen2.5:32b", "llama3.1:70b"]
    }
  ],
  "metadata": {
    "total_conversations": 5,
    "total_analyses": 10,
    "unique_stocks_found": 8
  }
}''',
                    lines=10,
                    info="從 JSON Batch Analysis 的 Merged JSON Output 複製，或從上面的提取結果自動填充",
                )
                
                gr.Markdown("#### LLM 設定")
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=MODEL_OPTIONS,
                        value=DEFAULT_MODEL,
                        label="模型",
                        allow_custom_value=True,
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.3,
                        step=0.1,
                        label="Temperature",
                        info="較低的溫度會讓結果更確定",
                    )
                
                ollama_url_box = gr.Textbox(
                    label="Ollama URL",
                    value=DEFAULT_OLLAMA_URL,
                    placeholder="http://localhost:11434",
                )
                
                system_message_box = gr.Textbox(
                    label="系統訊息 (System Message)",
                    value=DEFAULT_SYSTEM_MESSAGE,
                    lines=6,
                )
                
                analyze_btn = gr.Button(
                    "🚀 開始分析交易",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("#### 分析結果")
                
                # Summary Result Textbox (all transactions)
                summary_result_box = gr.Textbox(
                    label="📊 完整結果摘要 (All Transactions)",
                    lines=18,
                    interactive=False,
                    show_copy_button=True,
                )
                
                # JSON Result Textbox (raw output)
                json_result_box = gr.Textbox(
                    label="📋 Pydantic JSON 輸出 (JSON Output)",
                    lines=18,
                    interactive=False,
                    show_copy_button=True,
                )
        
        # Connect the extract stocks button
        extract_stocks_btn.click(
            fn=process_conversation_json_to_merged,
            inputs=[
                conversation_json_box,
                conv_llm_checkboxes,
                ollama_url_box,
                temperature_slider,
                use_vector_correction_checkbox,
            ],
            outputs=[
                extraction_status_box,
                merged_json_box,
            ],
        )
        
        # Connect the analyze button
        analyze_btn.click(
            fn=analyze_transactions_with_json,
            inputs=[
                conversation_json_box,
                merged_json_box,
                model_dropdown,
                ollama_url_box,
                system_message_box,
                temperature_slider,
            ],
            outputs=[
                summary_result_box,
                json_result_box,
            ],
        )

