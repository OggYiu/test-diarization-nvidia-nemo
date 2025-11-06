"""
Tab: STT & Stock Extraction Comparison
Compare up to three transcriptions from different STT models and extract stock information using multiple LLMs.
Empty transcriptions are automatically skipped to improve performance.
"""

import json
import traceback
import logging
import time
import re
from typing import List, Optional
from datetime import datetime
from collections import Counter
import gradio as gr
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser

# Import centralized model configuration
from model_config import MODEL_OPTIONS, DEFAULT_OLLAMA_URL

# Import stock verification functionality
from stock_verifier_module.stock_verifier_improved import (
    get_vector_store,
    verify_and_correct_stock,
    StockCorrectionResult,
    SearchStrategy,
)


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

class StockInfo(BaseModel):
    """Information about a single stock mentioned in the conversation"""
    stock_number: str = Field(
        description="The stock code/number (e.g., '00700', '1810', '18138')"
    )
    stock_name: str = Field(
        description="The stock name in Traditional Chinese (e.g., '騰訊', '小米', '招商局置地')"
    )
    original_word: Optional[str] = Field(
        default=None,
        description="The exact original word/phrase from the transcription (e.g., '金碟' if it was misheard as '金蝶'). Only include if different from stock_name."
    )
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'"
    )
    relevance_score: float = Field(
        description="How sure the conversation talks about this specific stock (0.0=not discussed, 0.5=mentioned briefly, 1.0=actively discussed/traded). Use values between 0.0 and 1.0."
    )
    quantity: Optional[str] = Field(
        default=None,
        description="The quantity/amount of stocks mentioned in the conversation (e.g., '1000股', '10手', '100張'). Include if mentioned."
    )
    price: Optional[str] = Field(
        default=None,
        description="The price mentioned in the conversation (e.g., 'HK$350', '$12.5', '市價'). Include if mentioned."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of how the stock was identified or any corrections made"
    )
    corrected_stock_name: Optional[str] = Field(
        default=None,
        description="Stock name after vector store correction (if different from original)"
    )
    corrected_stock_number: Optional[str] = Field(
        default=None,
        description="Stock number after vector store correction (if different from original)"
    )
    correction_confidence: Optional[float] = Field(
        default=None,
        description="Confidence score for the correction from vector store (0.0-1.0)"
    )


class ConversationStockExtraction(BaseModel):
    """Complete extraction result from a conversation"""
    stocks: List[StockInfo] = Field(
        description="List of all stocks mentioned in the conversation"
    )
    summary: str = Field(
        description="Brief summary of the conversation context"
    )


# ============================================================================
# Utility Functions
# ============================================================================

def convert_chinese_number_to_digit(text: str) -> str:
    """
    Convert Chinese numerals and text quantities to numeric digits.
    
    Examples:
        "一千" -> "1000"
        "兩萬" -> "20000"
        "10手" -> "10"
        "1000股" -> "1000"
        "三百五十" -> "350"
        "5萬股" -> "50000"
        
    Args:
        text: Input text containing numbers (Chinese or Arabic)
        
    Returns:
        str: Numeric value as string, or empty string if conversion fails
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text).strip()
    
    # If already a pure number, return it
    if text.isdigit():
        return text
    
    # Try to match pure decimal numbers (including floats)
    pure_number_match = re.match(r'^(\d+(?:\.\d+)?)$', text)
    if pure_number_match:
        try:
            return str(int(float(pure_number_match.group(1))))
        except:
            return pure_number_match.group(1)
    
    # Chinese number mappings
    chinese_digits = {
        '零': 0, '〇': 0,
        '一': 1, '壹': 1,
        '二': 2, '貳': 2, '兩': 2, '两': 2,
        '三': 3, '參': 3, '叁': 3,
        '四': 4, '肆': 4,
        '五': 5, '伍': 5,
        '六': 6, '陸': 6,
        '七': 7, '柒': 7,
        '八': 8, '捌': 8,
        '九': 9, '玖': 9,
    }
    
    chinese_units = {
        '十': 10, '拾': 10,
        '百': 100, '佰': 100,
        '千': 1000, '仟': 1000,
        '萬': 10000, '万': 10000,
        '億': 100000000, '亿': 100000000,
    }
    
    # Try to extract existing Arabic numerals first
    arabic_match = re.search(r'(\d+(?:\.\d+)?)', text)
    if arabic_match:
        num_str = arabic_match.group(1)
        # Check if there's a Chinese unit after the number (e.g., "5萬")
        remaining = text[arabic_match.end():]
        for unit_char, unit_val in chinese_units.items():
            if unit_char in remaining:
                try:
                    return str(int(float(num_str) * unit_val))
                except:
                    pass
        # Return just the numeric part, removing any non-numeric characters
        try:
            return str(int(float(num_str)))
        except:
            return num_str
    
    # Convert pure Chinese numerals
    try:
        # Remove common suffixes like 股, 手, 張, etc.
        cleaned = re.sub(r'[股手張张块塊元蚊]', '', text).strip()
        
        if not any(c in cleaned for c in chinese_digits.keys() | chinese_units.keys()):
            # No Chinese numerals found, return empty string instead of original text
            logging.warning(f"No numeric value found in '{text}'")
            return ""
        
        total = 0
        current = 0
        
        i = 0
        while i < len(cleaned):
            char = cleaned[i]
            
            if char in chinese_digits:
                current = chinese_digits[char]
                i += 1
            elif char in chinese_units:
                unit = chinese_units[char]
                if unit >= 10000:  # 萬 or 億
                    total = (total + current) * unit
                    current = 0
                else:  # 十, 百, 千
                    if current == 0:
                        current = 1  # Handle cases like "十" meaning "10"
                    total += current * unit
                    current = 0
                i += 1
            else:
                i += 1
        
        total += current
        
        # If total is still 0, return empty string instead of original text
        if total <= 0:
            logging.warning(f"Failed to convert Chinese number '{text}' - total is 0")
            return ""
            
        return str(int(total))
        
    except Exception as e:
        logging.warning(f"Failed to convert Chinese number '{text}': {e}")
        # Return empty string instead of original text
        return ""


# ============================================================================
# Model Configuration
# ============================================================================

# Note: MODEL_OPTIONS and DEFAULT_OLLAMA_URL are imported from model_config.py
# Use MODEL_OPTIONS as LLM_OPTIONS for consistency with this module
LLM_OPTIONS = MODEL_OPTIONS

DEFAULT_SYSTEM_MESSAGE = """你是一位精通粵語的香港股市分析專家。你的任務是從電話錄音的文字轉錄中識別所有提及的股票。

**重要提示:**
- 由於Speech-to-Text技術的誤差，文字中可能有誤認詞彙
- 你需要運用專業知識和邏輯推理，推斷並還原正確的股票資訊
- 股票代號可能以不同形式出現（例如：「七百」可能是「00700」騰訊）

**常見誤差:**
- 百: 八, 例子: 一百一三八 → 18138
- 孤: 沽
- 古: 沽
- 星: 升
- 號: 毫

**常見術語:**
- 掛: 掛單
- 沽: 賣出
- 孤: 賣出
- 轮: 窩輪
- 星: 升
- 號: 毫
- 手: 股票交易單位（1手通常=100股，但某些股票不同）
- 張: 窩輪/牛熊證的交易單位

**你的目標:**
1. 識別所有提及的股票代號和名稱
2. 修正任何可能的Speech-to-Text誤差
3. **如果你修正了股票名稱（即轉錄文本中的詞與正確股票名稱不同），請在 original_word 欄位中提供轉錄文本中的原始詞語**
4. **提取交易數量和價格信息**（如果在對話中提及）
5. 評估每個識別的置信度（high/medium/low）
6. 評估對話與該股票的相關程度（relevance_score）：
   - 0.0: 沒有實質討論（僅背景噪音或無關提及）
   - 0.5: 簡短提及或詢問（例如：問價、一般查詢）
   - 1.0: 積極討論或交易（例如：下單、詳細分析、交易確認）
   - 可以使用 0.0 到 1.0 之間的任何數值（例如：0.3, 0.7, 0.9 等）
7. 提供簡要的推理解釋

**關於 original_word 欄位:**
- 只在你修正了STT誤差時才填寫此欄位
- 例如：如果轉錄文本說「金碟」但正確的是「金蝶國際」，則 original_word 應為「金碟」
- 如果轉錄文本本身就是正確的，則省略此欄位

**關於 quantity 和 price 欄位:**
- **quantity**: 提取對話中提及的股票數量，例如：
  - "1000股" → quantity: "1000股"
  - "10手" → quantity: "10手"
  - "5萬股" → quantity: "5萬股"
  - "100張" (窩輪) → quantity: "100張"
- **price**: 提取對話中提及的股票價格，例如：
  - "$350" → price: "HK$350"
  - "三百五十蚊" → price: "HK$350"
  - "市價" → price: "市價"
  - "十二點五" → price: "HK$12.5"
  - "三毫" → price: "HK$0.3"
- 如果對話中沒有明確提及數量或價格，則省略這些欄位

請以結構化的JSON格式返回結果。"""


# ============================================================================
# Core Extraction Functions
# ============================================================================

def extract_stocks_with_single_llm(
    model: str,
    conversation_text: str,
    system_message: str,
    ollama_url: str,
    temperature: float,
    stt_source: str,
    use_vector_correction: bool = True,
) -> tuple[str, str, str]:
    """
    Extract stock information using a single LLM.
    
    Args:
        model: LLM model name
        conversation_text: The conversation transcript
        system_message: System message for the LLM
        ollama_url: Ollama server URL
        temperature: Temperature for generation
        stt_source: Label for the STT source (for display)
        use_vector_correction: Whether to use vector store for stock name correction
        
    Returns:
        tuple[str, str, str]: (model_name, formatted_result, raw_json)
    """
    try:
        # Initialize parser and LLM
        parser = PydanticOutputParser(pydantic_object=ConversationStockExtraction)
        
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=temperature,
        )
        
        # Build the complete prompt with format instructions
        format_instructions = parser.get_format_instructions()
        
        full_prompt = f"""{conversation_text}

{format_instructions}

請按照上述格式返回所有識別出的股票資訊。"""
        
        # Prepare messages
        messages = [
            ("system", system_message),
            ("human", full_prompt),
        ]
        
        # Get response from LLM
        resp = chat_llm.invoke(messages)
        
        # Extract content
        try:
            response_content = getattr(resp, "content", str(resp))
        except Exception:
            response_content = str(resp)
        
        # Parse the response
        try:
            parsed_result: ConversationStockExtraction = parser.parse(response_content)
            
            # Convert quantities to numeric format
            for stock in parsed_result.stocks:
                if stock.quantity:
                    stock.quantity = convert_chinese_number_to_digit(stock.quantity)
            
            # Apply vector store correction if enabled
            if use_vector_correction:
                vector_store = get_vector_store()
                if vector_store.initialize():
                    corrected_stocks = []
                    for stock in parsed_result.stocks:
                        # Use the new optimized verification function
                        correction_result = verify_and_correct_stock(
                            stock_name=stock.stock_name,
                            stock_code=stock.stock_number,
                            vector_store=vector_store,
                            strategy=SearchStrategy.OPTIMIZED,
                        )
                        
                        # Always populate corrected fields
                        if correction_result.correction_applied:
                            # Correction was applied - use corrected values
                            stock.corrected_stock_name = correction_result.corrected_stock_name or stock.stock_name
                            stock.corrected_stock_number = correction_result.corrected_stock_code or stock.stock_number
                            stock.correction_confidence = correction_result.confidence
                            
                            # Update reasoning
                            if stock.reasoning:
                                stock.reasoning = f"{stock.reasoning} | {correction_result.reasoning}"
                            else:
                                stock.reasoning = correction_result.reasoning
                        else:
                            # No correction needed - use original values with 1.0 confidence
                            stock.corrected_stock_name = stock.stock_name
                            stock.corrected_stock_number = stock.stock_number
                            stock.correction_confidence = 1.0
                        
                        corrected_stocks.append(stock)
                    
                    parsed_result.stocks = corrected_stocks
            else:
                # Vector correction disabled - still populate with original values
                for stock in parsed_result.stocks:
                    stock.corrected_stock_name = stock.stock_name
                    stock.corrected_stock_number = stock.stock_number
                    stock.correction_confidence = 1.0
            
            # Format the result for display
            formatted_output = format_extraction_result(parsed_result, model, stt_source)
            
            # Also return the raw JSON for reference
            # Note: exclude_none=False ensures corrected_stock_name and corrected_stock_number are always present
            raw_json = parsed_result.model_dump_json(indent=2, exclude_none=False)
            
            return (model, formatted_output, raw_json)
            
        except Exception as parse_error:
            error_msg = f"⚠️ Warning: Could not parse structured output\n\n"
            error_msg += f"Parse Error: {str(parse_error)}\n\n"
            error_msg += f"Raw LLM Response:\n{response_content}"
            return (model, error_msg, response_content)
    
    except Exception as e:
        error_msg = f"❌ Error with {model}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return (model, error_msg, "")


def format_extraction_result(result: ConversationStockExtraction, model: str, stt_source: str) -> str:
    """Format the extraction result for display"""
    output = []
    
    output.append("=" * 80)
    output.append(f"📊 股票提取結果 (Stock Extraction Results)")
    output.append(f"🤖 LLM 模型: {model}")
    output.append(f"🎤 STT 來源: {stt_source}")
    output.append("=" * 80)
    output.append("")
    
    # Summary
    output.append(f"📝 對話摘要: {result.summary}")
    output.append("")
    
    # Stocks found
    output.append(f"🔍 找到 {len(result.stocks)} 個股票:")
    output.append("")
    
    if not result.stocks:
        output.append("   ⚠️ 未找到任何股票資訊")
    else:
        for i, stock in enumerate(result.stocks, 1):
            confidence_emoji = {
                "high": "✅",
                "medium": "⚡",
                "low": "⚠️"
            }.get(stock.confidence.lower(), "❓")
            
            # Determine relevance emoji based on score ranges
            if stock.relevance_score < 0.25:
                relevance_emoji = "⚫"  # Not discussed
            elif stock.relevance_score < 0.75:
                relevance_emoji = "🔵"  # Mentioned briefly
            else:
                relevance_emoji = "🟢"  # Actively discussed
            
            output.append(f"   {i}. {confidence_emoji} 股票 #{i}")
            output.append(f"      • 股票代號: {stock.stock_number}")
            output.append(f"      • 股票名稱: {stock.stock_name}")
            
            # Show original word if available
            if stock.original_word:
                output.append(f"      • 原始詞語: {stock.original_word}")
            
            # Show quantity and price if available
            if stock.quantity:
                output.append(f"      • 數量: {stock.quantity}")
            if stock.price:
                output.append(f"      • 價格: {stock.price}")
            
            # Show corrections if available
            if stock.corrected_stock_number or stock.corrected_stock_name:
                output.append(f"      🔧 修正後:")
                if stock.corrected_stock_number:
                    output.append(f"         ◦ 股票代號: {stock.corrected_stock_number}")
                if stock.corrected_stock_name:
                    output.append(f"         ◦ 股票名稱: {stock.corrected_stock_name}")
                if stock.correction_confidence:
                    output.append(f"         ◦ 修正信心: {stock.correction_confidence:.2%}")
            
            output.append(f"      • 置信度: {stock.confidence.upper()}")
            output.append(f"      • 相關程度: {relevance_emoji} {stock.relevance_score:.2f}")
            
            if stock.reasoning:
                output.append(f"      • 推理: {stock.reasoning}")
            
            output.append("")
    
    output.append("=" * 80)
    
    return "\n".join(output)


def process_transcriptions(
    transcription1: str,
    transcription2: str,
    transcription3: str,
    selected_llms: list[str],
    system_message: str,
    ollama_url: str,
    temperature: float,
    use_vector_correction: bool = True,
) -> tuple[str, str, str]:
    """
    Process all three transcriptions with selected LLMs and compare results.
    
    Args:
        transcription1: First transcription text
        transcription2: Second transcription text
        transcription3: Third transcription text
        selected_llms: List of selected LLM names
        system_message: System message for the LLMs
        ollama_url: Ollama server URL
        temperature: Temperature parameter
        use_vector_correction: Whether to use vector store for stock name correction
        
    Returns:
        tuple[str, str, str]: (formatted_comparison, raw_json_collection, combined_json)
    """
    try:
        # Check which transcriptions are provided
        has_trans1 = bool(transcription1 and transcription1.strip())
        has_trans2 = bool(transcription2 and transcription2.strip())
        has_trans3 = bool(transcription3 and transcription3.strip())
        
        # Validate that at least one transcription is provided
        if not (has_trans1 or has_trans2 or has_trans3):
            return "❌ Error: Please provide at least one transcription", "", ""
        
        if not selected_llms or len(selected_llms) == 0:
            return "❌ Error: Please select at least one LLM", "", ""
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL", "", ""
        
        # Results storage
        results_trans1 = {}
        results_trans2 = {}
        results_trans3 = {}
        raw_jsons = {}
        
        # Process one LLM at a time to avoid overwhelming VRAM
        # For each LLM, process all transcriptions sequentially
        for model in selected_llms:
            # Process transcription 1 (only if provided)
            if has_trans1:
                msg = f"Starting analysis for STT Model 1 with LLM: {model}"
                logging.info(msg)
                print(msg)
                start_time = time.time()
                
                result_model, formatted_result, raw_json = extract_stocks_with_single_llm(
                    model,
                    transcription1,
                    system_message,
                    ollama_url,
                    temperature,
                    "STT Model 1",
                    use_vector_correction
                )
                
                elapsed_time = time.time() - start_time
                msg = f"Completed analysis for STT Model 1 with LLM: {model} - Time taken: {elapsed_time:.2f} seconds"
                logging.info(msg)
                print(msg)
                
                results_trans1[model] = formatted_result
                raw_jsons[f"{model}_trans1"] = raw_json
            
            # Process transcription 2 (only if provided)
            if has_trans2:
                msg = f"Starting analysis for STT Model 2 with LLM: {model}"
                logging.info(msg)
                print(msg)
                start_time = time.time()
                
                result_model, formatted_result, raw_json = extract_stocks_with_single_llm(
                    model,
                    transcription2,
                    system_message,
                    ollama_url,
                    temperature,
                    "STT Model 2",
                    use_vector_correction
                )
                
                elapsed_time = time.time() - start_time
                msg = f"Completed analysis for STT Model 2 with LLM: {model} - Time taken: {elapsed_time:.2f} seconds"
                logging.info(msg)
                print(msg)
                
                results_trans2[model] = formatted_result
                raw_jsons[f"{model}_trans2"] = raw_json
            
            # Process transcription 3 (only if provided)
            if has_trans3:
                msg = f"Starting analysis for STT Model 3 with LLM: {model}"
                logging.info(msg)
                print(msg)
                start_time = time.time()
                
                result_model, formatted_result, raw_json = extract_stocks_with_single_llm(
                    model,
                    transcription3,
                    system_message,
                    ollama_url,
                    temperature,
                    "STT Model 3",
                    use_vector_correction
                )
                
                elapsed_time = time.time() - start_time
                msg = f"Completed analysis for STT Model 3 with LLM: {model} - Time taken: {elapsed_time:.2f} seconds"
                logging.info(msg)
                print(msg)
                
                results_trans3[model] = formatted_result
                raw_jsons[f"{model}_trans3"] = raw_json
        
        # Format output
        output_parts = []
        output_parts.append("=" * 80)
        output_parts.append("🔬 STT & STOCK EXTRACTION COMPARISON")
        output_parts.append(f"Selected LLMs: {len(selected_llms)}")
        
        # Show which transcriptions are active
        active_trans_list = []
        if has_trans1:
            active_trans_list.append("Transcription 1")
        if has_trans2:
            active_trans_list.append("Transcription 2")
        if has_trans3:
            active_trans_list.append("Transcription 3")
        
        output_parts.append(f"Active Transcriptions: {len(active_trans_list)}/3")
        output_parts.append(f"Analyzing: {', '.join(active_trans_list)}")
        output_parts.append("=" * 80)
        output_parts.append("")
        
        for i, model in enumerate(selected_llms, 1):
            output_parts.append(f"\n{'=' * 80}")
            output_parts.append(f"🤖 LLM {i}/{len(selected_llms)}: {model}")
            output_parts.append("=" * 80)
            output_parts.append("")
            
            # Results from transcription 1 (only if provided)
            if has_trans1:
                output_parts.append("┌─ 📄 TRANSCRIPTION 1 RESULTS")
                output_parts.append("│")
                result1 = results_trans1.get(model, "❌ No response")
                for line in result1.split("\n"):
                    output_parts.append(f"│  {line}")
                output_parts.append("└" + "─" * 79)
                output_parts.append("")
            
            # Results from transcription 2 (only if provided)
            if has_trans2:
                output_parts.append("┌─ 📄 TRANSCRIPTION 2 RESULTS")
                output_parts.append("│")
                result2 = results_trans2.get(model, "❌ No response")
                for line in result2.split("\n"):
                    output_parts.append(f"│  {line}")
                output_parts.append("└" + "─" * 79)
                output_parts.append("")
            
            # Results from transcription 3 (only if provided)
            if has_trans3:
                output_parts.append("┌─ 📄 TRANSCRIPTION 3 RESULTS")
                output_parts.append("│")
                result3 = results_trans3.get(model, "❌ No response")
                for line in result3.split("\n"):
                    output_parts.append(f"│  {line}")
                output_parts.append("└" + "─" * 79)
                output_parts.append("")
        
        output_parts.append("=" * 80)
        output_parts.append("✓ All comparisons completed")
        output_parts.append("=" * 80)
        
        # Format raw JSON output
        json_output = []
        json_output.append("=" * 80)
        json_output.append("RAW JSON OUTPUTS")
        json_output.append("=" * 80)
        json_output.append("")
        
        for key, value in raw_jsons.items():
            json_output.append(f"\n--- {key} ---")
            json_output.append(value)
            json_output.append("")
        
        # Create combined JSON structure with merged stocks
        # Dictionary to track stocks by stock_number
        stocks_dict = {}  # key: stock_number, value: list of stock data
        
        # Parse all results and collect stocks
        for key, json_str in raw_jsons.items():
            if json_str and json_str.strip():
                try:
                    parsed = json.loads(json_str)
                    # Extract stocks from the parsed result
                    stocks = parsed.get("stocks", [])
                    
                    for stock in stocks:
                        stock_number = stock.get("stock_number", "")
                        if stock_number:
                            if stock_number not in stocks_dict:
                                stocks_dict[stock_number] = []
                            stocks_dict[stock_number].append(stock)
                
                except json.JSONDecodeError:
                    # Skip invalid JSON
                    continue
        
        # Merge stocks and calculate average relevance_score
        merged_stocks = []
        
        # Calculate total number of analyses (should match len(raw_jsons))
        total_analyses = len(raw_jsons)
        
        for stock_number, stock_list in stocks_dict.items():
            if not stock_list:
                continue
            
            # Calculate average relevance_score across ALL analyses
            # (not just the ones where the stock appeared)
            relevance_scores = [s.get("relevance_score", 0) for s in stock_list]
            total_score = sum(relevance_scores)
            avg_relevance_score = total_score / total_analyses if total_analyses > 0 else 0
            
            # Use the first stock's data as base
            merged_stock = {
                "stock_number": stock_number,
                "stock_name": stock_list[0].get("stock_name", ""),
                "relevance_score": round(avg_relevance_score, 2),
            }
            
            # Include original_word if present (use first non-empty one)
            original_words = [s.get("original_word", "") for s in stock_list if s.get("original_word")]
            if original_words:
                # Use the most common original word, or first if tied
                word_counts = Counter(original_words)
                most_common_word = word_counts.most_common(1)[0][0]
                merged_stock["original_word"] = most_common_word
            
            # Include corrected stock information (always populate, use first available or original values)
            corrected_names = [s.get("corrected_stock_name") for s in stock_list if s.get("corrected_stock_name")]
            corrected_numbers = [s.get("corrected_stock_number") for s in stock_list if s.get("corrected_stock_number")]
            correction_confidences = [s.get("correction_confidence") for s in stock_list if s.get("correction_confidence")]
            
            # Always include these fields - use corrected values if available, otherwise use original values
            merged_stock["corrected_stock_number"] = corrected_numbers[0] if corrected_numbers else stock_number
            merged_stock["corrected_stock_name"] = corrected_names[0] if corrected_names else stock_list[0].get("stock_name", "")
            merged_stock["correction_confidence"] = correction_confidences[0] if correction_confidences else 1.0
            
            # Determine confidence - use the most common one, or highest if tied
            confidences = [s.get("confidence", "low").lower() for s in stock_list]
            confidence_priority = {"high": 3, "medium": 2, "low": 1}
            most_confident = max(confidences, key=lambda c: (confidences.count(c), confidence_priority.get(c, 0)))
            merged_stock["confidence"] = most_confident
            
            # Combine reasoning from all sources (if present)
            reasonings = [s.get("reasoning", "") for s in stock_list if s.get("reasoning")]
            if reasonings:
                # Only include unique reasonings
                unique_reasonings = list(dict.fromkeys(reasonings))  # Preserve order while removing duplicates
                if len(unique_reasonings) == 1:
                    merged_stock["reasoning"] = unique_reasonings[0]
                else:
                    merged_stock["reasoning"] = " | ".join(unique_reasonings)
            
            # Add metadata about how many times this stock was found
            merged_stock["detection_count"] = len(stock_list)
            
            merged_stocks.append(merged_stock)
        
        # Sort by relevance_score (descending) then by stock_number
        merged_stocks.sort(key=lambda s: (-s["relevance_score"], s["stock_number"]))
        
        # Create simplified combined JSON with only stocks
        combined_data = {
            "stocks": merged_stocks
        }
        
        # Format combined JSON
        combined_json = json.dumps(combined_data, indent=2, ensure_ascii=False)
        
        return "\n".join(output_parts), "\n".join(json_output), combined_json
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, "", ""


# ============================================================================
# Gradio Tab Creation
# ============================================================================

def create_stt_stock_comparison_tab():
    """Create and return the STT & Stock Comparison tab"""
    with gr.Tab("9️⃣ STT Stock Comparison"):
        gr.Markdown("### Compare Transcriptions & Extract Stock Information")
        gr.Markdown("Input up to three different transcriptions from different STT models and compare stock extraction results using multiple LLMs. Empty transcriptions will be skipped automatically.")
        gr.Markdown("**🔧 New Feature**: Enable Vector Store Correction to automatically correct stock names that may have STT errors by matching against your Milvus stock database.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📝 Transcription Inputs")
                
                # Two textboxes for different transcriptions
                transcription1_box = gr.Textbox(
                    label="🎤 Transcription 1 (STT Model 1)",
                    placeholder="請輸入第一個 STT 模型的轉錄文本...\n\n例如：\n券商：你好，請問需要什麼幫助？\n客戶：我想買騰訊\n券商：好的，七百號，買多少？",
                    lines=10,
                    show_copy_button=True,
                )
                
                transcription2_box = gr.Textbox(
                    label="🎤 Transcription 2 (STT Model 2)",
                    placeholder="請輸入第二個 STT 模型的轉錄文本...\n\n例如：\n券商：你好，請問需要咩幫助？\n客戶：我想買騰訊\n券商：好嘅，七百號，買幾多？",
                    lines=10,
                    show_copy_button=True,
                )
                
                transcription3_box = gr.Textbox(
                    label="🎤 Transcription 3 (STT Model 3)",
                    placeholder="請輸入第三個 STT 模型的轉錄文本...\n\n例如：\n券商：你好，請問需要咩幫助？\n客戶：我想買騰訊\n券商：好嘅，七百號，買幾多？",
                    lines=10,
                    show_copy_button=True,
                )
                
                gr.Markdown("#### 🤖 Select LLMs for Analysis")
                
                llm_checkboxes = gr.CheckboxGroup(
                    choices=LLM_OPTIONS,
                    label="Available LLMs",
                    value=[LLM_OPTIONS[0]],  # Default to first model
                    info="Select one or more LLMs to compare their stock extraction results"
                )
                
                gr.Markdown("#### ⚙️ Advanced Settings")
                
                use_vector_correction_checkbox = gr.Checkbox(
                    label="🔧 Enable Vector Store Correction",
                    value=True,
                    info="Use Milvus vector store to correct stock names that may have STT errors"
                )
                
                system_message_box = gr.Textbox(
                    label="系統訊息 (System Message)",
                    value=DEFAULT_SYSTEM_MESSAGE,
                    lines=15,
                    max_lines=20,
                    show_copy_button=True,
                )
                
                with gr.Row():
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.1,
                        step=0.1,
                        label="Temperature",
                        info="Lower = more deterministic"
                    )
                
                ollama_url_box = gr.Textbox(
                    label="Ollama URL",
                    value=DEFAULT_OLLAMA_URL,
                    placeholder="http://localhost:11434",
                )
                
                analyze_btn = gr.Button(
                    "🚀 Analyze & Compare Stock Extraction",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("#### 📊 Comparison Results")
                
                results_box = gr.Textbox(
                    label="Stock Extraction Comparison",
                    lines=20,
                    interactive=False,
                    show_copy_button=True,
                )
                
                gr.Markdown("#### 🔧 Raw JSON Outputs")
                
                json_box = gr.Textbox(
                    label="Individual JSON Results",
                    lines=10,
                    interactive=False,
                    show_copy_button=True,
                )
                
                gr.Markdown("#### 📦 Combined JSON Output")
                
                combined_json_box = gr.Textbox(
                    label="Single Unified JSON",
                    lines=15,
                    interactive=False,
                    show_copy_button=True,
                    info="All results combined into a single JSON structure"
                )
        
        
        # Connect the analyze button
        analyze_btn.click(
            fn=process_transcriptions,
            inputs=[
                transcription1_box,
                transcription2_box,
                transcription3_box,
                llm_checkboxes,
                system_message_box,
                ollama_url_box,
                temperature_slider,
                use_vector_correction_checkbox,
            ],
            outputs=[results_box, json_box, combined_json_box],
        )
        


