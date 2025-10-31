"""
Tab: STT & Stock Extraction Comparison
Compare transcriptions from different STT models and extract stock information using multiple LLMs
"""

import traceback
from typing import List, Optional
import gradio as gr
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser


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
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'"
    )
    relevance_score: int = Field(
        description="How sure the conversation talks about this specific stock (0=not discussed, 1=mentioned briefly, 2=actively discussed/traded)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of how the stock was identified or any corrections made"
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
# Model Configuration
# ============================================================================

LLM_OPTIONS = [
    "qwen3:32b",
    "gpt-oss:20b",
    "gemma3-27b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
    "qwen2.5:72b",
    "llama3.3:70b",
]

DEFAULT_OLLAMA_URL = "http://localhost:11434"

DEFAULT_SYSTEM_MESSAGE = """你是一位精通粵語的香港股市分析專家。你的任務是從電話錄音的文字轉錄中識別所有提及的股票。

**重要提示:**
- 由於Speech-to-Text技術的誤差，文字中可能有誤認詞彙
- 你需要運用專業知識和邏輯推理，推斷並還原正確的股票資訊
- 股票代號可能以不同形式出現（例如：「七百」可能是「00700」騰訊）

**常見誤差:**
- 誤認: 百 → 正確: 八 (例: 一百一三八 → 18138)
- 誤認: 孤/沽 → 正確: 賣出
- 誤認: 轮 → 正確: 窩輪

**你的目標:**
1. 識別所有提及的股票代號和名稱
2. 修正任何可能的Speech-to-Text誤差
3. 評估每個識別的置信度（high/medium/low）
4. 評估對話與該股票的相關程度（relevance_score）：
   - 0: 沒有實質討論（僅背景噪音或無關提及）
   - 1: 簡短提及或詢問（例如：問價、一般查詢）
   - 2: 積極討論或交易（例如：下單、詳細分析、交易確認）
5. 提供簡要的推理解釋

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
            
            # Format the result for display
            formatted_output = format_extraction_result(parsed_result, model, stt_source)
            
            # Also return the raw JSON for reference
            raw_json = parsed_result.model_dump_json(indent=2, exclude_none=True)
            
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
            
            relevance_emoji = {
                0: "⚫",  # Not discussed
                1: "🔵",  # Mentioned briefly
                2: "🟢"   # Actively discussed
            }.get(stock.relevance_score, "❓")
            
            output.append(f"   {i}. {confidence_emoji} 股票 #{i}")
            output.append(f"      • 股票代號: {stock.stock_number}")
            output.append(f"      • 股票名稱: {stock.stock_name}")
            output.append(f"      • 置信度: {stock.confidence.upper()}")
            output.append(f"      • 相關程度: {relevance_emoji} {stock.relevance_score}/2")
            
            if stock.reasoning:
                output.append(f"      • 推理: {stock.reasoning}")
            
            output.append("")
    
    output.append("=" * 80)
    
    return "\n".join(output)


def process_transcriptions(
    transcription1: str,
    transcription2: str,
    selected_llms: list[str],
    system_message: str,
    ollama_url: str,
    temperature: float,
) -> tuple[str, str]:
    """
    Process both transcriptions with selected LLMs and compare results.
    
    Args:
        transcription1: First transcription text
        transcription2: Second transcription text
        selected_llms: List of selected LLM names
        system_message: System message for the LLMs
        ollama_url: Ollama server URL
        temperature: Temperature parameter
        
    Returns:
        tuple[str, str]: (formatted_comparison, raw_json_collection)
    """
    try:
        # Validate inputs
        if not transcription1 or not transcription1.strip():
            return "❌ Error: Please provide transcription 1", ""
        
        if not transcription2 or not transcription2.strip():
            return "❌ Error: Please provide transcription 2", ""
        
        if not selected_llms or len(selected_llms) == 0:
            return "❌ Error: Please select at least one LLM", ""
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL", ""
        
        # Results storage
        results_trans1 = {}
        results_trans2 = {}
        raw_jsons = {}
        
        # Process both transcriptions with all selected LLMs concurrently
        with ThreadPoolExecutor(max_workers=len(selected_llms) * 2) as executor:
            futures = {}
            
            # Submit tasks for transcription 1
            for model in selected_llms:
                future = executor.submit(
                    extract_stocks_with_single_llm,
                    model,
                    transcription1,
                    system_message,
                    ollama_url,
                    temperature,
                    "STT Model 1"
                )
                futures[future] = (model, 1)
            
            # Submit tasks for transcription 2
            for model in selected_llms:
                future = executor.submit(
                    extract_stocks_with_single_llm,
                    model,
                    transcription2,
                    system_message,
                    ollama_url,
                    temperature,
                    "STT Model 2"
                )
                futures[future] = (model, 2)
            
            # Collect results as they complete
            for future in as_completed(futures):
                model, trans_num = futures[future]
                result_model, formatted_result, raw_json = future.result()
                
                if trans_num == 1:
                    results_trans1[model] = formatted_result
                    raw_jsons[f"{model}_trans1"] = raw_json
                else:
                    results_trans2[model] = formatted_result
                    raw_jsons[f"{model}_trans2"] = raw_json
        
        # Format output
        output_parts = []
        output_parts.append("=" * 80)
        output_parts.append("🔬 STT & STOCK EXTRACTION COMPARISON")
        output_parts.append(f"Selected LLMs: {len(selected_llms)}")
        output_parts.append("=" * 80)
        output_parts.append("")
        
        for i, model in enumerate(selected_llms, 1):
            output_parts.append(f"\n{'=' * 80}")
            output_parts.append(f"🤖 LLM {i}/{len(selected_llms)}: {model}")
            output_parts.append("=" * 80)
            output_parts.append("")
            
            # Results from transcription 1
            output_parts.append("┌─ 📄 TRANSCRIPTION 1 RESULTS")
            output_parts.append("│")
            result1 = results_trans1.get(model, "❌ No response")
            for line in result1.split("\n"):
                output_parts.append(f"│  {line}")
            output_parts.append("└" + "─" * 79)
            output_parts.append("")
            
            # Results from transcription 2
            output_parts.append("┌─ 📄 TRANSCRIPTION 2 RESULTS")
            output_parts.append("│")
            result2 = results_trans2.get(model, "❌ No response")
            for line in result2.split("\n"):
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
        
        return "\n".join(output_parts), "\n".join(json_output)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, ""


# ============================================================================
# Gradio Tab Creation
# ============================================================================

def create_stt_stock_comparison_tab():
    """Create and return the STT & Stock Comparison tab"""
    with gr.Tab("9️⃣ STT Stock Comparison"):
        gr.Markdown("### Compare Transcriptions & Extract Stock Information")
        gr.Markdown("Input two different transcriptions from different STT models and compare stock extraction results using multiple LLMs.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📝 Transcription Inputs")
                
                # Two textboxes for different transcriptions
                transcription1_box = gr.Textbox(
                    label="🎤 Transcription 1 (STT Model 1)",
                    placeholder="請輸入第一個 STT 模型的轉錄文本...\n\n例如：\n券商：你好，請問需要什麼幫助？\n客戶：我想買騰訊\n券商：好的，七百號，買多少？",
                    lines=10,
                )
                
                transcription2_box = gr.Textbox(
                    label="🎤 Transcription 2 (STT Model 2)",
                    placeholder="請輸入第二個 STT 模型的轉錄文本...\n\n例如：\n券商：你好，請問需要咩幫助？\n客戶：我想買騰訊\n券商：好嘅，七百號，買幾多？",
                    lines=10,
                )
                
                gr.Markdown("#### 🤖 Select LLMs for Analysis")
                
                llm_checkboxes = gr.CheckboxGroup(
                    choices=LLM_OPTIONS,
                    label="Available LLMs",
                    value=[LLM_OPTIONS[0]],  # Default to first model
                    info="Select one or more LLMs to compare their stock extraction results"
                )
                
                gr.Markdown("#### ⚙️ Advanced Settings")
                
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
                    lines=25,
                    interactive=False,
                    show_copy_button=True,
                )
                
                gr.Markdown("#### 🔧 Raw JSON Outputs")
                
                json_box = gr.Textbox(
                    label="Structured Data (JSON)",
                    lines=12,
                    interactive=False,
                    show_copy_button=True,
                )
        
        # Example buttons
        gr.Markdown("### 💡 示例 (Examples)")
        
        with gr.Row():
            example_1 = gr.Button("示例 1: 騰訊交易對比", size="sm")
            example_2 = gr.Button("示例 2: 多股票 STT 差異", size="sm")
            example_3 = gr.Button("示例 3: 粵語變體對比", size="sm")
        
        # Connect the analyze button
        analyze_btn.click(
            fn=process_transcriptions,
            inputs=[
                transcription1_box,
                transcription2_box,
                llm_checkboxes,
                system_message_box,
                ollama_url_box,
                temperature_slider,
            ],
            outputs=[results_box, json_box],
        )
        
        # Example button handlers
        example_1.click(
            fn=lambda: (
                """券商：你好，請問需要什麼幫助？
客戶：我想買騰訊
券商：好的，七百號騰訊，買多少？
客戶：一千股，市價買入
券商：確認一下，七百號騰訊，買入一千股，市價，對嗎？
客戶：對的，謝謝""",
                """券商：你好，請問需要咩幫助？
客戶：我想買騰訊
券商：好嘅，七百號騰訊，買幾多？
客戶：一千股，市價買入
券商：確認一下，七百號騰訊，買入一千股，市價，啱唔啱？
客戶：啱，謝謝"""
            ),
            outputs=[transcription1_box, transcription2_box],
        )
        
        example_2.click(
            fn=lambda: (
                """客戶：早晨，我想問下小米同比亞迪今日走勢
券商：你好！小米一八一零今日升咗2%，比亞迪二一一一跌咗1%
客戶：咁我想沽五百股比亞迪，再入一千股小米
券商：好的，確認一下：沽出比亞迪二一一一五百股，買入小米一八一零一千股，啱唔啱？
客戶：啱，就咁做""",
                """客戶：早晨，我想問下小米同比亞迪今日走勢
券商：你好！小米18一零今日升左2%，比亞迪21一一跌左1%
客戶：咁我想沽五百股比亞迪，再入一千股小米
券商：好嘅，確認一下：沽出比亞迪21一一五百股，買入小米18一零一千股，啱唔啱？
客戶：啱，就咁做"""
            ),
            outputs=[transcription1_box, transcription2_box],
        )
        
        example_3.click(
            fn=lambda: (
                """客戶：我想買招商局置地
券商：招商局置地，係一百一三八號？
客戶：係呀
券商：買幾多？
客戶：五百股""",
                """客戶：我想買招商局置地
券商：招商局置地，係一八一三八號？
客戶：係呀
券商：買幾多？
客戶：五百股"""
            ),
            outputs=[transcription1_box, transcription2_box],
        )
        
        gr.Markdown(
            """
            ---
            ### 📌 使用說明 (Instructions)
            
            1. **輸入轉錄**: 將兩個不同 STT 模型的轉錄結果分別貼入兩個文本框
            2. **選擇 LLM**: 勾選一個或多個 LLM 模型進行股票提取分析
            3. **調整設置**: 可選調整系統訊息、Temperature 等參數
            4. **開始分析**: 點擊「Analyze & Compare」按鈕
            5. **查看對比**: 系統會並行處理所有組合，顯示詳細對比結果
            
            ### 🎯 功能特點 (Features)
            
            - 🔄 **雙轉錄對比**: 同時處理兩個不同 STT 模型的輸出
            - 🤖 **多 LLM 分析**: 使用多個 LLM 模型進行交叉驗證
            - ⚡ **並行處理**: 所有 LLM 查詢並行執行，提高效率
            - 📊 **結構化輸出**: 使用 Pydantic 保證數據格式一致
            - 🔍 **智能修正**: 自動識別並修正 STT 錯誤
            - 📈 **置信度評估**: 每個識別結果都有置信度評分
            """
        )


