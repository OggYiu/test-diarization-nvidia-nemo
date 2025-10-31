"""
Stock Information Extractor
Extract stock names and numbers from conversations using LLMs with Pydantic structured output
"""

import traceback
from typing import List, Optional
import gradio as gr
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

MODEL_OPTIONS = [
    "qwen3:32b",
    "gpt-oss:20b",
    "gemma3-27b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
    "qwen2.5:72b",
    "llama3.3:70b",
]

DEFAULT_MODEL = MODEL_OPTIONS[0]
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
4. 提供簡要的推理解釋

請以結構化的JSON格式返回結果。"""


# ============================================================================
# Core Extraction Function
# ============================================================================

def extract_stocks_from_conversation(
    conversation_text: str,
    model: str,
    ollama_url: str,
    system_message: str,
    temperature: float,
) -> tuple[str, str]:
    """
    Extract stock information from conversation using LLM with Pydantic structured output
    
    Args:
        conversation_text: The conversation transcript
        model: LLM model name
        ollama_url: Ollama server URL
        system_message: System message for the LLM
        temperature: Temperature for generation
    
    Returns:
        tuple[str, str]: (formatted_result, raw_json)
    """
    try:
        # Validate inputs
        if not conversation_text or not conversation_text.strip():
            return "❌ Error: Please provide conversation text", ""
        
        if not model or not model.strip():
            return "❌ Error: Please specify a model name", ""
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL", ""
        
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
        print(f"🔄 Calling {model}...")
        resp = chat_llm.invoke(messages)
        
        # Extract content
        try:
            response_content = getattr(resp, "content", str(resp))
        except Exception:
            response_content = str(resp)
        
        print(f"📥 Raw response received (length: {len(response_content)} chars)")
        
        # Parse the response
        try:
            parsed_result: ConversationStockExtraction = parser.parse(response_content)
            
            # Format the result for display
            formatted_output = format_extraction_result(parsed_result)
            
            # Also return the raw JSON for reference
            raw_json = parsed_result.model_dump_json(indent=2, exclude_none=True)
            
            return formatted_output, raw_json
            
        except Exception as parse_error:
            error_msg = f"⚠️ Warning: Could not parse structured output\n\n"
            error_msg += f"Parse Error: {str(parse_error)}\n\n"
            error_msg += f"Raw LLM Response:\n{response_content}"
            return error_msg, response_content
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, ""


def format_extraction_result(result: ConversationStockExtraction) -> str:
    """Format the extraction result for display"""
    output = []
    
    output.append("=" * 80)
    output.append("📊 股票提取結果 (Stock Extraction Results)")
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
            
            output.append(f"   {i}. {confidence_emoji} 股票 #{i}")
            output.append(f"      • 股票代號: {stock.stock_number}")
            output.append(f"      • 股票名稱: {stock.stock_name}")
            output.append(f"      • 置信度: {stock.confidence.upper()}")
            
            if stock.reasoning:
                output.append(f"      • 推理: {stock.reasoning}")
            
            output.append("")
    
    output.append("=" * 80)
    
    return "\n".join(output)


# ============================================================================
# Gradio Interface
# ============================================================================

def create_stock_extractor_interface():
    """Create the Gradio interface for stock extraction"""
    
    with gr.Blocks(title="Stock Extractor", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📈 智能股票信息提取器
            ## Stock Information Extractor with Pydantic & LLM
            
            從對話記錄中自動識別和提取股票代號和名稱，使用結構化輸出保證數據質量。
            """
        )
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📝 輸入設置 (Input Settings)")
                
                conversation_input = gr.Textbox(
                    label="對話記錄 (Conversation Text)",
                    placeholder="請輸入對話內容...\n\n例如：\n券商：你好，請問需要什麼幫助？\n客戶：我想買騰訊\n券商：好的，七百號，買多少？\n客戶：一千股...",
                    lines=12,
                )
                
                gr.Markdown("### ⚙️ LLM 模型設置 (LLM Settings)")
                
                model_dropdown = gr.Dropdown(
                    choices=MODEL_OPTIONS,
                    value=DEFAULT_MODEL,
                    label="選擇模型 (Select Model)",
                    allow_custom_value=True,
                )
                
                with gr.Row():
                    ollama_url_input = gr.Textbox(
                        label="Ollama URL",
                        value=DEFAULT_OLLAMA_URL,
                        scale=3,
                    )
                    
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.1,
                        step=0.1,
                        label="Temperature",
                        scale=1,
                    )
                
                system_message_input = gr.Textbox(
                    label="系統訊息 (System Message)",
                    value=DEFAULT_SYSTEM_MESSAGE,
                    lines=8,
                )
                
                extract_btn = gr.Button(
                    "🚀 開始提取股票資訊",
                    variant="primary",
                    size="lg",
                )
            
            # Right column - Output
            with gr.Column(scale=1):
                gr.Markdown("### 📊 提取結果 (Extraction Results)")
                
                result_output = gr.Textbox(
                    label="格式化結果 (Formatted Results)",
                    lines=15,
                    interactive=False,
                )
                
                gr.Markdown("### 🔧 原始 JSON 輸出 (Raw JSON Output)")
                
                json_output = gr.Textbox(
                    label="結構化數據 (Structured Data)",
                    lines=12,
                    interactive=False,
                    show_copy_button=True,
                )
        
        # Example conversations
        gr.Markdown("### 💡 示例對話 (Example Conversations)")
        
        with gr.Row():
            example_1 = gr.Button("示例 1: 騰訊交易", size="sm")
            example_2 = gr.Button("示例 2: 多隻股票討論", size="sm")
            example_3 = gr.Button("示例 3: 語音識別誤差", size="sm")
        
        # Event handlers
        extract_btn.click(
            fn=extract_stocks_from_conversation,
            inputs=[
                conversation_input,
                model_dropdown,
                ollama_url_input,
                system_message_input,
                temperature_slider,
            ],
            outputs=[result_output, json_output],
        )
        
        # Example button handlers
        example_1.click(
            fn=lambda: """券商：你好，請問需要什麼幫助？
客戶：我想買騰訊
券商：好的，七百號騰訊，買多少？
客戶：一千股，市價買入
券商：確認一下，七百號騰訊，買入一千股，市價，對嗎？
客戶：對的，謝謝""",
            outputs=[conversation_input],
        )
        
        example_2.click(
            fn=lambda: """客戶：早晨，我想問下小米同比亞迪今日走勢
券商：你好！小米一八一零今日升咗2%，比亞迪二一一一跌咗1%
客戶：咁我想沽五百股比亞迪，再入一千股小米
券商：好的，確認一下：沽出比亞迪二一一一五百股，買入小米一八一零一千股，啱唔啱？
客戶：啱，就咁做""",
            outputs=[conversation_input],
        )
        
        example_3.click(
            fn=lambda: """客戶：我想買招商局置地
券商：招商局置地，係一百一三八號？
客戶：係呀
券商：買幾多？
客戶：五百股
券商：確認：一百一三八號招商局置地，買入五百股
客戶：正確

註：這裡「一百一三八」是語音識別錯誤，應該是「一八一三八」(18138)""",
            outputs=[conversation_input],
        )
        
        gr.Markdown(
            """
            ---
            ### 📌 使用說明 (Instructions)
            
            1. **輸入對話**: 在左側文本框中輸入或貼上對話記錄
            2. **選擇模型**: 從下拉菜單選擇 LLM 模型（建議使用 qwen3:32b 或 deepseek-r1:32b）
            3. **調整參數**: 可選調整 Temperature（建議 0.1-0.3 以獲得更穩定的結果）
            4. **開始提取**: 點擊「開始提取股票資訊」按鈕
            5. **查看結果**: 右側會顯示格式化的結果和原始 JSON 數據
            
            ### 🎯 功能特點 (Features)
            
            - ✅ **結構化輸出**: 使用 Pydantic 保證數據格式一致
            - 🔍 **智能識別**: 自動識別股票代號和名稱
            - 🛠️ **誤差修正**: 可以識別並修正 Speech-to-Text 錯誤
            - 📊 **置信度評估**: 每個識別結果都有置信度評分
            - 🔄 **多模型支持**: 可以選擇不同的 LLM 模型進行比較
            """
        )
    
    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Launch the Gradio interface"""
    demo = create_stock_extractor_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()

