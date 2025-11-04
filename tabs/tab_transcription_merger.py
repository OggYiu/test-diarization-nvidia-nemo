"""
Tab 10: Transcription Merger
Merge two different STT transcriptions using LLM analysis
"""

import traceback
import gradio as gr
from langchain_ollama import ChatOllama

# Import centralized model configuration
from model_config import MODEL_OPTIONS, DEFAULT_MODEL, DEFAULT_OLLAMA_URL

# Default system message for transcription merging
DEFAULT_SYSTEM_MESSAGE = """你是一位專業的語音轉文字分析專家，精通粵語和繁體中文。

你的任務是分析和比較兩個不同的STT（語音轉文字）模型產生的轉錄文本，並輸出一個改進的、更準確的轉錄版本。 嘗試還原原本的對話內容。

# 重要指引：
1. **仔細比較兩個轉錄版本**：每個STT模型有不同的優勢，一個模型可能能夠準確識別某些詞語，而另一個模型可能在其他詞語上表現更好。

2. **識別差異和優勢**：
   - 找出兩個版本中不同的地方
   - 判斷哪個版本在特定詞語或短語上更準確
   - 考慮上下文和語意的連貫性

3. **合併最佳結果**：
   - 從每個轉錄中選擇最準確的部分
   - 確保最終結果語意通順、符合邏輯
   - 修正明顯的錯誤（如數字、專有名詞、股票代號等）

4. **處理常見STT錯誤**：
   - 粵語同音字誤認（如「八」與「百」、「沽」與「孤」）
   - 數字識別錯誤
   - 專有名詞（公司名稱、人名、股票名稱）
   - 語氣詞和連接詞

5. **輸出格式**：
   - 首先輸出改進後的完整轉錄文本
   - 然後提供一個簡短的分析說明，解釋主要的改進點和為何做出這些選擇

# 注意事項：
- 保持原對話的意思和語氣
- 使用繁體中文
- 如果兩個版本都不清楚，選擇最合理的解釋，並在分析中說明不確定性
- 重點關注關鍵信息的準確性（如股票交易相關的數字、代號、操作）
"""

# Example for Hong Kong stock trading context
STOCK_TRADING_SYSTEM_MESSAGE = """你是一位專業的語音轉文字分析專家，精通粵語和繁體中文，並且熟悉香港股市交易術語。

你的任務是分析和比較兩個不同的STT（語音轉文字）模型產生的轉錄文本，並輸出一個改進的、更準確的轉錄版本。

# 重要指引：
1. **仔細比較兩個轉錄版本**：每個STT模型有不同的優勢，一個模型可能能夠準確識別某些詞語，而另一個模型可能在其他詞語上表現更好。

2. **識別差異和優勢**：
   - 找出兩個版本中不同的地方
   - 判斷哪個版本在特定詞語或短語上更準確
   - 考慮上下文和語意的連貫性

3. **合併最佳結果**：
   - 從每個轉錄中選擇最準確的部分
   - 確保最終結果語意通順、符合邏輯
   - 特別注意股票相關資訊的準確性（代號、價格、數量、買/賣方向）

4. **處理常見STT錯誤**：
   - 粵語同音字誤認（如「八」與「百」、「沽」與「孤」）
   - 數字識別錯誤（股票代號常被誤認）
   - 專有名詞（公司名稱、股票名稱）
   - 交易術語（窩輪→輪、賣出→沽/孤）

5. **股市專業術語參考**：
   - 簡稱: 輪，全寫: 窩輪
   - 簡稱: 沽/孤，全寫: 賣出
   - 常見誤認: 「百」→「八」（例：一百一三八 應為 一八一三八，即18138）

6. **輸出格式**：
   - 首先輸出改進後的完整轉錄文本
   - 然後提供一個分析說明：
     * 主要改進點
     * 識別出的股票代號和相關交易資訊
     * 為何做出這些選擇
     * 任何不確定的地方

# 注意事項：
- 保持原對話的意思和語氣
- 使用繁體中文
- 券商會重複客戶的下單資訊以確認，注意識別這種確認對話
- 如果兩個版本都不清楚，選擇最合理的解釋，並在分析中說明不確定性
"""


def merge_transcriptions(
    transcription1: str,
    transcription2: str,
    model: str,
    ollama_url: str,
    system_message: str,
    temperature: float,
    label1: str = "STT模型 1",
    label2: str = "STT模型 2",
) -> str:
    """
    Merge two transcriptions using LLM analysis
    
    Args:
        transcription1: First transcription text
        transcription2: Second transcription text
        model: LLM model name
        ollama_url: Ollama server URL
        system_message: System prompt for the LLM
        temperature: Temperature for LLM generation
        label1: Label for first transcription
        label2: Label for second transcription
    
    Returns:
        str: Merged transcription with analysis
    """
    try:
        # Validate inputs
        if not transcription1 or not transcription1.strip():
            return "❌ 錯誤：請輸入第一個轉錄文本"
        
        if not transcription2 or not transcription2.strip():
            return "❌ 錯誤：請輸入第二個轉錄文本"
        
        if not model or not model.strip():
            return "❌ 錯誤：請選擇一個模型"
        
        if not ollama_url or not ollama_url.strip():
            return "❌ 錯誤：請輸入 Ollama URL"
        
        # Build the user prompt with both transcriptions
        user_prompt = f"""請分析以下兩個不同STT模型產生的轉錄文本，並輸出一個改進的版本：

## 【{label1}】的轉錄：
{transcription1}

## 【{label2}】的轉錄：
{transcription2}

請根據上述兩個版本，輸出：
1. 改進後的完整轉錄文本
2. 分析說明（解釋主要改進點和選擇理由）
"""
        
        print(f"\n{'='*60}")
        print(f"Starting transcription merge analysis...")
        print(f"Model: {model}")
        print(f"Ollama URL: {ollama_url}")
        print(f"Temperature: {temperature}")
        print(f"{'='*60}\n")
        
        # Initialize the LLM
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=temperature,
        )
        
        # Prepare messages
        messages = [
            ("system", system_message),
            ("human", user_prompt),
        ]
        
        print("Sending request to LLM...")
        
        # Get response
        resp = chat_llm.invoke(messages)
        
        # Extract content
        try:
            response_content = getattr(resp, "content", str(resp))
        except Exception:
            response_content = str(resp)
        
        print("✓ Received response from LLM\n")
        
        return response_content
        
    except Exception as e:
        error_msg = f"❌ 錯誤: {str(e)}\n\n追蹤訊息:\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


def create_transcription_merger_tab():
    """Create and return the Transcription Merger tab"""
    
    with gr.Tab("🔀 Transcription Merger"):
        gr.Markdown(
            """
            ### 🎯 合併兩個 STT 轉錄文本
            使用 LLM 分析並合併兩個不同 STT 模型的轉錄結果，創造更準確的文本。
            每個 STT 模型都有其優勢和弱點，這個工具會識別每個版本的優點並合併成更準確的最終版本。
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 📝 輸入兩個轉錄文本")
                
                with gr.Group():
                    label1_input = gr.Textbox(
                        label="第一個轉錄的標籤",
                        value="STT模型 1",
                        placeholder="例如: Whisper Large v3, SenseVoice, WSYue",
                        info="給這個轉錄一個識別名稱"
                    )
                    
                    transcription1_input = gr.Textbox(
                        label="第一個轉錄文本",
                        placeholder="在此貼上第一個 STT 模型的轉錄結果...",
                        lines=8,
                    )
                
                with gr.Group():
                    label2_input = gr.Textbox(
                        label="第二個轉錄的標籤",
                        value="STT模型 2",
                        placeholder="例如: Conformer CTC, FunASR, Paraformer",
                        info="給這個轉錄一個識別名稱"
                    )
                    
                    transcription2_input = gr.Textbox(
                        label="第二個轉錄文本",
                        placeholder="在此貼上第二個 STT 模型的轉錄結果...",
                        lines=8,
                    )
            
            with gr.Column(scale=1):
                gr.Markdown("#### ⚙️ LLM 設定")
                
                with gr.Group():
                    model_dropdown = gr.Dropdown(
                        choices=MODEL_OPTIONS,
                        value=DEFAULT_MODEL,
                        label="選擇 LLM 模型",
                        allow_custom_value=True,
                        info="選擇用於分析的語言模型"
                    )
                    
                    ollama_url_input = gr.Textbox(
                        label="Ollama URL",
                        value=DEFAULT_OLLAMA_URL,
                        placeholder="http://localhost:11434",
                        info="Ollama 服務器地址"
                    )
                    
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.3,
                        step=0.1,
                        label="Temperature",
                        info="較低的值會產生更一致的結果"
                    )
                
                gr.Markdown("#### 📋 系統提示詞")
                
                with gr.Accordion("選擇預設模板或自訂", open=False):
                    template_dropdown = gr.Dropdown(
                        choices=[
                            "通用粵語轉錄",
                            "香港股市交易對話",
                            "自訂",
                        ],
                        value="通用粵語轉錄",
                        label="系統提示詞模板",
                    )
                    
                    system_message_input = gr.Textbox(
                        label="系統提示詞 (System Message)",
                        value=DEFAULT_SYSTEM_MESSAGE,
                        lines=8,
                        info="指導 LLM 如何分析和合併轉錄文本"
                    )
                
                merge_btn = gr.Button("🚀 開始分析並合併", variant="primary", size="lg")
        
        gr.Markdown("#### 📊 合併結果")
        
        output_box = gr.Textbox(
            label="改進後的轉錄文本及分析",
            lines=15,
            interactive=True,
            show_copy_button=True,
        )
        
        # Event handlers
        def update_system_message(template):
            if template == "通用粵語轉錄":
                return DEFAULT_SYSTEM_MESSAGE
            elif template == "香港股市交易對話":
                return STOCK_TRADING_SYSTEM_MESSAGE
            else:
                return ""  # Custom - let user write their own
        
        template_dropdown.change(
            fn=update_system_message,
            inputs=[template_dropdown],
            outputs=[system_message_input],
        )
        
        merge_btn.click(
            fn=merge_transcriptions,
            inputs=[
                transcription1_input,
                transcription2_input,
                model_dropdown,
                ollama_url_input,
                system_message_input,
                temperature_slider,
                label1_input,
                label2_input,
            ],
            outputs=[output_box],
        )
        
        # Usage tips
        with gr.Accordion("💡 使用提示", open=False):
            gr.Markdown("""
            1. **準備轉錄文本**：從兩個不同的 STT 模型獲取同一音頻的轉錄結果
            2. **貼上文本**：將兩個轉錄分別貼到對應的文本框中
            3. **設定標籤**：為每個轉錄設定易於識別的名稱（如模型名稱）
            4. **選擇模型**：選擇一個 LLM 模型進行分析（推薦使用較大的模型如 32b 或以上）
            5. **選擇模板**：根據對話內容選擇合適的系統提示詞模板
            6. **調整溫度**：
               - 0.1-0.3：更保守，更接近原文
               - 0.4-0.7：平衡創意和準確性
               - 0.8-1.0：更有創意，但可能偏離原文
            7. **開始分析**：點擊按鈕，等待 LLM 分析並輸出改進的版本
            
            **注意**：分析時間取決於轉錄長度和所選模型，通常需要幾秒到幾十秒。
            """)

