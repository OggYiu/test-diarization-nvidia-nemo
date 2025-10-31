"""
Tab: Transaction Analysis
Analyze two transcriptions to identify stock transactions with confidence scoring
"""

import traceback
from typing import Literal, Optional
import gradio as gr

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


# Common model options
MODEL_OPTIONS = [
    "qwen3:32b",
    "gpt-oss:20b",
    "gemma3-27b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
]

DEFAULT_MODEL = MODEL_OPTIONS[0]
DEFAULT_OLLAMA_URL = "http://localhost:11434"


# Pydantic models for structured transaction output
class Transaction(BaseModel):
    """Represents a single transaction"""
    
    transaction_type: Literal["buy", "sell", "queue"] = Field(
        description="The type of transaction identified: buy, sell, or queue"
    )
    
    confidence_score: float = Field(
        ge=0.0, 
        le=2.0,
        description="Confidence score from 0 to 2. 0=not sure at all, 1=moderately confident, 2=very confident"
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


DEFAULT_SYSTEM_MESSAGE = """你是一位精通粵語的香港股市分析師，專門分析電話錄音中的股票交易。

你的任務是：
1. 比較兩個不同STT模型生成的轉錄文字
2. **重要：檢查股票參考資料中列出的所有股票**，識別對話中是否提及這些股票
3. 識別對話中的所有股票交易（買入buy、賣出sell、排隊queue）
4. 對話中可能有多個交易、一個交易、或沒有交易
5. 為每個交易評估置信度（0-2分）：
   - **0分：完全不確定** - 只是討論、沒有明確下單意圖
   - **1分：有一定證據但不完全確定** - 提到交易但沒有完整確認流程
   - **2分：非常確定有交易發生** - 券商重複下單資料，客戶明確確認
6. 提取每個交易的細節（股票代號、股票名稱、數量、價格等）
7. **使用股票參考資料來匹配和驗證對話中提到的股票代號和名稱**

# 判斷準則
- 券商必須重複下單資料讓客戶確認，才算是真正的下單（confidence_score: 2）
- 如果有提到交易細節但缺少確認步驟（confidence_score: 1）
- 如果只是討論而沒有下單意圖（confidence_score: 0，或不列為交易）
- 一個對話中可能包含多個不同的交易
- 兩個轉錄文字可能有差異，請綜合判斷
- **仔細對照股票參考資料中的所有股票，確保沒有遺漏任何提及的股票**

# 粵語術語和簡稱
- 轮 = 窩輪
- 沽/孤 = 賣出
- 買入/入 = 買入

# 常見STT誤差
- 「百」可能是「八」：例如「一百一三八」應該是「一八一三八」（18138）
- 使用股票參考資料來修正可能的STT誤差

# 輸出格式
**必須**返回有效的JSON格式，嚴格遵守以下結構：

{
  "transactions": [
    {
      "transaction_type": "buy",  // 必須是 "buy", "sell", 或 "queue"
      "confidence_score": 2.0,    // 必須是數字 0.0-2.0，不能是字符串
      "stock_code": "0700",
      "stock_name": "騰訊",
      "quantity": "1000",
      "price": "350",
      "explanation": "客戶要求買入騰訊0700，券商重複確認了股票代號、數量和價格，客戶明確確認，因此置信度為2分"
    }
  ],
  "transcription_comparison": "兩個轉錄的比較說明...",
  "overall_summary": "整體摘要...（必須包含對股票參考資料中所有股票的檢查結果）"
}

**重要提示：**
- confidence_score 必須是數字（float），不能是字符串
- 如果有明確的交易確認，confidence_score 應該是 1.5 或 2.0，不要總是給 0
- explanation 字段必須詳細說明為什麼給這個置信度分數
- overall_summary 必須說明是否檢查了股票參考資料中的所有股票
"""


def analyze_transactions(
    transcription1: str,
    transcription2: str,
    stock_reference: str,
    model: str,
    ollama_url: str,
    system_message: str,
    temperature: float,
) -> tuple[str, str]:
    """
    Analyze two transcriptions to identify transactions
    
    Returns:
        tuple: (summary_result, json_result)
    """
    try:
        # Validate inputs
        if not transcription1.strip() and not transcription2.strip():
            error_msg = "❌ 錯誤：請至少提供一個轉錄文字"
            return (error_msg, "")
        
        if not model or not model.strip():
            error_msg = "❌ 錯誤：請指定模型名稱"
            return (error_msg, "")
        
        if not ollama_url or not ollama_url.strip():
            error_msg = "❌ 錯誤：請指定 Ollama URL"
            return (error_msg, "")
        
        # Build the prompt
        stock_ref_text = stock_reference.strip() if stock_reference.strip() else "（無提供）"
        prompt = f"""請分析以下兩個STT模型生成的轉錄文字，識別是否有股票交易發生。

## 轉錄文字 1：
{transcription1}

## 轉錄文字 2：
{transcription2}

## 股票參考資料：
{stock_ref_text}

**重要任務：**
1. 請仔細檢查對話中是否提及股票參考資料中列出的**所有股票**
2. 對於每個提及的股票，判斷是否有交易發生（買入/賣出/排隊）
3. 使用股票參考資料來驗證和修正轉錄文字中可能的股票代號或名稱錯誤
4. 在 overall_summary 中明確說明檢查了哪些股票，哪些股票在對話中被提及，哪些沒有被提及

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
        
        # Try to parse as structured output
        try:
            import json
            result_dict = json.loads(response_content)
            
            # Debug: Print raw JSON response
            print("="*60)
            print("🔍 DEBUG: Raw LLM JSON Response:")
            print(json.dumps(result_dict, indent=2, ensure_ascii=False))
            print("="*60)
            
            # Extract the new structure
            transactions = result_dict.get("transactions", [])
            comparison = result_dict.get("transcription_comparison", "")
            overall_summary = result_dict.get("overall_summary", "")
            
            # Create formatted summary result for all transactions
            summary_result = f"""📊 交易分析結果
{'='*50}

📋 總共識別到 {len(transactions)} 個交易

"""
            
            if len(transactions) == 0:
                summary_result += "❌ 沒有識別到任何交易\n\n"
            else:
                for idx, tx in enumerate(transactions, 1):
                    tx_type = tx.get("transaction_type", "none")
                    tx_conf = tx.get("confidence_score", 0.0)
                    tx_code = tx.get("stock_code", "") or "N/A"
                    tx_name = tx.get("stock_name", "") or "N/A"
                    tx_qty = tx.get("quantity", "") or "N/A"
                    tx_price = tx.get("price", "") or "N/A"
                    tx_exp = tx.get("explanation", "")
                    
                    summary_result += f"""{'─'*50}
交易 #{idx}
{'─'*50}
🔖 交易類型: {tx_type}
⭐ 置信度分數: {tx_conf} / 2.0
📈 股票代號: {tx_code}
🏢 股票名稱: {tx_name}
🔢 數量: {tx_qty}
💰 價格: {tx_price}

📝 分析說明:
{tx_exp}

"""
            
            summary_result += f"""{'='*50}
🔄 轉錄比較:
{comparison}

{'='*50}
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


def create_transaction_analysis_tab():
    """Create and return the Transaction Analysis tab"""
    with gr.Tab("📊 Transaction Analysis"):
        gr.Markdown(
            """
            ### 交易分析 - 比較兩個STT轉錄並識別交易
            使用Pydantic結構化輸出識別所有股票交易（買入/賣出/排隊）並評估置信度（0-2分）
            
            **支持功能：**
            - ✅ 識別多個交易（一個對話中可能有多筆交易）
            - ✅ 識別單個交易
            - ✅ 識別無交易的對話
            - ✅ 檢查並分析股票參考資料中的所有股票
            - ✅ 使用股票參考資料來驗證和修正STT誤差
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 輸入設定")
                
                transcription1_box = gr.Textbox(
                    label="轉錄文字 1 (STT Model 1)",
                    placeholder="請輸入第一個STT模型的轉錄結果...",
                    lines=10,
                )
                
                transcription2_box = gr.Textbox(
                    label="轉錄文字 2 (STT Model 2)",
                    placeholder="請輸入第二個STT模型的轉錄結果...",
                    lines=10,
                )
                
                stock_reference_box = gr.Textbox(
                    label="股票參考資料 (Stock References)",
                    placeholder="例如：\n騰訊 0700\n阿里巴巴 9988\n滙豐 0005",
                    lines=5,
                    info="輸入可能在對話中出現的股票名稱和代號。LLM將檢查並分析所有列出的股票。",
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
                    lines=8,
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
                    lines=15,
                    interactive=False,
                    show_copy_button=True,
                )
                
                # JSON Result Textbox (raw output)
                json_result_box = gr.Textbox(
                    label="📋 Pydantic JSON 輸出 (JSON Output)",
                    lines=15,
                    interactive=False,
                    show_copy_button=True,
                )
        
        # Connect the button
        analyze_btn.click(
            fn=analyze_transactions,
            inputs=[
                transcription1_box,
                transcription2_box,
                stock_reference_box,
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

