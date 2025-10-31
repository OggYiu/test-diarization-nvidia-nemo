"""
Tab 4: LLM Analysis
Analyze transcriptions using Large Language Models
"""

import csv
import traceback
from pathlib import Path
import gradio as gr

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

DEFAULT_SYSTEM_MESSAGE = (
    f"""你是一位精通粵語的香港股市的分析師，現在你的工作是電話錄音分析專員。你將會分析電話錄音的文字版本，但由於Speech To Text 技術的誤差，有很多的文字會出現誤認使對話內容難以理解，要用你的聽聰明才智去想像原本的對話內容。

# 請回應以下有關問題
- 請用繁體中文回應，並從下方對話中判斷誰是券商、誰是客戶，整理最終下單（股票代號、買/賣、價格、數量）。
- 留意在對話中，可能只是券商跟客戶的討論，並沒有任何交易，若有交易，下單的數量有可能多於一單。

# 請用下列的資料作判別準則
- 留意客戶下單時候，券商一定會將下單的資料重覆一次讓客戶確定，若對話中沒有確定的對話，很可能不是下單。
- 嘗試在對話找出股票號碼和股票名稱，並列出可能的股票號碼和股票名稱。

# 以下是簡稱和術語，可以查看一下去協助你了解對話內容:
- 簡稱: 轮，全寫: 窩輪
- 簡稱: 沽/孤，全寫: 賣出

# 以下是常見的Speech To Text 技術的誤差:
- 誤認: 百，正寫: 八，例子: 一百一三八 -> 一八一三八, 即是 18138
    """

#     """
#     你是一位精通粵語的香港股市分析師，現在你的角色是電話錄音分析專員。你將分析電話錄音的文字轉錄版本（來自Speech-to-Text技術），但由於轉錄技術的誤差，文字中可能出現大量誤認詞彙，導致對話內容難以理解。你需要運用你的專業知識和邏輯推理，推斷並還原原本的對話意圖，尤其是涉及股票相關的部分。

# 你的主要任務是：從轉錄文字中找出所有提及的股票名稱和股票代碼。如果文字有誤認，請修正它們，並解釋你的推理過程。最終輸出應包括：
# - 修正後的股票名稱和代碼列表。
# - 對每個修正的簡要解釋。


# ### 以下是簡稱和術語，可以查看一下去協助你了解對話內容:
# - 簡稱: 轮，全寫: 窩輪
# - 簡稱: 沽/孤，全寫: 賣出

# ### 常見的Speech-to-Text誤差類型（基於粵語發音相似性）：
# - 誤認示例：將「八」誤認爲「百」。  
#   例子：轉錄文字爲「一百百七」，正確應爲「一八八七」，即股票代碼1887。

# 在分析時，請考慮上下文推斷可能的修正。如果無法確定，請標註爲「不確定」並提供備選可能性。
# """
)


def parse_metadata(metadata_text: str) -> dict[str, str]:
    """
    Parse metadata from pasted text format.
    
    Expected format:
        Broker Name: Dickson Lau
        Broker Id: 0489
        Client Number: 97501167
        Client Name: CHAN CHO WING and CHAN MAN LEE
        Client Id: P77751
        UTC: 2025-10-10T01:45:10
        HKT: 2025-10-10T09:45:10
    
    Returns:
        dict: Dictionary with keys: broker_name, broker_id, client_number, 
              client_id, client_name, utc_time, hkt_time
    """
    result = {
        "broker_name": "",
        "broker_id": "",
        "client_number": "",
        "client_id": "",
        "client_name": "",
        "utc_time": "",
        "hkt_time": ""
    }
    
    if not metadata_text or not metadata_text.strip():
        return result
    
    # Parse line by line
    lines = metadata_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if ':' not in line:
            continue
            
        # Split only on first colon to handle values that contain colons (like times)
        key, value = line.split(':', 1)
        key = key.strip().lower()
        value = value.strip()
        
        # Map the keys to result dictionary
        if 'broker name' in key:
            result['broker_name'] = value
        elif 'broker id' in key:
            result['broker_id'] = value
        elif 'client number' in key:
            result['client_number'] = value
        elif 'client id' in key:
            result['client_id'] = value
        elif 'client name' in key:
            result['client_name'] = value
        elif key == 'utc':
            result['utc_time'] = value
        elif key == 'hkt':
            result['hkt_time'] = value
    
    return result


def analyze_with_llm(
    prompt_text: str,
    prompt_file,
    model: str,
    ollama_url: str,
    system_message: str,
    temperature: float,
    metadata_text: str = "",
) -> str:
    """
    Analyze text with LLM
    
    Returns:
        str: response_text
    """
    try:
        # Determine the prompt source
        final_prompt = None
        
        if prompt_file is not None:
            # Read from uploaded file
            try:
                file_path = Path(prompt_file.name)
                final_prompt = file_path.read_text(encoding="utf-8")
                status = f"✓ Loaded prompt from file: {file_path.name}"
            except Exception as e:
                return f"❌ Error reading file: {str(e)}"
        elif prompt_text and prompt_text.strip():
            # Use text input
            final_prompt = prompt_text.strip()
            status = "✓ Using text input"
        else:
            return "❌ Error: Please provide either text input or upload a file"
        
        # Validate inputs
        if not model or not model.strip():
            return "❌ Error: Please specify a model name"
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL"
        
        # Parse metadata from the pasted text
        metadata_dict = parse_metadata(metadata_text)
        
        # Build metadata context if any fields are provided
        metadata_lines = []
        if metadata_dict['broker_name']:
            metadata_lines.append(f"Broker Name: {metadata_dict['broker_name']}")
        if metadata_dict['broker_id']:
            metadata_lines.append(f"Broker Id: {metadata_dict['broker_id']}")
        if metadata_dict['client_number']:
            metadata_lines.append(f"Client Number: {metadata_dict['client_number']}")
        if metadata_dict['client_name']:
            metadata_lines.append(f"Client Name: {metadata_dict['client_name']}")
        if metadata_dict['client_id']:
            metadata_lines.append(f"Client Id: {metadata_dict['client_id']}")
        if metadata_dict['utc_time']:
            metadata_lines.append(f"UTC: {metadata_dict['utc_time']}")
        if metadata_dict['hkt_time']:
            metadata_lines.append(f"HKT: {metadata_dict['hkt_time']}")
        
        # Prepend metadata to system message if available
        final_system_message = system_message
        if metadata_lines:
            metadata_context = "\n".join(metadata_lines)
            final_system_message = f"{system_message}\n\n以下是對話的資料背景, 可能會幫助你分析對話:\n{metadata_context}"
        
        print(f"{'*' * 30} final_system_message: {final_system_message} {'*' * 30}")
        
        # Initialize the LLM
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=temperature,
        )
        
        # Prepare messages
        messages = [
            ("system", final_system_message),
            ("human", final_prompt),
        ]
        
        # Get response
        resp = chat_llm.invoke(messages)
        
        # Extract content
        try:
            response_content = getattr(resp, "content", str(resp))
        except Exception:
            response_content = str(resp)
        
        return response_content
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg



def create_llm_analysis_tab():
    """Create and return the LLM Analysis tab"""
    with gr.Tab("4️⃣ LLM Analysis"):
        gr.Markdown("### Analyze transcriptions using Large Language Models")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("#### Input Settings")
                
                with gr.Tab("文本輸入"):
                    llm_prompt_textbox = gr.Textbox(
                        label="對話記錄",
                        placeholder="",
                        lines=15,
                        value="",
                    )
                
                with gr.Tab("文件上傳"):
                    llm_prompt_file = gr.File(
                        label="上傳對話記錄文件 (.txt, .json)",
                        file_types=[".txt", ".json"],
                    )
                    gr.Markdown("*上傳文件將優先於文本輸入*")
                
                gr.Markdown("#### Context Information (Metadata)")
                gr.Markdown("*This information will be included in the system prompt*")
                
                llm_metadata_textbox = gr.Textbox(
                    label="Paste Context Information",
                    placeholder="""""",
                    lines=8,
                    info="Paste all context information at once in the format shown above"
                )
                
                gr.Markdown("#### LLM Settings")
                
                with gr.Row():
                    llm_model_dropdown = gr.Dropdown(
                        choices=MODEL_OPTIONS,
                        value=DEFAULT_MODEL,
                        label="模型",
                        allow_custom_value=True,
                    )
                    llm_temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                
                llm_ollama_url = gr.Textbox(
                    label="Ollama URL",
                    value=DEFAULT_OLLAMA_URL,
                    placeholder="http://localhost:11434",
                )
                
                llm_system_message = gr.Textbox(
                    label="系統訊息 (System Message)",
                    value=DEFAULT_SYSTEM_MESSAGE,
                    lines=15,
                )
                
                llm_analyze_btn = gr.Button("🚀 開始分析", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("#### Analysis Results")
                
                llm_response_box = gr.Textbox(
                    label="LLM 回應",
                    lines=20,
                    interactive=False,
                )
        
        llm_analyze_btn.click(
            fn=analyze_with_llm,
            inputs=[
                llm_prompt_textbox,
                llm_prompt_file,
                llm_model_dropdown,
                llm_ollama_url,
                llm_system_message,
                llm_temperature_slider,
                llm_metadata_textbox,
            ],
            outputs=[llm_response_box],
        )

