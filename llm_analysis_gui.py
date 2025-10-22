"""
Gradio GUI for LLM Analysis using Ollama
Analyzes phone call transcriptions using LangChain and Ollama
"""

import gradio as gr
from langchain_ollama import ChatOllama
from pathlib import Path
import traceback

# Common model options
MODEL_OPTIONS = [
    "qwen2.5vl:32b",
    "gpt-oss:20b",
    "qwen3:30b",
    "gemma3-27b",
]

# Default configuration
DEFAULT_MODEL = MODEL_OPTIONS[0]
DEFAULT_OLLAMA_URL = "http://192.168.61.2:11434"
DEFAULT_SYSTEM_MESSAGE = (
    "你是一位精通粵語以及香港股市的分析師。請用繁體中文回應，"
    "並從下方對話中判斷誰是券商、誰是客戶，整理最終下單（股票代號、買/賣、價格、數量），"
)


def analyze_with_llm(
    prompt_text: str,
    prompt_file,
    model: str,
    ollama_url: str,
    system_message: str,
    temperature: float,
) -> tuple[str, str]:
    """
    Analyze text with LLM
    
    Returns:
        tuple: (status_message, response_text)
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
                return f"❌ Error reading file: {str(e)}", ""
        elif prompt_text and prompt_text.strip():
            # Use text input
            final_prompt = prompt_text.strip()
            status = "✓ Using text input"
        else:
            return "❌ Error: Please provide either text input or upload a file", ""
        
        # Validate inputs
        if not model or not model.strip():
            return "❌ Error: Please specify a model name", ""
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL", ""
        
        # Initialize the LLM
        status += f"\n✓ Connecting to Ollama at: {ollama_url}"
        status += f"\n✓ Using model: {model}"
        status += f"\n✓ Temperature: {temperature}"
        
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=temperature,
        )
        
        # Prepare messages
        messages = [
            ("system", system_message),
            ("human", final_prompt),
        ]
        
        status += "\n✓ Sending request to LLM..."
        
        # Get response
        resp = chat_llm.invoke(messages)
        
        # Extract content
        try:
            response_content = getattr(resp, "content", str(resp))
        except Exception:
            response_content = str(resp)
        
        status += "\n✓ Analysis complete!"
        
        return status, response_content
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, ""


def load_example_file():
    """Load an example transcription file if available"""
    example_paths = [
        Path("demo/transcriptions/conversation.txt"),
        Path("output/transcriptions/conversation.txt"),
    ]
    
    for path in example_paths:
        if path.exists():
            return path.read_text(encoding="utf-8")
    
    return "請輸入電話對話記錄文本，或上傳文件。"


def create_ui():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="LLM Phone Call Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📞 LLM Phone Call Analyzer
            使用 Ollama 和 LangChain 分析電話對話記錄
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 輸入設定")
                
                # Prompt input
                with gr.Tab("文本輸入"):
                    prompt_textbox = gr.Textbox(
                        label="對話記錄",
                        placeholder="請輸入或粘貼電話對話記錄...",
                        lines=15,
                        value=load_example_file(),
                    )
                
                with gr.Tab("文件上傳"):
                    prompt_file = gr.File(
                        label="上傳對話記錄文件 (.txt, .json)",
                        file_types=[".txt", ".json"],
                    )
                    gr.Markdown("*上傳文件將優先於文本輸入*")
                
                # Configuration
                gr.Markdown("### LLM 設定")
                
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
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                
                ollama_url = gr.Textbox(
                    label="Ollama URL",
                    value=DEFAULT_OLLAMA_URL,
                    placeholder="http://localhost:11434",
                )
                
                system_message = gr.Textbox(
                    label="系統訊息 (System Message)",
                    value=DEFAULT_SYSTEM_MESSAGE,
                    lines=3,
                )
                
                # Action button
                analyze_btn = gr.Button("🚀 開始分析", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 分析結果")
                
                status_box = gr.Textbox(
                    label="狀態",
                    lines=6,
                    interactive=False,
                )
                
                response_box = gr.Textbox(
                    label="LLM 回應",
                    lines=20,
                    interactive=False,
                )
        
        # Examples
        gr.Markdown("### 💡 提示")
        gr.Markdown(
            """
            - 支持從文本框直接輸入或上傳 .txt/.json 文件
            - 系統訊息可以自定義以適應不同的分析需求
            - Temperature 越高，回應越有創意；越低，回應越確定
            - 確保 Ollama 服務正在運行並且可以訪問
            """
        )
        
        # Connect the button
        analyze_btn.click(
            fn=analyze_with_llm,
            inputs=[
                prompt_textbox,
                prompt_file,
                model_dropdown,
                ollama_url,
                system_message,
                temperature_slider,
            ],
            outputs=[status_box, response_box],
        )
    
    return demo


def main():
    """Launch the Gradio app"""
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()

