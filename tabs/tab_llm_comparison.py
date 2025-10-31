"""
Tab 7: LLM Comparison
Compare multiple LLM models simultaneously
"""

import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "你是一位精通粵語以及香港股市的分析師。請用繁體中文回應，"
    "並從下方對話中判斷誰是券商、誰是客戶，整理最終下單（股票代號、買/賣、價格、數量），"
)


def analyze_single_model(
    model: str,
    prompt: str,
    ollama_url: str,
    system_message: str,
    temperature: float,
) -> Tuple[str, str, float]:
    """
    Analyze text with a single LLM model
    
    Returns:
        tuple: (model_name, response_text, elapsed_time)
    """
    start_time = time.time()
    
    try:
        # Initialize the LLM
        chat_llm = ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=temperature,
        )
        
        # Prepare messages
        messages = [
            ("system", system_message),
            ("human", prompt),
        ]
        
        # Get response
        resp = chat_llm.invoke(messages)
        
        # Extract content
        try:
            response_content = getattr(resp, "content", str(resp))
        except Exception:
            response_content = str(resp)
        
        elapsed_time = time.time() - start_time
        
        return model, response_content, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return model, error_msg, elapsed_time


def compare_models(
    prompt_text: str,
    prompt_file,
    selected_models: List[str],
    ollama_url: str,
    system_message: str,
    temperature: float,
) -> Tuple[str, Dict[str, Tuple[str, float]]]:
    """
    Analyze text with multiple LLM models in parallel
    
    Returns:
        tuple: (status_message, dict of {model_name: (response, elapsed_time)})
    """
    try:
        # Determine the prompt source
        final_prompt = None
        
        if prompt_file is not None:
            # Read from uploaded file
            try:
                file_path = Path(prompt_file.name)
                final_prompt = file_path.read_text(encoding="utf-8")
                status = f"✓ Loaded prompt from file: {file_path.name}\n"
            except Exception as e:
                return f"❌ Error reading file: {str(e)}", {}
        elif prompt_text and prompt_text.strip():
            # Use text input
            final_prompt = prompt_text.strip()
            status = "✓ Using text input\n"
        else:
            return "❌ Error: Please provide either text input or upload a file", {}
        
        # Validate inputs
        if not selected_models or len(selected_models) == 0:
            return "❌ Error: Please select at least one model", {}
        
        if not ollama_url or not ollama_url.strip():
            return "❌ Error: Please specify Ollama URL", {}
        
        status += f"✓ Running {len(selected_models)} model(s) in parallel...\n"
        status += f"✓ Ollama URL: {ollama_url}\n"
        status += f"✓ Temperature: {temperature}\n"
        status += f"✓ Models: {', '.join(selected_models)}\n"
        status += "\n" + "="*50 + "\n"
        
        # Run models in parallel
        results = {}
        total_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(
                    analyze_single_model,
                    model,
                    final_prompt,
                    ollama_url,
                    system_message,
                    temperature
                ): model for model in selected_models
            }
            
            # Process completed tasks
            for future in as_completed(future_to_model):
                model_name, response, elapsed = future.result()
                results[model_name] = (response, elapsed)
                status += f"✓ {model_name} completed in {elapsed:.2f}s\n"
        
        total_elapsed = time.time() - total_start_time
        status += "="*50 + "\n"
        status += f"✓ All models completed in {total_elapsed:.2f}s\n"
        status += f"✓ Average time per model: {total_elapsed/len(selected_models):.2f}s\n"
        
        return status, results
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, {}


def format_comparison_results(results: Dict[str, Tuple[str, float]]) -> List[Tuple[str, str, str]]:
    """
    Format results for display in comparison boxes
    
    Returns:
        list: List of tuples (model_name, time_info, response), one per model
    """
    if not results:
        return []
    
    formatted = []
    for model, (response, elapsed) in results.items():
        model_name = f"🤖 {model}"
        time_info = f"⏱️ {elapsed:.2f} 秒"
        formatted.append((model_name, time_info, response))
    
    return formatted


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


def create_llm_comparison_tab():
    """Create and return the LLM Comparison tab"""
    with gr.Tab("7️⃣ LLM Comparison"):
        gr.Markdown("### Compare multiple LLM models simultaneously")
        gr.Markdown("*Run multiple models in parallel and compare their responses*")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Input Settings")
                
                with gr.Tab("文本輸入"):
                    comp_prompt_textbox = gr.Textbox(
                        label="對話記錄",
                        placeholder="請輸入或粘貼電話對話記錄...",
                        lines=12,
                        value=load_example_file(),
                    )
                
                with gr.Tab("文件上傳"):
                    comp_prompt_file = gr.File(
                        label="上傳對話記錄文件 (.txt, .json)",
                        file_types=[".txt", ".json"],
                    )
                    gr.Markdown("*上傳文件將優先於文本輸入*")
                
                gr.Markdown("#### LLM Settings")
                
                comp_model_checkboxes = gr.CheckboxGroup(
                    choices=MODEL_OPTIONS,
                    value=[MODEL_OPTIONS[0], MODEL_OPTIONS[1]],
                    label="選擇模型 (可選多個)",
                    info="選擇要同時運行的模型",
                )
                
                comp_temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                )
                
                comp_ollama_url = gr.Textbox(
                    label="Ollama URL",
                    value=DEFAULT_OLLAMA_URL,
                    placeholder="http://localhost:11434",
                )
                
                comp_system_message = gr.Textbox(
                    label="系統訊息 (System Message)",
                    value=DEFAULT_SYSTEM_MESSAGE,
                    lines=3,
                )
                
                comp_compare_btn = gr.Button("🚀 開始比較", variant="primary", size="lg")
                
                comp_status_box = gr.Textbox(
                    label="執行狀態",
                    lines=12,
                    interactive=False,
                )
            
            with gr.Column(scale=2):
                gr.Markdown("#### Comparison Results")
                
                # Store results in state
                comp_results_state = gr.State({})
                
                # Create result boxes for up to 6 models
                comp_result_boxes = []
                
                for i in range(6):
                    with gr.Row(visible=False) as row:
                        with gr.Column():
                            model_label = gr.Markdown(f"", visible=True)
                            time_label = gr.Markdown(f"", visible=True)
                            result_text = gr.Textbox(
                                label="",
                                lines=12,
                                interactive=False,
                                show_label=False,
                                show_copy_button=True,
                            )
                    comp_result_boxes.append((row, model_label, time_label, result_text))
        
        # Helper function to update UI
        def update_comparison_ui(
            prompt_text,
            prompt_file,
            selected_models,
            ollama_url,
            system_message,
            temperature,
        ):
            status, results = compare_models(
                prompt_text,
                prompt_file,
                selected_models,
                ollama_url,
                system_message,
                temperature,
            )
            
            formatted_results = format_comparison_results(results)
            
            # Prepare outputs for all result boxes
            outputs = [status, results]
            
            # Update each result box
            for i in range(6):
                if i < len(formatted_results):
                    model_name, time_info, response = formatted_results[i]
                    outputs.extend([
                        gr.Row(visible=True),      # row visibility
                        model_name,                  # model label
                        time_info,                   # time label
                        response,                    # result text
                    ])
                else:
                    # Hide unused boxes
                    outputs.extend([
                        gr.Row(visible=False),       # row visibility
                        "",                          # model label
                        "",                          # time label
                        "",                          # result text
                    ])
            
            return outputs
        
        # Connect the button
        comp_outputs = [comp_status_box, comp_results_state]
        for row, model_label, time_label, result_text in comp_result_boxes:
            comp_outputs.extend([row, model_label, time_label, result_text])
        
        comp_compare_btn.click(
            fn=update_comparison_ui,
            inputs=[
                comp_prompt_textbox,
                comp_prompt_file,
                comp_model_checkboxes,
                comp_ollama_url,
                comp_system_message,
                comp_temperature_slider,
            ],
            outputs=comp_outputs,
        )

