import gradio as gr
import time

def streaming_text_generator():
    accumulated_text = ""
    for i in range(10):
        accumulated_text += f"Step {i+1} completed.\n"
        yield accumulated_text
        time.sleep(1) # Simulate some work

with gr.Blocks() as demo:
    output_textbox = gr.Textbox(label="Real-time Output")
    stream_button = gr.Button("Start Streaming")
    stream_button.click(streaming_text_generator, outputs=output_textbox)

demo.launch()