# Simple program to test Ollama LLM with LangChain

from langchain_ollama import ChatOllama
import argparse
from pathlib import Path

default_model = "gpt-oss:20b"
# model = "gemma3:270m"
# model = "qwen2.5:7b-instruct"
# model = "qwen3:30b"

default_ollama_url = "http://192.168.61.2:11434"
# default_ollama_url = "http://127.0.0.1:11434"

def main(prompt_text: str | None = None, model: str = default_model, ollama_url: str = default_ollama_url):
    print(f"\nInitializing Ollama Chat LLM (model: {model})...")
    print(f"Connecting to: {ollama_url}")
    print(f"Prompt: {prompt_text}")
    # Use the chat interface so we can provide a system message
    chat_llm = ChatOllama(
        model=model,
        base_url=ollama_url,
        temperature=0.7,
    )
    
    print(f"\nPrompt: {prompt_text}")
    # Prepare a system message (instructions for the assistant) and the user message
    system_message = (
        "你是一位精通粵語以及香港股市的分析師。請用繁體中文回應，"
        "並從下方對話中判斷誰是券商、誰是客戶，整理最終下單（股票代號、買/賣、價格、數量），"
    )

    messages = [
        ("system", system_message),
        ("human", prompt_text),
    ]

    print("\nResponse:")
    resp = chat_llm.invoke(messages)
    # ChatOllama returns a LangChain ChatResponse-like object; print its content
    try:
        print(getattr(resp, "content", resp))
    except Exception:
        print(resp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chat analysis against an Ollama model. If no prompt is given, the embedded example prompt is used.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt", help="Prompt string to send to the model")
    group.add_argument("--prompt-file", help="Path to a file containing the prompt text")
    parser.add_argument("--model", help="Ollama model name", default=default_model)
    parser.add_argument("--url", help="Ollama base URL", default=default_ollama_url)
    args = parser.parse_args()

    prompt_text = None
    if args.prompt:
        prompt_text = args.prompt
    elif args.prompt_file:
        p = Path(args.prompt_file)
        if p.exists():
            prompt_text = p.read_text(encoding="utf-8")
        else:
            print(f"Prompt file not found: {args.prompt_file}")
            raise SystemExit(2)
    else:
        print(f"Prompt or Prompt file not found")
        raise SystemExit(2)

    try:
        main(prompt_text, args.model, args.url)
    except Exception as e:
        print(f"\nError: {e}")


