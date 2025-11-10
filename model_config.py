"""
Centralized model configuration for the project.
This file contains model options used across different modules.
"""

# Common model options for Ollama LLM
MODEL_OPTIONS = [
    "qwen3:14b",
    "qwen3:32b",
    "qwen3:8b",
    "gpt-oss:20b",
    "gemma3:27b",
    "deepseek-r1:32b",
]

# Default model selection
DEFAULT_MODEL = MODEL_OPTIONS[0]

# Default Ollama URL
DEFAULT_OLLAMA_URL = "http://localhost:11434"

