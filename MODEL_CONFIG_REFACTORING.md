# Model Configuration Refactoring

## Summary

Centralized all `MODEL_OPTIONS` definitions across the project into a single configuration file to eliminate code duplication and make it easier to manage model options.

## Changes Made

### New File Created
- **`model_config.py`**: Centralized configuration file containing:
  - `MODEL_OPTIONS`: List of all available Ollama models
  - `DEFAULT_MODEL`: Default model selection (first in the list)
  - `DEFAULT_OLLAMA_URL`: Default Ollama server URL

### Files Updated (8 total)

#### Tab Files (in `tabs/` directory)
1. `tab_transaction_analysis.py`
2. `tab_llm_analysis.py`
3. `tab_llm_comparison.py`
4. `tab_transcription_merger.py`

#### Root-level GUI Files
5. `transcription_merger.py`
6. `llm_analysis_gui.py`
7. `stock_extractor_gui.py`
8. `llm_comparison_gui.py`

### What Changed in Each File

**Before:**
```python
# Common model options
MODEL_OPTIONS = [
    "qwen3:32b",
    "gpt-oss:20b",
    "gemma3:27b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
]

DEFAULT_MODEL = MODEL_OPTIONS[0]
DEFAULT_OLLAMA_URL = "http://localhost:11434"
```

**After:**
```python
# Import centralized model configuration
from model_config import MODEL_OPTIONS, DEFAULT_MODEL, DEFAULT_OLLAMA_URL
```

## Benefits

1. **Single Source of Truth**: Model options are defined in one place (`model_config.py`)
2. **Easy Maintenance**: To add or remove a model, just edit one file
3. **Consistency**: All modules now use the same model list
4. **Reduced Code Duplication**: Eliminated 8 duplicate definitions
5. **Flexibility**: Individual files can still override defaults if needed (e.g., `llm_analysis_gui.py` has a custom `DEFAULT_OLLAMA_URL`)

## How to Add/Remove Models

Simply edit `model_config.py`:

```python
MODEL_OPTIONS = [
    "qwen3:32b",
    "gpt-oss:20b",
    "gemma3:27b",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
    "qwen2.5:72b",
    # Add new models here
    "new-model:size",
]
```

All modules importing from `model_config` will automatically use the updated list.

## Notes

- The centralized config includes all models that were previously defined across different files
- Some files had slightly different model lists; the new config includes the superset of all models
- Files that need custom URLs can still override `DEFAULT_OLLAMA_URL` locally (see `llm_analysis_gui.py` as an example)

