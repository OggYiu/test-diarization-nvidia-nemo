# Tabs Module - Modular GUI Structure

This directory contains the modular implementation of the Unified Phone Call Analysis Suite. Each tab is in its own separate file for better organization and maintainability.

## Structure

```
tabs/
├── __init__.py                    # Module initialization and exports
├── tab_file_metadata.py           # Tab 8: File Metadata Extraction
├── tab_diarization.py             # Tab 1: Speaker Diarization
├── tab_chopper.py                 # Tab 2: Audio Chopper
├── tab_stt.py                     # Tab 3: Batch Speech-to-Text
├── tab_llm_analysis.py            # Tab 4: LLM Analysis
├── tab_speaker_separation.py      # Tab 5: Speaker Separation
├── tab_audio_enhancement.py       # Tab 6: Audio Enhancement
└── tab_llm_comparison.py          # Tab 7: LLM Comparison
```

## How It Works

Each tab file contains:
1. **Processing functions** - Business logic for that specific tab
2. **UI creation function** - A `create_*_tab()` function that builds the Gradio interface
3. **Event handlers** - Functions connected to buttons and interactions

The main `unified_gui.py` file simply imports all tab creation functions and assembles them into the final interface.

## Adding a New Tab

To add a new tab:

1. **Create a new file** in the `tabs/` directory (e.g., `tab_my_feature.py`)

2. **Implement your tab**:
```python
"""
Tab X: My Feature
Description of what this tab does
"""

import gradio as gr
# Import any other dependencies you need

def my_processing_function(input_data):
    """Process the input data"""
    # Your processing logic here
    return result

def create_my_feature_tab():
    """Create and return the My Feature tab"""
    with gr.Tab("X️⃣ My Feature"):
        gr.Markdown("### Description of my feature")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                input_component = gr.Textbox(label="Input")
                process_btn = gr.Button("Process", variant="primary")
                
            with gr.Column(scale=2):
                # Output components
                output_component = gr.Textbox(label="Output")
        
        # Connect the button
        process_btn.click(
            fn=my_processing_function,
            inputs=[input_component],
            outputs=[output_component]
        )
```

3. **Export in `__init__.py`**:
```python
from .tab_my_feature import create_my_feature_tab

__all__ = [
    # ... existing exports ...
    'create_my_feature_tab',
]
```

4. **Add to main GUI** in `unified_gui.py`:
```python
from tabs import (
    # ... existing imports ...
    create_my_feature_tab,
)

# In create_unified_interface():
with gr.Tabs():
    # ... existing tabs ...
    create_my_feature_tab()
```

## Removing a Tab

To remove a tab:

1. Comment out or delete the import in `unified_gui.py`
2. Comment out or delete the function call in `create_unified_interface()`
3. Optionally delete the tab file from the `tabs/` directory

## Benefits of This Structure

✅ **Modular** - Each tab is self-contained and independent
✅ **Maintainable** - Easy to find and modify specific features
✅ **Scalable** - Add or remove tabs without affecting others
✅ **Testable** - Test individual tabs in isolation
✅ **Readable** - No more 2000+ line files
✅ **Team-friendly** - Multiple developers can work on different tabs simultaneously

## Dependencies

Each tab file imports only what it needs. Common dependencies are:
- `gradio` - For UI components
- Processing modules from the parent directory (e.g., `diarization`, `audio_chopper`)
- External libraries specific to each feature

## Original Backup

The original monolithic `unified_gui.py` has been saved as `unified_gui_old.py` in the parent directory for reference.

