"""
Tab: Text Correction
Correct multiple wrong words in conversation text with a single correct word
"""

import json
import traceback
import gradio as gr


def correct_text(conversation_text: str, correction_json: str) -> str:
    """
    Replace multiple wrong words with a correct word in conversation text
    
    Args:
        conversation_text: The text to be corrected
        correction_json: JSON string with structure:
            {
                "wrong_words": ["word1", "word2", "word3"],
                "correct_word": "correctword"
            }
            or array of corrections:
            [
                {
                    "wrong_words": ["word1", "word2"],
                    "correct_word": "correct1"
                },
                {
                    "wrong_words": ["word3"],
                    "correct_word": "correct2"
                }
            ]
    
    Returns:
        str: Corrected text or error message
    """
    try:
        # Validate inputs
        if not conversation_text or not conversation_text.strip():
            return "❌ Error: Please provide conversation text"
        
        if not correction_json or not correction_json.strip():
            return "❌ Error: Please provide correction JSON"
        
        # Parse JSON
        try:
            correction_data = json.loads(correction_json)
        except json.JSONDecodeError as e:
            return f"❌ JSON Parse Error: {str(e)}\n\nPlease provide valid JSON format"
        
        corrected_text = conversation_text
        corrections_applied = []
        
        # Handle both single correction object and array of corrections
        if isinstance(correction_data, dict):
            # Single correction object
            correction_list = [correction_data]
        elif isinstance(correction_data, list):
            # Array of corrections
            correction_list = correction_data
        else:
            return "❌ Error: JSON must be an object or array of objects"
        
        # Apply each correction
        for idx, correction in enumerate(correction_list):
            # Validate structure
            if not isinstance(correction, dict):
                return f"❌ Error: Item {idx} is not an object"
            
            if "wrong_words" not in correction or "correct_word" not in correction:
                return f"❌ Error: Item {idx} missing 'wrong_words' or 'correct_word' field"
            
            wrong_words = correction["wrong_words"]
            correct_word = correction["correct_word"]
            
            # Validate types
            if not isinstance(wrong_words, list):
                return f"❌ Error: Item {idx} 'wrong_words' must be an array"
            
            if not isinstance(correct_word, str):
                return f"❌ Error: Item {idx} 'correct_word' must be a string"
            
            # Apply replacements
            for wrong_word in wrong_words:
                if not isinstance(wrong_word, str):
                    return f"❌ Error: All items in 'wrong_words' must be strings"
                
                if wrong_word in corrected_text:
                    count = corrected_text.count(wrong_word)
                    corrected_text = corrected_text.replace(wrong_word, correct_word)
                    corrections_applied.append(f"  • '{wrong_word}' → '{correct_word}' ({count} occurrence(s))")
        
        # Build result message
        if corrections_applied:
            result = "✅ Text corrected successfully!\n\n"
            result += f"📊 Corrections applied ({len(corrections_applied)}):\n"
            result += "\n".join(corrections_applied)
            result += "\n\n" + "=" * 60 + "\n\n"
            result += corrected_text
            return result
        else:
            result = "⚠️ No corrections applied (no matches found)\n\n"
            result += "=" * 60 + "\n\n"
            result += corrected_text
            return result
        
    except Exception as e:
        error_msg = f"❌ Unexpected Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg


def load_example_json():
    """Load an example JSON structure"""
    example = {
        "wrong_words": ["worng", "wrng", "wron"],
        "correct_word": "wrong"
    }
    return json.dumps(example, indent=2, ensure_ascii=False)


def load_example_json_multiple():
    """Load an example JSON structure with multiple corrections"""
    example = [
        {
            "wrong_words": ["worng", "wrng", "wron"],
            "correct_word": "wrong"
        },
        {
            "wrong_words": ["teh", "te"],
            "correct_word": "the"
        }
    ]
    return json.dumps(example, indent=2, ensure_ascii=False)


def create_text_correction_tab():
    """Create and return the Text Correction tab"""
    with gr.Tab("✏️ Text Correction"):
        gr.Markdown("### Text Correction Tool")
        gr.Markdown("*Replace multiple wrong words with correct words in your conversation text*")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Input")
                
                # Conversation text input
                conversation_text_input = gr.Textbox(
                    label="Conversation Text",
                    placeholder="Enter your conversation text here...",
                    lines=12,
                    info="The text that needs to be corrected",
                )
                
                gr.Markdown("---")
                
                # JSON correction input
                correction_json_input = gr.Textbox(
                    label="Correction JSON",
                    placeholder='{\n  "wrong_words": ["word1", "word2"],\n  "correct_word": "correct"\n}',
                    lines=10,
                    info="JSON with wrong words and correct word",
                )
                
                # Example buttons
                with gr.Row():
                    example_single_btn = gr.Button("📝 Load Single Example", size="sm")
                    example_multiple_btn = gr.Button("📝 Load Multiple Example", size="sm")
                
                gr.Markdown("---")
                
                # Correct button
                correct_btn = gr.Button("🔄 Correct Text", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("#### Output")
                
                # Corrected text output
                corrected_text_output = gr.Textbox(
                    label="Corrected Text",
                    lines=28,
                    interactive=False,
                    show_copy_button=True,
                    placeholder="Corrected text will appear here...",
                )
        
        # Help section
        with gr.Accordion("ℹ️ Help & Examples", open=False):
            gr.Markdown("""
            ### JSON Format
            
            **Single Correction:**
            ```json
            {
                "wrong_words": ["worng", "wrng", "wron"],
                "correct_word": "wrong"
            }
            ```
            
            **Multiple Corrections:**
            ```json
            [
                {
                    "wrong_words": ["worng", "wrng"],
                    "correct_word": "wrong"
                },
                {
                    "wrong_words": ["teh", "te"],
                    "correct_word": "the"
                }
            ]
            ```
            
            ### Features
            - Replace multiple variations of wrong words with a single correct word
            - Supports both single correction and batch corrections
            - Shows how many times each word was replaced
            - Case-sensitive replacement
            
            ### Tips
            - Make sure your JSON is valid (use quotes around strings)
            - The `wrong_words` field must be an array (even for single word)
            - All replacements are case-sensitive
            - Click the example buttons to see the format
            """)
        
        # Connect buttons
        correct_btn.click(
            fn=correct_text,
            inputs=[conversation_text_input, correction_json_input],
            outputs=corrected_text_output,
        )
        
        example_single_btn.click(
            fn=load_example_json,
            inputs=[],
            outputs=correction_json_input,
        )
        
        example_multiple_btn.click(
            fn=load_example_json_multiple,
            inputs=[],
            outputs=correction_json_input,
        )

