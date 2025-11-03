# U2pp-Conformer-Yue Model Integration

## Overview
Successfully integrated the U2pp-Conformer-Yue Cantonese ASR model into the STT (Speech-to-Text) system. This model uses sherpa-onnx and provides fast, CPU-based Cantonese transcription using the WenetSpeech-Yue U2pp-Conformer CTC model.

## Changes Made

### 1. Dependencies Added (`requirements.txt`)
```
sherpa-onnx>=1.9.0
resampy>=0.4.2
```

### 2. Model Features
- **Model**: WenetSpeech-Yue U2pp-Conformer-CTC (int8 quantized)
- **Framework**: sherpa-onnx (ONNX Runtime)
- **Device**: CPU (optimized with int8 quantization)
- **Sample Rate**: 16kHz
- **Language**: Cantonese (Yue) with Chinese/English support
- **Caching**: Full MongoDB cache support

### 3. Implementation Details

#### Added Components:
- **Global Variables**: `u2pp_conformer_model`, `current_u2pp_conformer_loaded`
- **MongoDB Collection**: `transcriptions_u2pp_conformer_yue`
- **Cache Functions**: 
  - `load_transcription_cache_u2pp()`
  - `save_transcription_to_cache_u2pp()`
- **Model Functions**:
  - `initialize_u2pp_conformer_model()` - Loads the ONNX model
  - `transcribe_single_audio_u2pp()` - Transcribes audio with caching
- **UI Components**:
  - Checkbox for U2pp-Conformer-Yue model selection
  - Output textbox for U2pp transcriptions
  - Integration in both pipeline modes

#### Model Loading Paths:
The model will be automatically searched in these locations:
1. `./model/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/`
2. `./model_cache/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/`
3. `~/.cache/sherpa-onnx/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/`

### 4. Updated Functions

#### `process_chop_and_transcribe()`
- Added `use_u2pp` parameter
- Added U2pp model initialization
- Added U2pp transcription loop
- Added U2pp results to JSON and text outputs
- Returns 9 values instead of 7

#### `process_batch_transcription()`
- Added `use_u2pp` parameter
- Added U2pp model initialization
- Added U2pp transcription loop
- Added U2pp results to JSON and text outputs
- Returns 8 values instead of 6

### 5. UI Updates in `create_stt_tab()`
- Added `stt_use_u2pp` checkbox (default: False)
- Added `stt_u2pp_output` textbox display (3-column layout)
- Updated button click handler to pass U2pp parameters

## Installation Instructions

### Step 1: Install Dependencies
```bash
# Activate your virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install new dependencies
pip install sherpa-onnx>=1.9.0 resampy>=0.4.2
```

### Step 2: Download the Model
Download the U2pp-Conformer-Yue model from:
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

Look for: **sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10**

Extract it to one of these locations:
- `./model/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/`
- `./model_cache/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10/`
- Or let it auto-download to `~/.cache/sherpa-onnx/`

### Step 3: Verify Model Files
The model directory should contain:
- `model.int8.onnx` (or `model.onnx`)
- `tokens.txt`

### Step 4: Run the Application
```bash
python unified_gui.py
```

## Usage

1. Navigate to the **"3️⃣ Auto-Diarize & Transcribe"** tab
2. Upload your audio file
3. Under "Model Selection", check **U2pp-Conformer-Yue**
4. Click **"🎯 Auto-Diarize & Transcribe"**
5. View results in the U2pp-Conformer-Yue column

## Features

### Caching
- All U2pp transcriptions are automatically cached in MongoDB
- Subsequent transcriptions of the same file are instant (cache retrieval)
- Cache hits are indicated with 💾 symbol in logs

### Output Files
When U2pp is selected, the following files are generated:
- `transcriptions.json` - Contains U2pp results under `u2pp_conformer_yue` key
- `conversation_u2pp_conformer_yue.txt` - Formatted conversation view
- `transcription_results.zip` - Contains all output files

### Multi-Model Support
You can select multiple models simultaneously:
- ✅ SenseVoiceSmall
- ✅ Whisper-v3-Cantonese  
- ✅ U2pp-Conformer-Yue

All selected models will process in sequence and save their results.

## Performance Notes

- **Speed**: U2pp is optimized with int8 quantization for fast CPU inference
- **Memory**: Lower memory footprint compared to large transformer models
- **Quality**: Good for Cantonese transcription with Chinese/English code-switching
- **Device**: Runs on CPU (no GPU required)

## Troubleshooting

### Model Not Found Error
If you see "Model not found in default locations":
1. Download the model from the GitHub release link above
2. Extract to one of the checked paths shown in the error message
3. Ensure `model.int8.onnx` and `tokens.txt` exist

### Import Error (sherpa_onnx)
If you see import errors:
```bash
pip install sherpa-onnx>=1.9.0 resampy>=0.4.2
```

### Audio Resampling Warning
If you see "resampy not available for resampling":
- The model requires 16kHz audio
- Install resampy: `pip install resampy>=0.4.2`
- Or convert audio to 16kHz before processing

## Technical Details

### Model Specifications
- **Architecture**: U2++ Conformer with CTC
- **Training Data**: WenetSpeech-Yue
- **Languages**: Cantonese (primary), Chinese, English
- **Quantization**: int8 (for efficiency)
- **Sample Rate**: 16kHz
- **Framework**: ONNX Runtime via sherpa-onnx

### API Compatibility
The U2pp implementation follows the same patterns as existing models:
- MongoDB caching
- Progress tracking
- Error handling
- Gradio UI integration
- Batch processing support

## Files Modified
- `requirements.txt` - Added dependencies
- `tabs/tab_stt.py` - Complete U2pp integration (~1700 lines)
- `temp.py` - Removed (was reference implementation)

## MongoDB Collections
- **Collection Name**: `transcriptions_u2pp_conformer_yue`
- **Unique Key**: `filename`
- **Fields**:
  - `filename`: Audio file name
  - `transcription`: Transcribed text
  - `raw_transcription`: Raw output (same as transcription for U2pp)
  - `processing_time`: Time taken in seconds
  - `timestamp`: ISO format timestamp
  - `model`: "U2pp-Conformer-Yue"

---

**Integration Complete!** 🎉

The U2pp-Conformer-Yue model is now fully integrated and ready to use. Install the dependencies, download the model, and start transcribing Cantonese audio!

