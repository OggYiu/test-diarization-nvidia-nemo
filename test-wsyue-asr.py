from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download
import librosa
import torch
import sys
import os

# Set console encoding to UTF-8 for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

print("Downloading WSYue-ASR model...")

# Download the model checkpoint from Hugging Face
model_path = hf_hub_download(
    repo_id="ASLP-lab/WSYue-ASR",
    filename="whisper_medium_yue/whisper_medium_yue.pt",
    cache_dir="./model_cache"
)

print(f"Model downloaded to: {model_path}")

# Load the base Whisper medium model architecture
print("Loading Whisper medium model architecture...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

# Load the fine-tuned weights
print("Loading fine-tuned weights for Cantonese...")
checkpoint = torch.load(model_path, map_location="cpu")

# The checkpoint might be stored in different formats, handle accordingly
if isinstance(checkpoint, dict):
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Load the state dict into the model
try:
    model.load_state_dict(state_dict, strict=False)
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load all weights. Error: {e}")
    print("Attempting to load with partial matching...")
    # Try to load compatible weights only
    model_dict = model.state_dict()
    compatible_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(compatible_dict)}/{len(state_dict)} weight tensors")

# Set model to evaluation mode
model.eval()

# Load the audio file
print("\nLoading audio file: test.wav")
audio_file = "test.wav"
audio, sampling_rate = librosa.load(audio_file, sr=16000)
print(f"Audio loaded: {len(audio)/sampling_rate:.2f} seconds")

# Process the audio
print("Processing audio...")
input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

# Generate transcription
print("Generating transcription...")
with torch.no_grad():
    # Generate with language set to Chinese (Yue/Cantonese)
    predicted_ids = model.generate(
        input_features,
        language="zh",
        task="transcribe",
        max_length=448
    )

# Decode the transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# Print the result (with proper encoding for Windows console)
result = transcription[0]
print("\n" + "="*50)
print("Transcription (WSYue-ASR):", result)
print("="*50)

# Save to a text file to avoid encoding issues
output_file = "transcription_wsyue_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Transcription (WSYue-ASR Model):\n{result}\n")

print(f"\nTranscription saved to: {output_file}")

