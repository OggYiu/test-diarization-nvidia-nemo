from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch
import sys

# Set console encoding to UTF-8 for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Load the model and processor
model = WhisperForConditionalGeneration.from_pretrained("khleeloo/whisper-large-v3-cantonese")
processor = WhisperProcessor.from_pretrained("khleeloo/whisper-large-v3-cantonese")

# Load the audio file
audio_file = "test.wav"
audio, sampling_rate = librosa.load(audio_file, sr=16000)

# Process the audio
input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(input_features)

# Decode the transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# Print the result (with proper encoding for Windows console)
result = transcription[0]
print("Transcription:", result)

# Also save to a text file to avoid encoding issues
with open("transcription_output.txt", "w", encoding="utf-8") as f:
    f.write(f"Transcription: {result}\n")
