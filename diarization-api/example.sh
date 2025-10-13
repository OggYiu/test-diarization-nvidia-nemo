#!/bin/bash
# Example usage of the audio pipeline

# Basic example - process audio with default settings
# python audio_pipeline.py ../demo/phone_recordings/test.wav

# Example with custom output directory
# python audio_pipeline.py ../demo/phone_recordings/test.wav --work-dir ./my_results

# Example with Chinese/Cantonese audio
# python audio_pipeline.py recording.wav --language yue --num-speakers 2

# Example skipping LLM analysis
python audio_pipeline.py ../demo/phone_recordings/test.wav \
  --work-dir ./output \
  --num-speakers 2 \
  --language auto \
  --skip-llm

# Example with LLM analysis (requires Ollama server)
# python audio_pipeline.py ../demo/phone_recordings/test.wav \
#   --work-dir ./output \
#   --num-speakers 2 \
#   --language yue \
#   --llm-model qwen2.5:7b-instruct \
#   --ollama-url http://localhost:11434

echo "Processing complete!"

