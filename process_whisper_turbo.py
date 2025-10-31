import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import numpy as np
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Try to import audio loading libraries
try:
    import librosa
    USE_LIBROSA = True
except ImportError:
    USE_LIBROSA = False
    try:
        import soundfile as sf
        USE_SOUNDFILE = True
    except ImportError:
        USE_SOUNDFILE = False
        from scipy.io import wavfile
        USE_SCIPY = True

def load_audio(audio_path, target_sr=16000):
    """
    Load audio file without using torchcodec.
    
    Args:
        audio_path: Path to the audio file
        target_sr: Target sampling rate (Whisper uses 16kHz)
    
    Returns:
        audio_array: Audio as numpy array
        sampling_rate: Sampling rate
    """
    if USE_LIBROSA:
        print("Loading audio with librosa...")
        audio_array, sampling_rate = librosa.load(audio_path, sr=target_sr)
    elif USE_SOUNDFILE:
        print("Loading audio with soundfile...")
        audio_array, sampling_rate = sf.read(audio_path)
        if sampling_rate != target_sr:
            # Simple resampling - note: this is basic, librosa is better
            import scipy.signal
            num_samples = int(len(audio_array) * target_sr / sampling_rate)
            audio_array = scipy.signal.resample(audio_array, num_samples)
            sampling_rate = target_sr
    else:
        print("Loading audio with scipy.io.wavfile...")
        sampling_rate, audio_array = wavfile.read(audio_path)
        
        # Normalize to float32 in range [-1, 1]
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio_array.dtype == np.int32:
            audio_array = audio_array.astype(np.float32) / 2147483648.0
        else:
            audio_array = audio_array.astype(np.float32)
        
        if sampling_rate != target_sr:
            import scipy.signal
            num_samples = int(len(audio_array) * target_sr / sampling_rate)
            audio_array = scipy.signal.resample(audio_array, num_samples)
            sampling_rate = target_sr
    
    return audio_array, sampling_rate

def transcribe_audio(audio_path):
    """
    Transcribe audio using Whisper large-v3-turbo model.
    
    Args:
        audio_path: Path to the audio file to transcribe
    """
    # Check if audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"Loading Whisper large-v3-turbo model...")
    
    # Set up device and data type
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Using device: {device}")
    
    model_id = "openai/whisper-large-v3-turbo"
    
    # Load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)
    
    print(f"Processing audio file: {audio_path}")
    
    # Load audio manually to avoid torchcodec dependency
    audio_array, sampling_rate = load_audio(audio_path, target_sr=16000)
    print(f"Audio loaded: duration={len(audio_array)/sampling_rate:.2f}s, sampling_rate={sampling_rate}Hz")
    
    # Process audio through the processor
    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        return_attention_mask=True
    )
    inputs = inputs.to(device, dtype=torch_dtype)
    
    # Generation parameters
    # Note: turbo model has max_target_positions=448, need to account for decoder_input_ids
    # Force language to Cantonese (yue)
    # forced_decoder_ids = processor.get_decoder_prompt_ids(language="yue", task="transcribe")
    
    gen_kwargs = {
        "max_new_tokens": 445,
        "num_beams": 1,
        "return_timestamps": True,
        # "forced_decoder_ids": forced_decoder_ids,
        "language": "yue",
    }
    
    print("Generating transcription (language: Cantonese/yue)...")
    
    # Generate transcription
    pred_ids = model.generate(**inputs, **gen_kwargs)
    
    # Decode with timestamps
    transcription = processor.batch_decode(
        pred_ids,
        skip_special_tokens=True,
        decode_with_timestamps=True
    )[0]
    
    # Also get version without timestamps
    transcription_clean = processor.batch_decode(
        pred_ids,
        skip_special_tokens=True,
        decode_with_timestamps=False
    )[0]
    
    # Create result dictionary similar to pipeline output
    result = {
        "text": transcription_clean,
        "transcription_with_timestamps": transcription
    }
    
    print("\n" + "="*80)
    print("TRANSCRIPTION RESULT")
    print("="*80)
    
    # Handle potential encoding issues in console output
    try:
        print(f"\nFull Text:\n{result['text']}")
        
        if 'transcription_with_timestamps' in result and result['transcription_with_timestamps']:
            print("\n" + "-"*80)
            print("TRANSCRIPTION WITH TIMESTAMPS")
            print("-"*80)
            print(result['transcription_with_timestamps'])
    except UnicodeEncodeError:
        # Fallback for Windows console encoding issues
        print(f"\nFull Text:\n{result['text'].encode('ascii', 'ignore').decode('ascii')}")
        print("\n(Note: Some non-ASCII characters couldn't be displayed in console)")
        print("(Check the output file for complete transcription)")
        
        if 'transcription_with_timestamps' in result and result['transcription_with_timestamps']:
            print("\n" + "-"*80)
            print("TRANSCRIPTION WITH TIMESTAMPS")
            print("-"*80)
            print(result['transcription_with_timestamps'].encode('ascii', 'ignore').decode('ascii'))
    
    print("\n" + "="*80)
    
    # Save to file
    output_file = audio_path.replace('.wav', '_whisper_turbo_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("WHISPER LARGE-V3-TURBO TRANSCRIPTION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Audio File: {audio_path}\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Device: {device}\n\n")
        f.write("FULL TRANSCRIPTION:\n")
        f.write(result['text'] + "\n\n")
        
        if 'transcription_with_timestamps' in result and result['transcription_with_timestamps']:
            f.write("\nTRANSCRIPTION WITH TIMESTAMPS:\n")
            f.write("-"*80 + "\n")
            f.write(result['transcription_with_timestamps'] + "\n")
    
    print(f"\nTranscription saved to: {output_file}")
    
    return result

if __name__ == "__main__":
    audio_file = "./test.wav"
    
    try:
        result = transcribe_audio(audio_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

