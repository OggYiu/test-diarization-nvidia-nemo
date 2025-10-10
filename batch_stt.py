# coding=utf-8

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
from funasr import AutoModel
import re

# Initialize the SenseVoice model
model = None

# NOTE: model initialization occurs in main() so the CLI can pick small/large/custom



# Emoji dictionaries for formatting
emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷"}


def format_str_v2(s):
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = s.count(sptk)
        s = s.replace(sptk, "")
    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    for e in event_dict:
        if sptk_dict[e] > 0:
            s = event_dict[e] + s
    s = s + emo_dict[emo]

    for emoji in emo_set.union(event_set):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)
    return s.strip()


def format_str_v3(s):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None

    def get_event(s):
        return s[0] if s[0] in event_set else None

    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        s = s.replace(lang, "<|lang|>")
    s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
    new_s = " " + s_list[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(s_list)):
        if len(s_list[i]) == 0:
            continue
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
            s_list[i] = s_list[i][1:]
        cur_ent_event = get_event(s_list[i])
        if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += s_list[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


def load_audio(audio_path, target_sr=16000):
    """Load audio file and resample to target sample rate."""
    try:
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Convert to numpy array
        audio_array = waveform.squeeze().numpy()
        
        return audio_array, target_sr
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None, None


def transcribe_audio(audio_path, language="auto"):
    """Transcribe a single audio file using SenseVoice."""
    print(f"\nProcessing: {audio_path}")
    
    # Load audio
    audio_array, sample_rate = load_audio(audio_path)
    if audio_array is None:
        return None
    
    # Run inference
    try:
        result = model.generate(
            input=audio_array,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True
        )
        
        # Extract and format text
        raw_text = result[0]["text"]
        formatted_text = format_str_v3(raw_text)
        
        print(f"Transcription: {formatted_text}")
        
        return {
            "file": os.path.basename(audio_path),
            "path": audio_path,
            "transcription": formatted_text,
            "raw_transcription": raw_text
        }
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return None


def process_folder(input_folder, output_file=None, language="auto"):
    """Process all .wav files in a folder."""
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Folder '{input_folder}' does not exist!")
        return
    
    # Find all .wav files
    wav_files = list(input_path.glob("*.wav"))

    # If files are named like 'segment_001.wav', simple lexicographical sort of
    # the filename yields the correct chronological order. Use string sort
    # primarily, but keep numeric detection as a fallback for mixed names.
    wav_files.sort(key=lambda p: p.name)
    
    if not wav_files:
        print(f"No .wav files found in '{input_folder}'")
        return
    
    print(f"Found {len(wav_files)} .wav file(s)")
    
    # Process each file (already sorted by segment index)
    results = []
    for wav_file in wav_files:
        result = transcribe_audio(str(wav_file), language)
        if result:
            results.append(result)
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Results saved to: {output_file}")

        # Also write a plain-text conversation file (conversation.txt) next to the JSON
        conv_path = output_path.parent / 'conversation.txt'
        try:
            with open(conv_path, 'w', encoding='utf-8') as cf:
                for r in results:
                    # Try to extract speaker id from filename like 'test_speaker_0_segment_001.wav'
                    # If no speaker token is present (e.g., files named 'segment_001.wav'),
                    # use the filename stem as the label so conversation ordering is clear.
                    fname = r.get('file', '')
                    speaker = None
                    m = re.search(r"speaker_(\d+)", fname)
                    if m:
                        speaker = f"speaker_{m.group(1)}"
                    else:
                        # use stem (filename without extension) as label
                        speaker = Path(fname).stem if fname else 'unknown'
                    cf.write(f"{speaker}: {r.get('transcription', '')}\n")
            print(f"✓ Conversation saved to: {conv_path}")
        except Exception as e:
            print(f"Warning: failed to write conversation file: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total files processed: {len(results)}/{len(wav_files)}")
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch Speech-to-Text using SenseVoice model"
    )
    parser.add_argument(
        "input_folder",
        nargs='?',
        type=str,
        # default=os.path.join(".", "demo", "phone_recordings"),
        default=os.path.join(".", "demo", "output", "chopped_audios"),
        help="Path to folder containing .wav files (default: ./demo/phone_recordings)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=os.path.join(".", "demo", "transcriptions", "transcriptions.json"),
        help="Output JSON file path (default: ./demo/transcriptions/transcriptions.json)"
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
        default="auto",
        help="Language code (default: auto)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "large", "custom"],
        default="small",
        help="SenseVoice model size to use (small or large). Use 'custom' with --model-name to specify an exact model string."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Explicit model string (overrides --model-size). Example: 'iic/SenseVoiceSmall' or your custom model id."
    )
    
    args = parser.parse_args()

    # Resolve model selection
    model_size = args.model_size
    if args.model_name:
        model_name = args.model_name
    else:
        if model_size == 'small':
            model_name = 'iic/SenseVoiceSmall'
        elif model_size == 'large':
            model_name = 'iic/SenseVoiceLarge'
        else:
            model_name = 'iic/SenseVoiceSmall'

    # Initialize the SenseVoice model (global)
    global model
    print(f"Loading SenseVoice model: {model_name} ...")
    try:
        model = AutoModel(
            model=model_name,
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 30000},
            trust_remote_code=False,
            disable_update=True,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model '{model_name}': {e}")
        # Provide helpful guidance and try fallback to the small model
        fallback = 'iic/SenseVoiceSmall'
        if model_name != fallback:
            print(f"Falling back to '{fallback}'. If you intended to use a different model, please verify the model id is correct and available on the model hub.")
            try:
                model = AutoModel(
                    model=fallback,
                    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    vad_kwargs={"max_single_segment_time": 30000},
                    trust_remote_code=False,
                    disable_update=True,
                )
                print(f"Fallback model '{fallback}' loaded successfully.")
            except Exception as e2:
                print(f"Fallback load failed: {e2}\nCannot continue without a working model. Exiting.")
                sys.exit(1)
        else:
            print("No fallback available. Exiting.")
            sys.exit(1)
    
    # Ensure output directory exists when an output path is provided
    output_path = Path(args.output)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process folder
    process_folder(args.input_folder, args.output, args.language)


if __name__ == "__main__":
    main()

