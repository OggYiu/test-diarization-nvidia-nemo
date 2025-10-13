"""
Audio Processing Pipeline
Combines diarization, audio chopping, speech-to-text, and LLM analysis.

All temporary files and results are stored within the diarization-api folder.
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import tempfile

# Third-party imports
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from omegaconf import OmegaConf

# NeMo imports for diarization
try:
    from nemo.collections.asr.models import ClusteringDiarizer
except ModuleNotFoundError:
    print("\nModule 'nemo' is not installed. Please install:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("  pip install nemo_toolkit[all]")
    raise

# FunASR for STT
try:
    from funasr import AutoModel
except ModuleNotFoundError:
    print("\nModule 'funasr' is not installed. Please install:")
    print("  pip install funasr")
    raise

# LangChain for LLM analysis
try:
    from langchain_ollama import ChatOllama
except ModuleNotFoundError:
    print("\nModule 'langchain_ollama' is not installed. Please install:")
    print("  pip install langchain-ollama")
    raise


class AudioPipeline:
    """Unified audio processing pipeline."""
    
    def __init__(self, work_dir="./output", num_speakers=2):
        """
        Initialize the pipeline.
        
        Args:
            work_dir: Working directory for all outputs (default: ./output)
            num_speakers: Number of speakers for diarization (default: 2)
        """
        self.work_dir = Path(work_dir)
        self.num_speakers = num_speakers
        
        # Create subdirectories
        self.diarization_dir = self.work_dir / "diarization"
        self.chunks_dir = self.work_dir / "audio_chunks"
        self.transcription_dir = self.work_dir / "transcriptions"
        self.analysis_dir = self.work_dir / "analysis"
        
        # Initialize STT model (lazy loading)
        self.stt_model = None
        
        print(f"Pipeline initialized with work directory: {self.work_dir.absolute()}")
    
    def _create_directories(self):
        """Create all necessary directories."""
        for dir_path in [self.diarization_dir, self.chunks_dir, 
                         self.transcription_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _cleanup_directory(self, dir_path):
        """Clean up a directory."""
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ========== Step 1: Diarization ==========
    
    def diarize_audio(self, audio_filepath):
        """
        Perform speaker diarization on an audio file.
        
        Args:
            audio_filepath: Path to the audio file
            
        Returns:
            Path to the RTTM file containing diarization results
        """
        print("\n" + "="*60)
        print("STEP 1: Speaker Diarization")
        print("="*60)
        
        # Clean up diarization directory
        self._cleanup_directory(self.diarization_dir)
        
        # Create temporary input manifest file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_manifest:
            manifest_data = {
                "audio_filepath": str(Path(audio_filepath).absolute()),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "num_speakers": self.num_speakers
            }
            json.dump(manifest_data, temp_manifest)
            manifest_filepath = temp_manifest.name
        
        try:
            # Create configuration
            CONFIG = OmegaConf.create({
                'batch_size': 32,
                'sample_rate': 16000,
                'verbose': True,
                'diarizer': {
                    'collar': 0.25,
                    'ignore_overlap': True,
                    'manifest_filepath': manifest_filepath,
                    'out_dir': str(self.diarization_dir),
                    'oracle_vad': False,
                    'vad': {
                        'model_path': 'vad_multilingual_marblenet',
                        'batch_size': 32,
                        'parameters': {
                            'window_length_in_sec': 0.63,
                            'shift_length_in_sec': 0.08,
                            'smoothing': False,
                            'overlap': 0.5,
                            'scale': 'absolute',
                            'onset': 0.7,
                            'offset': 0.4,
                            'pad_onset': 0.1,
                            'pad_offset': -0.05,
                            'min_duration_on': 0.1,
                            'min_duration_off': 0.3,
                            'filter_speech_first': True,
                            'normalize': False
                        }
                    },
                    'speaker_embeddings': {
                        'model_path': 'titanet_large',
                        'batch_size': 32,
                        'parameters': {
                            'window_length_in_sec': [1.5, 1.25, 1.0, 0.75, 0.5],
                            'shift_length_in_sec': [0.75, 0.625, 0.5, 0.375, 0.25],
                            'multiscale_weights': [1, 1, 1, 1, 1],
                            'save_embeddings': False
                        }
                    },
                    'clustering': {
                        'parameters': {
                            'oracle_num_speakers': False,
                            'max_num_speakers': 8,
                            'max_rp_threshold': 0.15,
                            'sparse_search_volume': 30
                        }
                    },
                    'msdd_model': {
                        'model_path': 'diar_msdd_telephonic',
                        'parameters': {
                            'sigmoid_threshold': [0.7, 1.0],
                            'use_speaker_embed': True,
                            'use_clus_as_spk_embed': False,
                            'infer_batch_size': 25,
                            'seq_eval_mode': False,
                            'diar_window_length': 50,
                            'overlap_infer_spk_limit': 5,
                            'max_overlap_spk_num': None
                        }
                    }
                },
                'num_workers': 0,
                'device': 'cpu'
            })
            
            # Run diarization
            print("Running diarization...")
            diarizer = ClusteringDiarizer(cfg=CONFIG)
            diarizer.diarize()
            
            # Find and return the .rttm file path
            rttm_dir = self.diarization_dir / 'pred_rttms'
            if not rttm_dir.exists():
                raise FileNotFoundError(f"RTTM directory not found: {rttm_dir}")
            
            rttm_files = list(rttm_dir.glob('*.rttm'))
            if not rttm_files:
                raise FileNotFoundError(f"No .rttm file found in: {rttm_dir}")
            
            rttm_filepath = rttm_files[0]
            print(f"✓ Diarization complete! RTTM file: {rttm_filepath}")
            return rttm_filepath
            
        finally:
            # Clean up temporary manifest file
            if os.path.exists(manifest_filepath):
                os.unlink(manifest_filepath)
    
    # ========== Step 2: Audio Chopping ==========
    
    def _read_rttm_file(self, rttm_path):
        """Read an RTTM file and extract speaker segments."""
        segments = []
        
        with open(rttm_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 4:
                    continue
                
                # Find start and duration indices
                start_idx = None
                for i in range(2, len(parts) - 1):
                    try:
                        _ = float(parts[i])
                        _ = float(parts[i + 1])
                    except Exception:
                        continue
                    
                    next_is_numeric = False
                    if i + 2 < len(parts):
                        try:
                            _ = float(parts[i + 2])
                            next_is_numeric = True
                        except Exception:
                            next_is_numeric = False
                    
                    if not next_is_numeric:
                        start_idx = i
                        break
                
                if start_idx is None:
                    continue
                
                channel_idx = start_idx - 1
                filename = ' '.join(parts[1:channel_idx]) if channel_idx > 1 else parts[1]
                
                try:
                    start = float(parts[start_idx])
                    duration = float(parts[start_idx + 1])
                except Exception:
                    continue
                
                spk_idx = start_idx + 4
                if spk_idx < len(parts):
                    speaker = parts[spk_idx]
                else:
                    speaker = parts[-1]
                
                segment = {
                    'filename': filename,
                    'start': start,
                    'duration': duration,
                    'speaker': speaker,
                    'end': start + duration
                }
                segments.append(segment)
        
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        return segments
    
    def chop_audio(self, audio_filepath, rttm_filepath, padding_ms=100):
        """
        Chop audio into segments based on RTTM diarization results.
        
        Args:
            audio_filepath: Path to the audio file
            rttm_filepath: Path to the RTTM file
            padding_ms: Padding in milliseconds (default: 100)
            
        Returns:
            List of paths to chopped audio files
        """
        print("\n" + "="*60)
        print("STEP 2: Audio Chopping")
        print("="*60)
        
        # Clean up chunks directory
        self._cleanup_directory(self.chunks_dir)
        
        # Read RTTM file
        print(f"Reading RTTM file: {rttm_filepath}")
        segments = self._read_rttm_file(rttm_filepath)
        print(f"Found {len(segments)} segments")
        
        # Load audio file
        print(f"Loading audio file: {audio_filepath}")
        audio = AudioSegment.from_wav(audio_filepath)
        
        # Chop audio into segments
        chopped_files = []
        for i, segment in enumerate(segments, 1):
            start_ms = max(0, segment['start'] * 1000 - padding_ms)
            end_ms = min(len(audio), segment['end'] * 1000 + padding_ms)
            
            # Extract segment
            segment_audio = audio[start_ms:end_ms]
            
            # Save segment
            output_filename = f"segment_{i:03d}.wav"
            output_path = self.chunks_dir / output_filename
            segment_audio.export(str(output_path), format="wav")
            
            chopped_files.append({
                'path': output_path,
                'segment_num': i,
                'speaker': segment['speaker'],
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration']
            })
            
            print(f"  ✓ {output_filename} ({segment['start']:.2f}s - {segment['end']:.2f}s, "
                  f"speaker: {segment['speaker']})")
        
        print(f"✓ Audio chopped into {len(chopped_files)} segments")
        return chopped_files
    
    # ========== Step 3: Speech-to-Text ==========
    
    def _init_stt_model(self, model_name='iic/SenseVoiceSmall'):
        """Initialize the STT model (lazy loading)."""
        if self.stt_model is None:
            print(f"Loading STT model: {model_name}...")
            self.stt_model = AutoModel(
                model=model_name,
                vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                vad_kwargs={"max_single_segment_time": 30000},
                trust_remote_code=False,
                disable_update=True,
            )
            print("✓ STT model loaded")
    
    def _format_stt_output(self, raw_text):
        """Format SenseVoice output (simplified version)."""
        # Emoji dictionaries
        emoji_dict = {
            "<|nospeech|><|Event_UNK|>": "❓",
            "<|zh|>": "", "<|en|>": "", "<|yue|>": "", "<|ja|>": "", "<|ko|>": "",
            "<|nospeech|>": "", "<|HAPPY|>": "😊", "<|SAD|>": "😔", "<|ANGRY|>": "😡",
            "<|NEUTRAL|>": "", "<|BGM|>": "🎼", "<|Speech|>": "", "<|Applause|>": "👏",
            "<|Laughter|>": "😀", "<|FEARFUL|>": "😰", "<|DISGUSTED|>": "🤢",
            "<|SURPRISED|>": "😮", "<|Cry|>": "😭", "<|EMO_UNKNOWN|>": "",
            "<|Sneeze|>": "🤧", "<|Breath|>": "", "<|Cough|>": "😷", "<|Sing|>": "",
            "<|Speech_Noise|>": "", "<|withitn|>": "", "<|woitn|>": "",
            "<|GBG|>": "", "<|Event_UNK|>": "",
        }
        
        # Remove special tokens
        formatted = raw_text
        for token in emoji_dict:
            formatted = formatted.replace(token, emoji_dict[token])
        
        return formatted.strip()
    
    def _load_audio_for_stt(self, audio_path, target_sr=16000):
        """Load audio file for STT."""
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        return waveform.squeeze().numpy()
    
    def transcribe_audio(self, chopped_files, language="auto", model_name='iic/SenseVoiceSmall'):
        """
        Transcribe audio segments using SenseVoice.
        
        Args:
            chopped_files: List of chopped audio file info
            language: Language code (default: "auto")
            model_name: STT model name (default: 'iic/SenseVoiceSmall')
            
        Returns:
            List of transcription results
        """
        print("\n" + "="*60)
        print("STEP 3: Speech-to-Text Transcription")
        print("="*60)
        
        # Initialize STT model
        self._init_stt_model(model_name)
        
        # Transcribe each segment
        results = []
        for file_info in chopped_files:
            audio_path = file_info['path']
            print(f"\nProcessing: {audio_path.name}")
            
            try:
                # Load audio
                audio_array = self._load_audio_for_stt(audio_path)
                
                # Run inference
                result = self.stt_model.generate(
                    input=audio_array,
                    cache={},
                    language=language,
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True
                )
                
                raw_text = result[0]["text"]
                formatted_text = self._format_stt_output(raw_text)
                
                print(f"  Transcription: {formatted_text}")
                
                results.append({
                    'file': audio_path.name,
                    'segment_num': file_info['segment_num'],
                    'speaker': file_info['speaker'],
                    'start': file_info['start'],
                    'end': file_info['end'],
                    'duration': file_info['duration'],
                    'transcription': formatted_text,
                    'raw_transcription': raw_text
                })
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    'file': audio_path.name,
                    'segment_num': file_info['segment_num'],
                    'speaker': file_info['speaker'],
                    'error': str(e)
                })
        
        # Save results
        self.transcription_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = self.transcription_dir / 'transcriptions.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Transcriptions saved to: {json_path}")
        
        # Save conversation text
        conversation_path = self.transcription_dir / 'conversation.txt'
        with open(conversation_path, 'w', encoding='utf-8') as f:
            for r in results:
                if 'transcription' in r:
                    f.write(f"{r['speaker']}: {r['transcription']}\n")
        print(f"✓ Conversation saved to: {conversation_path}")
        
        print(f"✓ Transcribed {len(results)} segments")
        return results
    
    # ========== Step 4: LLM Analysis ==========
    
    def analyze_transcripts(self, transcriptions, model="gpt-oss:20b", 
                          ollama_url="http://192.168.61.2:11434", 
                          custom_prompt=None):
        """
        Analyze transcripts using LLM.
        
        Args:
            transcriptions: List of transcription results
            model: Ollama model name (default: "gpt-oss:20b")
            ollama_url: Ollama server URL (default: "http://192.168.61.2:11434")
            custom_prompt: Custom system prompt (optional)
            
        Returns:
            LLM analysis result
        """
        print("\n" + "="*60)
        print("STEP 4: LLM Analysis")
        print("="*60)
        
        # Build conversation text
        conversation_text = ""
        for t in transcriptions:
            if 'transcription' in t:
                conversation_text += f"{t['speaker']}: {t['transcription']}\n"
        
        # Initialize LLM
        print(f"Connecting to Ollama at {ollama_url}...")
        print(f"Using model: {model}")
        
        try:
            chat_llm = ChatOllama(
                model=model,
                base_url=ollama_url,
                temperature=0.7,
            )
            
            # Prepare prompt
            system_message = custom_prompt or (
                "你是一位精通粵語以及香港股市的分析師。請用繁體中文回應，"
                "並從下方對話中判斷誰是券商、誰是客戶，整理最終下單（股票代號、買/賣、價格、數量）。"
            )
            
            messages = [
                ("system", system_message),
                ("human", conversation_text),
            ]
            
            print("\nSending request to LLM...")
            resp = chat_llm.invoke(messages)
            analysis_result = getattr(resp, "content", str(resp))
            
            print("\n--- LLM Analysis ---")
            print(analysis_result)
            print("--- End of Analysis ---\n")
            
            # Save analysis
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            
            analysis_path = self.analysis_dir / 'analysis.txt'
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write("=== Conversation ===\n\n")
                f.write(conversation_text)
                f.write("\n\n=== Analysis ===\n\n")
                f.write(analysis_result)
            
            print(f"✓ Analysis saved to: {analysis_path}")
            
            return analysis_result
            
        except Exception as e:
            print(f"Error during LLM analysis: {e}")
            print("Continuing without LLM analysis...")
            return None
    
    # ========== Main Pipeline ==========
    
    def process_audio(self, audio_filepath, language="auto", 
                     padding_ms=100, stt_model='iic/SenseVoiceSmall',
                     llm_model="gpt-oss:20b", ollama_url="http://192.168.61.2:11434",
                     skip_llm=False, custom_prompt=None):
        """
        Process audio through the complete pipeline.
        
        Args:
            audio_filepath: Path to the audio file
            language: Language for STT (default: "auto")
            padding_ms: Padding for audio chunks (default: 100)
            stt_model: STT model name (default: 'iic/SenseVoiceSmall')
            llm_model: LLM model name (default: "gpt-oss:20b")
            ollama_url: Ollama server URL (default: "http://192.168.61.2:11434")
            skip_llm: Skip LLM analysis (default: False)
            custom_prompt: Custom system prompt for LLM (optional)
            
        Returns:
            Dictionary with all results
        """
        start_time = datetime.now()
        print("\n" + "="*60)
        print("AUDIO PROCESSING PIPELINE")
        print("="*60)
        print(f"Input audio: {audio_filepath}")
        print(f"Work directory: {self.work_dir.absolute()}")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create directories
        self._create_directories()
        
        # Step 1: Diarization
        rttm_filepath = self.diarize_audio(audio_filepath)
        
        # Step 2: Audio Chopping
        chopped_files = self.chop_audio(audio_filepath, rttm_filepath, padding_ms)
        
        # Step 3: Speech-to-Text
        transcriptions = self.transcribe_audio(chopped_files, language, stt_model)
        
        # Step 4: LLM Analysis (optional)
        analysis_result = None
        if not skip_llm:
            analysis_result = self.analyze_transcripts(
                transcriptions, llm_model, ollama_url, custom_prompt
            )
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Duration: {duration:.2f} seconds")
        print(f"Results saved in: {self.work_dir.absolute()}")
        print(f"  - Diarization: {self.diarization_dir}")
        print(f"  - Audio chunks: {self.chunks_dir}")
        print(f"  - Transcriptions: {self.transcription_dir}")
        if not skip_llm:
            print(f"  - Analysis: {self.analysis_dir}")
        print("="*60)
        
        return {
            'rttm_file': str(rttm_filepath),
            'chopped_files': [str(f['path']) for f in chopped_files],
            'transcriptions': transcriptions,
            'analysis': analysis_result,
            'duration_seconds': duration
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Audio Processing Pipeline: Diarization → Chopping → STT → LLM Analysis"
    )
    
    # Input/output
    parser.add_argument('audio_file', type=str, 
                       help='Path to the audio file to process')
    parser.add_argument('--work-dir', type=str, default='./output',
                       help='Working directory for all outputs (default: ./output)')
    
    # Diarization
    parser.add_argument('--num-speakers', type=int, default=2,
                       help='Number of speakers for diarization (default: 2)')
    
    # Audio chopping
    parser.add_argument('--padding', type=int, default=100,
                       help='Padding in milliseconds for audio chunks (default: 100)')
    
    # STT
    parser.add_argument('--language', type=str, default='auto',
                       choices=['auto', 'zh', 'en', 'yue', 'ja', 'ko', 'nospeech'],
                       help='Language for STT (default: auto)')
    parser.add_argument('--stt-model', type=str, default='iic/SenseVoiceSmall',
                       help='STT model name (default: iic/SenseVoiceSmall)')
    
    # LLM
    parser.add_argument('--skip-llm', action='store_true',
                       help='Skip LLM analysis step')
    parser.add_argument('--llm-model', type=str, default='gpt-oss:20b',
                       help='Ollama model name (default: gpt-oss:20b)')
    parser.add_argument('--ollama-url', type=str, default='http://192.168.61.2:11434',
                       help='Ollama server URL (default: http://192.168.61.2:11434)')
    parser.add_argument('--prompt', type=str,
                       help='Custom system prompt for LLM analysis')
    
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = AudioPipeline(
        work_dir=args.work_dir,
        num_speakers=args.num_speakers
    )
    
    # Process audio
    try:
        results = pipeline.process_audio(
            audio_filepath=args.audio_file,
            language=args.language,
            padding_ms=args.padding,
            stt_model=args.stt_model,
            llm_model=args.llm_model,
            ollama_url=args.ollama_url,
            skip_llm=args.skip_llm,
            custom_prompt=args.prompt
        )
        
        # Save summary
        summary_path = Path(args.work_dir) / 'pipeline_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nPipeline summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

