import os
import json
import tempfile
import argparse
import shutil
import wget  # For downloading sample audio if needed
import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
from omegaconf import OmegaConf

# Try to import NVIDIA NeMo collections; provide clear instructions if missing
try:
    from nemo.collections.asr.models import ClusteringDiarizer
    from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
except ModuleNotFoundError:
    print("\nModule 'nemo' is not installed in the active environment.")
    print("Install a compatible PyTorch first (CPU example):")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("Then install NeMo (example, may require NVIDIA index):")
    print("  pip install nemo_toolkit[all] --extra-index-url https://pypi.ngc.nvidia.com")
    print("After installing, re-run this script.")
    raise


def diarize_audio(audio_filepath, out_dir, num_speakers=2):
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_filepath (str): Path to the audio file to diarize
        out_dir (str): Output directory for diarization results (default: 'diarization_output')
        num_speakers (int, optional): Number of speakers if known (default: None for automatic detection)
    
    Returns:
        str: Content of the .rttm file containing diarization results
    """
    # Delete output directory if it exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"Deleted existing output directory: {out_dir}")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Create temporary input manifest file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_manifest:
        manifest_data = {
            "audio_filepath": audio_filepath,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "num_speakers": num_speakers
        }
        json.dump(manifest_data, temp_manifest)
        manifest_filepath = temp_manifest.name
    
    try:
        # Create configuration
        CONFIG = OmegaConf.create({
            'batch_size': 32,  # Top-level for embedding extraction batching
            'sample_rate': 16000,
            'verbose': True,
            'diarizer': {
                'collar': 0.25,
                'ignore_overlap': True,
                'manifest_filepath': manifest_filepath,
                'out_dir': out_dir,
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
        diarizer = ClusteringDiarizer(cfg=CONFIG)
        diarizer.diarize()
        
        # Find and read the .rttm file
        rttm_dir = os.path.join(out_dir, 'pred_rttms')
        if not os.path.exists(rttm_dir):
            raise FileNotFoundError(f"RTTM directory not found: {rttm_dir}")
        
        rttm_files = [f for f in os.listdir(rttm_dir) if f.endswith('.rttm')]
        if not rttm_files:
            raise FileNotFoundError(f"No .rttm file found in: {rttm_dir}")
        
        # Read the first .rttm file found
        rttm_filepath = os.path.join(rttm_dir, rttm_files[0])
        with open(rttm_filepath, 'r') as f:
            rttm_content = f.read()
        
        print(f"\nDiarization complete! Results saved to: {out_dir}")
        print(f"RTTM file: {rttm_filepath}")
        return rttm_content
        
    finally:
        # Clean up temporary manifest file
        if os.path.exists(manifest_filepath):
            os.unlink(manifest_filepath)


# Example usage (can be commented out or removed when used as a module)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform speaker diarization on an audio file.')
    parser.add_argument('audio_file', type=str, nargs='?', default='./demo/phone_recordings/test.wav',
                        help='Path to the audio file to diarize (default: ./demo/phone_recordings/test.wav)')
    parser.add_argument('--out_dir', type=str, default='./temp/diarization_output',
                        help='Output directory for diarization results (default: ./temp/diarization_output)')
    
    args = parser.parse_args()
    
    if os.path.exists(args.audio_file):
        rttm_content = diarize_audio(args.audio_file, out_dir=args.out_dir)
        print("\n=== RTTM Content ===")
        print(rttm_content)
    else:
        print(f"Audio file not found: {args.audio_file}")
        print("Please provide a valid audio file path.")