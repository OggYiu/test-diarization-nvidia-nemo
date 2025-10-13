import os
import json
import tempfile
import shutil
from pathlib import Path
from omegaconf import OmegaConf

# Try to import NVIDIA NeMo collections
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


def diarize_audio(audio_filepath, out_dir='diarization_output', num_speakers=2):
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_filepath (str): Path to the audio file to diarize
        out_dir (str): Output directory for diarization results (default: 'diarization_output')
        num_speakers (int, optional): Number of speakers if known (default: 2)
    
    Returns:
        str: Path to the output directory containing diarization results
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
        
        print(f"\nDiarization complete! Results saved to: {out_dir}")
        return out_dir
        
    finally:
        # Clean up temporary manifest file
        if os.path.exists(manifest_filepath):
            os.unlink(manifest_filepath)


def get_rttm_content(output_dir):
    """
    Extract the content of the RTTM file from the diarization output directory.
    
    Args:
        output_dir (str): Path to the diarization output directory
    
    Returns:
        str: Content of the RTTM file, or None if not found
    """
    # Look for RTTM file in pred_rttms subdirectory
    rttm_dir = os.path.join(output_dir, "pred_rttms")
    
    if not os.path.exists(rttm_dir):
        print(f"RTTM directory not found: {rttm_dir}")
        return None
    
    # Find the first .rttm file
    rttm_files = [f for f in os.listdir(rttm_dir) if f.endswith('.rttm')]
    
    if not rttm_files:
        print(f"No RTTM files found in {rttm_dir}")
        return None
    
    # Read the first RTTM file
    rttm_path = os.path.join(rttm_dir, rttm_files[0])
    
    try:
        with open(rttm_path, 'r') as f:
            content = f.read()
        print(f"Successfully read RTTM file: {rttm_path}")
        return content
    except Exception as e:
        print(f"Error reading RTTM file {rttm_path}: {e}")
        return None

