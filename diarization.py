import os
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

CONFIG = OmegaConf.create({
    'batch_size': 32,  # Top-level for embedding extraction batching
    'sample_rate': 16000,
    'verbose': True,
    'diarizer': {
        'collar': 0.25,  # Add this line
        'ignore_overlap': True,  # Add this
        'manifest_filepath': './demo/input_manifest.json',
        'out_dir': "./demo/output",
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
                'sigmoid_threshold': [0.7, 1.0],'use_speaker_embed': True,  # Add this
                'use_clus_as_spk_embed': False,  # Add this
                'infer_batch_size': 25,  # Add this
                'seq_eval_mode': False,  # Add this
                'diar_window_length': 50,  # Add this
                'overlap_infer_spk_limit': 5,  # Add this
                'max_overlap_spk_num': None  # Add this (use None for null)
            }
        }
    },
    'num_workers': 0,
    'device': 'cpu'
})

diarizer = ClusteringDiarizer(cfg=CONFIG)
diarizer.diarize()