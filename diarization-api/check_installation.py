"""
Installation checker for audio processing pipeline.
Run this script to verify all dependencies are installed correctly.
"""

import sys

def check_module(module_name, import_path=None, friendly_name=None):
    """Check if a module can be imported."""
    if import_path is None:
        import_path = module_name
    if friendly_name is None:
        friendly_name = module_name
    
    try:
        __import__(import_path)
        print(f"✓ {friendly_name}")
        return True
    except ImportError as e:
        print(f"✗ {friendly_name} - NOT INSTALLED")
        print(f"  Error: {e}")
        return False

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print("✓ FFmpeg")
            return True
        else:
            print("✗ FFmpeg - NOT FOUND")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ FFmpeg - NOT FOUND")
        print("  Install: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (macOS)")
        return False

def main():
    """Main checker function."""
    print("="*60)
    print("Audio Processing Pipeline - Installation Checker")
    print("="*60)
    print()
    
    results = []
    
    print("Core Dependencies:")
    results.append(check_module("numpy"))
    results.append(check_module("torch"))
    results.append(check_module("torchaudio"))
    print()
    
    print("Audio Processing:")
    results.append(check_module("pydub"))
    results.append(check_module("librosa"))
    results.append(check_ffmpeg())
    print()
    
    print("NeMo Toolkit (Diarization):")
    results.append(check_module("nemo", "nemo.collections.asr", "nemo_toolkit"))
    print()
    
    print("Configuration:")
    results.append(check_module("omegaconf"))
    print()
    
    print("Speech-to-Text:")
    results.append(check_module("funasr"))
    print()
    
    print("LLM Integration (Optional):")
    results.append(check_module("langchain"))
    results.append(check_module("langchain_ollama", "langchain_ollama"))
    print()
    
    print("="*60)
    if all(results):
        print("✓ All dependencies installed successfully!")
        print("You're ready to use the audio pipeline.")
    else:
        print("✗ Some dependencies are missing.")
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("For PyTorch, install separately first:")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print()
        print("For NeMo:")
        print("  pip install nemo_toolkit[all]")
    print("="*60)
    
    return 0 if all(results) else 1

if __name__ == '__main__':
    sys.exit(main())

