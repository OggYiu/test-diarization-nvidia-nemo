"""
Simple test script for the audio pipeline.
This script helps you test the pipeline with your audio file.
"""

import sys
import os
from pathlib import Path

def main():
    """Run a simple test of the pipeline."""
    
    print("="*60)
    print("Audio Pipeline Test Script")
    print("="*60)
    print()
    
    # Check if audio file argument is provided
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <audio_file>")
        print()
        print("Example:")
        print("  python test_pipeline.py ../demo/phone_recordings/test.wav")
        print()
        print("This will run the pipeline with default settings:")
        print("  - 2 speakers")
        print("  - Auto language detection")
        print("  - Output to ./test_output/")
        print("  - Skip LLM analysis")
        return 1
    
    audio_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return 1
    
    print(f"Testing with audio file: {audio_file}")
    print()
    
    # Import the pipeline
    try:
        from audio_pipeline import AudioPipeline
    except ImportError as e:
        print(f"Error importing audio_pipeline: {e}")
        print()
        print("Make sure all dependencies are installed:")
        print("  python check_installation.py")
        return 1
    
    # Run the pipeline
    print("Initializing pipeline...")
    pipeline = AudioPipeline(
        work_dir='./test_output',
        num_speakers=2
    )
    
    print()
    print("Starting pipeline...")
    print("(This may take several minutes, especially on first run)")
    print()
    
    try:
        results = pipeline.process_audio(
            audio_filepath=audio_file,
            language='auto',
            skip_llm=True  # Skip LLM by default for testing
        )
        
        print()
        print("="*60)
        print("TEST SUCCESSFUL!")
        print("="*60)
        print()
        print("Results saved to: ./test_output/")
        print()
        print("Check these files:")
        print("  - ./test_output/transcriptions/conversation.txt")
        print("  - ./test_output/transcriptions/transcriptions.json")
        print("  - ./test_output/audio_chunks/ (segmented audio)")
        print()
        print("To view the conversation:")
        print("  cat ./test_output/transcriptions/conversation.txt")
        print()
        
        # Try to print the conversation
        conversation_file = Path('./test_output/transcriptions/conversation.txt')
        if conversation_file.exists():
            print("="*60)
            print("CONVERSATION PREVIEW:")
            print("="*60)
            with open(conversation_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Print first 500 characters
                if len(content) > 500:
                    print(content[:500] + "...")
                    print(f"\n(Full conversation has {len(content)} characters)")
                else:
                    print(content)
            print("="*60)
        
        return 0
        
    except Exception as e:
        print()
        print("="*60)
        print("TEST FAILED!")
        print("="*60)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

