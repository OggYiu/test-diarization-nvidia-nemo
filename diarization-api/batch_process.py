"""
Batch processing script for multiple audio files.
Process all audio files in a directory through the pipeline.
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import json

def main():
    """Batch process multiple audio files."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple audio files through the pipeline"
    )
    parser.add_argument('input_dir', type=str,
                       help='Directory containing audio files (.wav)')
    parser.add_argument('--output-dir', type=str, default='./batch_output',
                       help='Base output directory (default: ./batch_output)')
    parser.add_argument('--num-speakers', type=int, default=2,
                       help='Number of speakers (default: 2)')
    parser.add_argument('--language', type=str, default='auto',
                       help='Language code (default: auto)')
    parser.add_argument('--skip-llm', action='store_true',
                       help='Skip LLM analysis')
    parser.add_argument('--llm-model', type=str, default='gpt-oss:20b',
                       help='LLM model name (default: gpt-oss:20b)')
    parser.add_argument('--ollama-url', type=str, default='http://192.168.61.2:11434',
                       help='Ollama server URL')
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    # Find all .wav files
    wav_files = list(input_path.glob('*.wav'))
    if not wav_files:
        print(f"No .wav files found in: {args.input_dir}")
        return 1
    
    print("="*60)
    print("BATCH AUDIO PROCESSING")
    print("="*60)
    print(f"Input directory: {input_path.absolute()}")
    print(f"Output directory: {args.output_dir}")
    print(f"Found {len(wav_files)} audio file(s)")
    print("="*60)
    print()
    
    # Import pipeline
    try:
        from audio_pipeline import AudioPipeline
    except ImportError as e:
        print(f"Error importing audio_pipeline: {e}")
        print("Make sure all dependencies are installed.")
        return 1
    
    # Process each file
    results_summary = []
    successful = 0
    failed = 0
    
    for i, wav_file in enumerate(wav_files, 1):
        print()
        print("="*60)
        print(f"Processing file {i}/{len(wav_files)}: {wav_file.name}")
        print("="*60)
        
        # Create output directory for this file
        file_output_dir = Path(args.output_dir) / wav_file.stem
        
        try:
            # Initialize pipeline for this file
            pipeline = AudioPipeline(
                work_dir=str(file_output_dir),
                num_speakers=args.num_speakers
            )
            
            # Process the file
            result = pipeline.process_audio(
                audio_filepath=str(wav_file),
                language=args.language,
                skip_llm=args.skip_llm,
                llm_model=args.llm_model,
                ollama_url=args.ollama_url
            )
            
            results_summary.append({
                'file': wav_file.name,
                'status': 'success',
                'output_dir': str(file_output_dir),
                'duration': result.get('duration_seconds', 0),
                'num_segments': len(result.get('transcriptions', []))
            })
            
            successful += 1
            print(f"✓ Successfully processed: {wav_file.name}")
            
        except Exception as e:
            print(f"✗ Error processing {wav_file.name}: {e}")
            results_summary.append({
                'file': wav_file.name,
                'status': 'failed',
                'error': str(e)
            })
            failed += 1
    
    # Save summary
    print()
    print("="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Total files: {len(wav_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("="*60)
    
    summary_file = Path(args.output_dir) / 'batch_summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_files': len(wav_files),
            'successful': successful,
            'failed': failed,
            'results': results_summary
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {summary_file}")
    print()
    
    # Print individual results locations
    print("Individual results:")
    for result in results_summary:
        if result['status'] == 'success':
            print(f"  ✓ {result['file']}: {result['output_dir']}")
        else:
            print(f"  ✗ {result['file']}: {result.get('error', 'Unknown error')}")
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())

