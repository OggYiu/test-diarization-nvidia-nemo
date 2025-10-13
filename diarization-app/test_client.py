"""
Simple test client for the Speaker Diarization API
"""
import requests
import sys
import json
from pathlib import Path


def test_diarization_api(audio_file_path, num_speakers=2, api_url="http://localhost:8000"):
    """
    Test the diarization API by uploading an audio file.
    
    Args:
        audio_file_path: Path to the audio file
        num_speakers: Number of speakers (default: 2)
        api_url: Base URL of the API (default: http://localhost:8000)
    """
    
    # Check if file exists
    if not Path(audio_file_path).exists():
        print(f"Error: Audio file not found: {audio_file_path}")
        return
    
    print(f"Testing Diarization API...")
    print(f"API URL: {api_url}")
    print(f"Audio file: {audio_file_path}")
    print(f"Number of speakers: {num_speakers}")
    print("-" * 60)
    
    # Test health check endpoint
    try:
        print("\n1. Testing health check endpoint...")
        response = requests.get(f"{api_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the API server is running!")
        return
    
    # Test diarization endpoint
    print("\n2. Testing diarization endpoint...")
    print("Uploading audio file... (this may take a while)")
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'audio_file': f}
            data = {'num_speakers': num_speakers}
            
            response = requests.post(
                f"{api_url}/diarize",
                files=files,
                data=data
            )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Diarization successful!")
            print(f"Audio filename: {result.get('audio_filename')}")
            print(f"Number of speakers: {result.get('num_speakers')}")
            print(f"Message: {result.get('message')}")
            print("\nRTTM Content:")
            print("-" * 60)
            print(result.get('rttm_content', 'No RTTM content'))
            print("-" * 60)
            
            # Optionally save RTTM to file
            output_file = Path(audio_file_path).stem + "_output.rttm"
            with open(output_file, 'w') as f:
                f.write(result.get('rttm_content', ''))
            print(f"\n✓ RTTM content saved to: {output_file}")
            
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Error during diarization: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <audio_file_path> [num_speakers] [api_url]")
        print("\nExample:")
        print("  python test_client.py audio.wav")
        print("  python test_client.py audio.wav 3")
        print("  python test_client.py audio.wav 2 http://localhost:8000")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    api_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
    
    test_diarization_api(audio_file, num_speakers, api_url)

