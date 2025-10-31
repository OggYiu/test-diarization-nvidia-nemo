"""
Test script for transcription caching functionality
"""

import os
from datetime import datetime
from tabs.tab_stt import (
    load_transcription_cache_sensevoice,
    save_transcription_to_cache_sensevoice,
    load_transcription_cache_whisperv3,
    save_transcription_to_cache_whisperv3,
    TRANSCRIPTION_SENSEVOICE_COLLECTION,
    TRANSCRIPTION_WHISPERV3_COLLECTION
)
from mongodb_utils import count_documents, delete_from_mongodb

def test_sensevoice_cache():
    """Test SenseVoiceSmall cache operations"""
    print("=" * 60)
    print("Testing SenseVoiceSmall Cache")
    print("=" * 60)
    
    # Clean up test data first
    delete_from_mongodb(TRANSCRIPTION_SENSEVOICE_COLLECTION, {'filename': 'test_audio.wav'})
    
    # Test saving to cache
    print("\n1. Saving test transcription to cache...")
    save_transcription_to_cache_sensevoice(
        filename='test_audio.wav',
        transcription='這是一個測試',
        raw_transcription='<|zh|><|NEUTRAL|><|Speech|><|woitn|>這是一個測試',
        language='yue',
        processing_time=2.5
    )
    
    # Test loading from cache
    print("\n2. Loading from cache...")
    cache = load_transcription_cache_sensevoice()
    
    if 'test_audio.wav' in cache:
        print("✅ Cache entry found!")
        cached_data = cache['test_audio.wav']
        print(f"   Transcription: {cached_data['transcription']}")
        print(f"   Language: {cached_data['language']}")
        print(f"   Processing time: {cached_data['processing_time']}s")
        print(f"   Timestamp: {cached_data['timestamp']}")
    else:
        print("❌ Cache entry not found!")
    
    # Test cache count
    count = count_documents(TRANSCRIPTION_SENSEVOICE_COLLECTION)
    print(f"\n3. Total SenseVoice cache entries: {count}")
    
    # Clean up
    delete_from_mongodb(TRANSCRIPTION_SENSEVOICE_COLLECTION, {'filename': 'test_audio.wav'})
    print("\n4. Test data cleaned up")


def test_whisperv3_cache():
    """Test Whisper-v3-Cantonese cache operations"""
    print("\n" + "=" * 60)
    print("Testing Whisper-v3-Cantonese Cache")
    print("=" * 60)
    
    # Clean up test data first
    delete_from_mongodb(TRANSCRIPTION_WHISPERV3_COLLECTION, {'filename': 'test_audio.wav'})
    
    # Test saving to cache
    print("\n1. Saving test transcription to cache...")
    save_transcription_to_cache_whisperv3(
        filename='test_audio.wav',
        transcription='呢個係測試',
        processing_time=3.2
    )
    
    # Test loading from cache
    print("\n2. Loading from cache...")
    cache = load_transcription_cache_whisperv3()
    
    if 'test_audio.wav' in cache:
        print("✅ Cache entry found!")
        cached_data = cache['test_audio.wav']
        print(f"   Transcription: {cached_data['transcription']}")
        print(f"   Processing time: {cached_data['processing_time']}s")
        print(f"   Timestamp: {cached_data['timestamp']}")
    else:
        print("❌ Cache entry not found!")
    
    # Test cache count
    count = count_documents(TRANSCRIPTION_WHISPERV3_COLLECTION)
    print(f"\n3. Total Whisper-v3 cache entries: {count}")
    
    # Clean up
    delete_from_mongodb(TRANSCRIPTION_WHISPERV3_COLLECTION, {'filename': 'test_audio.wav'})
    print("\n4. Test data cleaned up")


def show_cache_stats():
    """Display current cache statistics"""
    print("\n" + "=" * 60)
    print("Cache Statistics")
    print("=" * 60)
    
    sensevoice_count = count_documents(TRANSCRIPTION_SENSEVOICE_COLLECTION)
    whisperv3_count = count_documents(TRANSCRIPTION_WHISPERV3_COLLECTION)
    
    print(f"\nSenseVoiceSmall cache entries: {sensevoice_count}")
    print(f"Whisper-v3-Cantonese cache entries: {whisperv3_count}")
    print(f"Total transcription cache entries: {sensevoice_count + whisperv3_count}")


if __name__ == "__main__":
    print("Transcription Cache Test Suite")
    print("Starting at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()
    
    try:
        # Run tests
        test_sensevoice_cache()
        test_whisperv3_cache()
        show_cache_stats()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()

