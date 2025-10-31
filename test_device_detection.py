"""
Test script to verify GPU/CPU detection for speech-to-text models
"""

import torch

def check_device():
    """Check and display device information"""
    print("=" * 60)
    print("Device Detection Test")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\n🔍 Checking for CUDA support...")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"\n🚀 GPU Information:")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"     Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"     Total Memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"     Compute Capability: {props.major}.{props.minor}")
            print(f"     Multi-Processor Count: {props.multi_processor_count}")
        
        # Current device
        current_device = torch.cuda.current_device()
        print(f"\n   Current Device: GPU {current_device}")
        
        # Memory info
        print(f"\n   Memory Status:")
        print(f"     Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        print(f"     Cached: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
        
        device = torch.device("cuda")
        device_info = f"🚀 GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / (1024**3):.1f} GB)"
    else:
        print(f"\n💻 CPU Information:")
        print(f"   PyTorch Threads: {torch.get_num_threads()}")
        print(f"   Number of CPUs: {torch.multiprocessing.cpu_count()}")
        
        device = torch.device("cpu")
        device_info = f"💻 CPU: {torch.get_num_threads()} threads"
    
    print(f"\n✅ Selected Device: {device}")
    print(f"   Device Info: {device_info}")
    
    # Test tensor operation
    print(f"\n🧪 Testing tensor operations...")
    test_tensor = torch.randn(1000, 1000).to(device)
    result = torch.matmul(test_tensor, test_tensor)
    print(f"   ✅ Tensor operation successful on {device}")
    print(f"   Tensor shape: {result.shape}")
    print(f"   Tensor device: {result.device}")
    
    # Clean up
    del test_tensor, result
    if cuda_available:
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    
    return device, device_info

if __name__ == "__main__":
    device, info = check_device()
    
    print(f"\n💡 Your system will use: {info}")
    
    if torch.cuda.is_available():
        print("\n🎉 GPU acceleration is available!")
        print("   Your models will run faster with GPU support.")
    else:
        print("\n⚠️  No GPU detected. Models will run on CPU.")
        print("   Consider using a GPU for faster inference.")

