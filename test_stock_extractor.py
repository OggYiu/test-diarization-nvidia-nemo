"""
Test/Demo Script for Stock Extractor
Run this to test the stock extraction functionality without the GUI
"""

from stock_extractor_gui import extract_stocks_from_conversation, DEFAULT_SYSTEM_MESSAGE

# Test conversations
TEST_CONVERSATIONS = {
    "simple": """券商：你好，請問需要什麼幫助？
客戶：我想買騰訊
券商：好的，七百號騰訊，買多少？
客戶：一千股，市價買入
券商：確認一下，七百號騰訊，買入一千股，市價，對嗎？
客戶：對的，謝謝""",
    
    "multiple_stocks": """客戶：早晨，我想問下小米同比亞迪今日走勢
券商：你好！小米一八一零今日升咗2%，比亞迪二一一一跌咗1%
客戶：咁我想沽五百股比亞迪，再入一千股小米
券商：好的，確認一下：沽出比亞迪二一一一五百股，買入小米一八一零一千股，啱唔啱？
客戶：啱，就咁做""",
    
    "with_errors": """客戶：我想買招商局置地
券商：招商局置地，係一百一三八號？
客戶：係呀
券商：買幾多？
客戶：五百股
券商：確認：一百一三八號招商局置地，買入五百股
客戶：正確"""
}


def test_extraction(conversation_name: str, model: str = "qwen3:32b"):
    """
    Test stock extraction with a specific conversation
    
    Args:
        conversation_name: Key from TEST_CONVERSATIONS
        model: LLM model to use
    """
    print("\n" + "=" * 80)
    print(f"Testing: {conversation_name}")
    print("=" * 80)
    
    conversation = TEST_CONVERSATIONS.get(conversation_name)
    if not conversation:
        print(f"❌ Conversation '{conversation_name}' not found!")
        return
    
    print("\n📝 Input Conversation:")
    print("-" * 80)
    print(conversation)
    print("-" * 80)
    
    print(f"\n🔄 Extracting with model: {model}...")
    
    try:
        result, json_output = extract_stocks_from_conversation(
            conversation_text=conversation,
            model=model,
            ollama_url="http://localhost:11434",
            system_message=DEFAULT_SYSTEM_MESSAGE,
            temperature=0.1,
        )
        
        print("\n" + result)
        
        print("\n🔧 Raw JSON Output:")
        print("-" * 80)
        print(json_output)
        print("-" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def test_all_conversations(model: str = "qwen3:32b"):
    """Test all conversations with the specified model"""
    print("\n" + "🎯" * 40)
    print(f"   TESTING ALL CONVERSATIONS WITH MODEL: {model}")
    print("🎯" * 40)
    
    for conv_name in TEST_CONVERSATIONS.keys():
        test_extraction(conv_name, model)
        print("\n")


def compare_models(conversation_name: str = "multiple_stocks"):
    """Compare results from different models on the same conversation"""
    models = ["qwen3:32b", "deepseek-r1:32b"]
    
    print("\n" + "🔬" * 40)
    print(f"   MODEL COMPARISON TEST: {conversation_name}")
    print("🔬" * 40)
    
    for model in models:
        test_extraction(conversation_name, model)
        print("\n" + "─" * 80 + "\n")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "all":
            # Test all conversations with default model
            model = sys.argv[2] if len(sys.argv) > 2 else "qwen3:32b"
            test_all_conversations(model)
            
        elif command == "compare":
            # Compare models
            conv_name = sys.argv[2] if len(sys.argv) > 2 else "multiple_stocks"
            compare_models(conv_name)
            
        elif command in TEST_CONVERSATIONS:
            # Test specific conversation
            model = sys.argv[2] if len(sys.argv) > 2 else "qwen3:32b"
            test_extraction(command, model)
            
        else:
            print(f"❌ Unknown command: {command}")
            print("\nUsage:")
            print("  python test_stock_extractor.py [command] [options]")
            print("\nCommands:")
            print("  all [model]              - Test all conversations (default model: qwen3:32b)")
            print("  compare [conversation]   - Compare different models (default: multiple_stocks)")
            print("  simple [model]          - Test simple conversation")
            print("  multiple_stocks [model] - Test multiple stocks conversation")
            print("  with_errors [model]     - Test conversation with STT errors")
            print("\nExamples:")
            print("  python test_stock_extractor.py all")
            print("  python test_stock_extractor.py all qwen2.5:72b")
            print("  python test_stock_extractor.py simple deepseek-r1:32b")
            print("  python test_stock_extractor.py compare multiple_stocks")
    else:
        # Default: run simple test
        print("\n💡 TIP: Run with arguments for more options (try 'python test_stock_extractor.py help')")
        print("\nRunning default test (simple conversation with qwen3:32b)...\n")
        test_extraction("simple", "qwen3:32b")

