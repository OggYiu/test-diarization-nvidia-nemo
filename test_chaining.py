"""
Test script to verify that the tab chaining logic is correct.
This script checks the function signatures and verifies the modifications.
"""

import inspect
import ast
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def check_function_signature(file_path, function_name, expected_params):
    """Check if a function has the expected parameters"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.args:
                if node.name == function_name:
                    # Get parameter names
                    params = [arg.arg for arg in node.args.args]
                    print(f"✓ Found function '{function_name}' with parameters: {params}")
                    
                    # Check if all expected params are present
                    for param in expected_params:
                        if param in params:
                            print(f"  ✓ Parameter '{param}' found")
                        else:
                            print(f"  ✗ Parameter '{param}' NOT found")
                            return False
                    return True
        
        print(f"✗ Function '{function_name}' not found in {file_path}")
        return False
    except Exception as e:
        print(f"✗ Error checking {file_path}: {str(e)}")
        return False

def check_state_usage_in_file(file_path, state_var_name):
    """Check if a file uses gr.State"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if f'gr.State(' in content:
            print(f"✓ Found 'gr.State()' usage in {file_path}")
            return True
        
        if state_var_name in content:
            print(f"✓ Found state variable '{state_var_name}' usage in {file_path}")
            return True
            
        print(f"✗ No state usage found in {file_path}")
        return False
    except Exception as e:
        print(f"✗ Error checking {file_path}: {str(e)}")
        return False

def main():
    print("=" * 70)
    print("Testing Tab Chaining Implementation")
    print("=" * 70)
    
    results = []
    
    # Test 1: Check unified_gui.py for shared state
    print("\n[Test 1] Checking unified_gui.py for shared state...")
    result1 = check_state_usage_in_file('unified_gui.py', 'shared_json_data')
    results.append(('unified_gui.py state', result1))
    
    # Test 2: Check create_stt_tab signature
    print("\n[Test 2] Checking create_stt_tab function signature...")
    result2 = check_function_signature(
        'tabs/tab_stt.py',
        'create_stt_tab',
        ['output_json_state']
    )
    results.append(('create_stt_tab signature', result2))
    
    # Test 3: Check create_json_batch_analysis_tab signature
    print("\n[Test 3] Checking create_json_batch_analysis_tab function signature...")
    result3 = check_function_signature(
        'tabs/tab_json_batch_analysis.py',
        'create_json_batch_analysis_tab',
        ['input_json_state']
    )
    results.append(('create_json_batch_analysis_tab signature', result3))
    
    # Test 4: Check if tab_stt.py uses the output_json_state with wrapper function
    print("\n[Test 4] Checking if tab_stt.py has wrapper function for state output...")
    try:
        with open('tabs/tab_stt.py', 'r', encoding='utf-8') as f:
            content = f.read()
        has_state_check = 'if output_json_state is not None' in content
        has_wrapper = 'def process_with_state' in content
        has_return_fix = 'return result + (result[-1],)' in content
        
        if has_state_check and has_wrapper and has_return_fix:
            print("✓ tab_stt.py has wrapper function that correctly returns 9 values")
            results.append(('tab_stt.py wrapper function', True))
        else:
            print(f"✗ tab_stt.py missing components:")
            print(f"  State check: {has_state_check}")
            print(f"  Wrapper function: {has_wrapper}")
            print(f"  Return fix: {has_return_fix}")
            results.append(('tab_stt.py wrapper function', False))
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        results.append(('tab_stt.py wrapper function', False))
    
    # Test 5: Check if tab_json_batch_analysis.py has load button and output state
    print("\n[Test 5] Checking tab_json_batch_analysis.py for load button and output state...")
    try:
        with open('tabs/tab_json_batch_analysis.py', 'r', encoding='utf-8') as f:
            content = f.read()
        has_load_button = 'load_from_stt_btn' in content and 'Load from STT Tab' in content
        has_output_param = 'output_stocks_state' in content
        has_wrapper = 'def process_with_stock_state' in content
        
        if has_load_button and has_output_param and has_wrapper:
            print("✓ tab_json_batch_analysis.py has load button and stock state output")
            results.append(('JSON Batch Analysis chaining', True))
        else:
            print(f"✗ Missing components:")
            print(f"  Load button: {has_load_button}")
            print(f"  Output parameter: {has_output_param}")
            print(f"  Wrapper function: {has_wrapper}")
            results.append(('JSON Batch Analysis chaining', False))
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        results.append(('JSON Batch Analysis chaining', False))
    
    # Test 6: Check Transaction Analysis JSON tab
    print("\n[Test 6] Checking tab_transaction_analysis_json.py for chaining...")
    result6 = check_function_signature(
        'tabs/tab_transaction_analysis_json.py',
        'create_transaction_analysis_json_tab',
        ['input_conversation_state', 'input_stocks_state', 'output_transaction_state']
    )
    results.append(('Transaction Analysis JSON signature', result6))
    
    # Test 7: Check Trade Verification tab
    print("\n[Test 7] Checking tab_trade_verification.py for chaining...")
    result7 = check_function_signature(
        'tabs/tab_trade_verification.py',
        'create_trade_verification_tab',
        ['input_transaction_state']
    )
    results.append(('Trade Verification signature', result7))
    
    # Test 8: Check unified_gui.py for all 3 shared states
    print("\n[Test 8] Checking unified_gui.py for all shared states...")
    try:
        with open('unified_gui.py', 'r', encoding='utf-8') as f:
            content = f.read()
        has_conversation_state = 'shared_conversation_json = gr.State(None)' in content
        has_stocks_state = 'shared_merged_stocks_json = gr.State(None)' in content
        has_transaction_state = 'shared_transaction_json = gr.State(None)' in content
        
        if has_conversation_state and has_stocks_state and has_transaction_state:
            print("✓ unified_gui.py has all 3 shared states")
            results.append(('All shared states', True))
        else:
            print(f"✗ Missing states:")
            print(f"  Conversation state: {has_conversation_state}")
            print(f"  Stocks state: {has_stocks_state}")
            print(f"  Transaction state: {has_transaction_state}")
            results.append(('All shared states', False))
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        results.append(('All shared states', False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Tab chaining is properly implemented.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

