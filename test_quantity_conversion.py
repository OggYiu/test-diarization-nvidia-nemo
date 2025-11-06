"""
Test script to verify quantity conversion from Chinese text to digits
"""

import re
import logging

def convert_chinese_number_to_digit(text: str) -> str:
    """
    Convert Chinese numerals and text quantities to numeric digits.
    
    Examples:
        "一千" -> "1000"
        "兩萬" -> "20000"
        "10手" -> "10"
        "1000股" -> "1000"
        "三百五十" -> "350"
        "5萬股" -> "50000"
        
    Args:
        text: Input text containing numbers (Chinese or Arabic)
        
    Returns:
        str: Numeric value as string, or original text if conversion fails
    """
    if not text:
        return ""
    
    # If already a pure number, return it
    if text.isdigit():
        return text
    
    # Chinese number mappings
    chinese_digits = {
        '零': 0, '〇': 0,
        '一': 1, '壹': 1,
        '二': 2, '貳': 2, '兩': 2, '两': 2,
        '三': 3, '參': 3, '叁': 3,
        '四': 4, '肆': 4,
        '五': 5, '伍': 5,
        '六': 6, '陸': 6,
        '七': 7, '柒': 7,
        '八': 8, '捌': 8,
        '九': 9, '玖': 9,
    }
    
    chinese_units = {
        '十': 10, '拾': 10,
        '百': 100, '佰': 100,
        '千': 1000, '仟': 1000,
        '萬': 10000, '万': 10000,
        '億': 100000000, '亿': 100000000,
    }
    
    # Try to extract existing Arabic numerals first
    arabic_match = re.search(r'(\d+(?:\.\d+)?)', text)
    if arabic_match:
        num_str = arabic_match.group(1)
        # Check if there's a Chinese unit after the number (e.g., "5萬")
        remaining = text[arabic_match.end():]
        for unit_char, unit_val in chinese_units.items():
            if unit_char in remaining:
                try:
                    return str(int(float(num_str) * unit_val))
                except:
                    pass
        return num_str
    
    # Convert pure Chinese numerals
    try:
        # Remove common suffixes like 股, 手, 張, etc.
        cleaned = re.sub(r'[股手張张块塊元蚊]', '', text).strip()
        
        if not any(c in cleaned for c in chinese_digits.keys() | chinese_units.keys()):
            return text  # No Chinese numerals found
        
        total = 0
        current = 0
        
        i = 0
        while i < len(cleaned):
            char = cleaned[i]
            
            if char in chinese_digits:
                current = chinese_digits[char]
                i += 1
            elif char in chinese_units:
                unit = chinese_units[char]
                if unit >= 10000:  # 萬 or 億
                    total = (total + current) * unit
                    current = 0
                else:  # 十, 百, 千
                    if current == 0:
                        current = 1  # Handle cases like "十" meaning "10"
                    total += current * unit
                    current = 0
                i += 1
            else:
                i += 1
        
        total += current
        return str(int(total)) if total > 0 else text
        
    except Exception as e:
        logging.warning(f"Failed to convert Chinese number '{text}': {e}")
        return text

# Test cases
test_cases = [
    # Format: (input, expected_output, description)
    ("一千", "1000", "Pure Chinese - one thousand"),
    ("兩萬", "20000", "Pure Chinese - twenty thousand"),
    ("三百五十", "350", "Pure Chinese - three hundred fifty"),
    ("十", "10", "Pure Chinese - ten"),
    ("五十", "50", "Pure Chinese - fifty"),
    ("一百", "100", "Pure Chinese - one hundred"),
    ("1000股", "1000", "Arabic with Chinese suffix"),
    ("10手", "10", "Arabic with Chinese suffix"),
    ("5萬股", "50000", "Mixed Arabic and Chinese unit"),
    ("100張", "100", "Arabic with Chinese suffix"),
    ("1000", "1000", "Pure Arabic number"),
    ("三千五百", "3500", "Pure Chinese - three thousand five hundred"),
    ("兩千", "2000", "Pure Chinese - two thousand"),
    ("五萬", "50000", "Pure Chinese - fifty thousand"),
    ("一萬五千", "15000", "Pure Chinese - fifteen thousand"),
    ("二十萬", "200000", "Pure Chinese - two hundred thousand"),
]

print("=" * 80)
print("Testing Quantity Conversion: Chinese/Text to Numeric Digits")
print("=" * 80)
print()

passed = 0
failed = 0

for input_text, expected, description in test_cases:
    result = convert_chinese_number_to_digit(input_text)
    status = "✅ PASS" if result == expected else "❌ FAIL"
    
    if result == expected:
        passed += 1
    else:
        failed += 1
    
    print(f"{status} | {description}")
    print(f"   Input:    '{input_text}'")
    print(f"   Expected: '{expected}'")
    print(f"   Got:      '{result}'")
    print()

print("=" * 80)
print(f"Test Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("=" * 80)

if failed == 0:
    print("\n✅ All tests passed! Quantity conversion is working correctly.")
else:
    print(f"\n⚠️ {failed} test(s) failed. Please review the conversion logic.")

