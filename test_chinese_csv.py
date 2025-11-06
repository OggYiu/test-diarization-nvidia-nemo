"""
Test script to verify CSV writing with Chinese characters
"""
import csv
import sys

# Set console encoding to UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Test data with Chinese characters
test_data = [
    {
        "client_id": "P77197",
        "broker_id": "0489",
        "stock_name": "長和",
        "stock_code": "0001",
        "verification_summary": "買入 長和 股票"
    },
    {
        "client_id": "M9136",
        "broker_id": "0489", 
        "stock_name": "吉利汽車",
        "stock_code": "0175",
        "verification_summary": "賣出 吉利汽車 股票"
    }
]

# Test with utf-8-sig encoding (should work with Excel)
print("Writing test_chinese_utf8sig.csv with utf-8-sig encoding...")
with open('test_chinese_utf8sig.csv', 'w', encoding='utf-8-sig', newline='') as f:
    fieldnames = ["client_id", "broker_id", "stock_name", "stock_code", "verification_summary"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(test_data)
print("[OK] Done! Check test_chinese_utf8sig.csv")

# Test with plain utf-8 encoding (may not work with Excel)
print("\nWriting test_chinese_utf8.csv with utf-8 encoding...")
with open('test_chinese_utf8.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ["client_id", "broker_id", "stock_name", "stock_code", "verification_summary"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(test_data)
print("[OK] Done! Check test_chinese_utf8.csv")

print("\n" + "="*50)
print("Compare both files in Excel or Notepad++")
print("The utf-8-sig version should display Chinese correctly in Excel")
print("="*50)

