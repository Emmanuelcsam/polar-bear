#!/usr/bin/env python3
import requests
import json

# Test the API endpoint
print("Testing Guitar Tab Reader Integration...")
print("=" * 50)

# Test 1: Check if server is running
try:
    response = requests.get("http://localhost:5001/api/test")
    if response.status_code == 200:
        print("✓ Server is running")
        print(f"  Response: {response.json()}")
    else:
        print(f"✗ Server test failed: {response.status_code}")
except Exception as e:
    print(f"✗ Cannot connect to server: {e}")
    exit(1)

# Test 2: Test file upload
print("\nTesting file upload...")
try:
    with open('test_guitar_tab.png', 'rb') as f:
        files = {'file': f}
        response = requests.post("http://localhost:5001/api/upload", files=files)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print("✓ File upload successful")
            print(f"  Extracted {len(data['data']['measures'])} measure(s)")
            print(f"  Total notes: {sum(len(m['notes']) for m in data['data']['measures'])}")
        else:
            print(f"✗ File processing failed: {data.get('error')}")
    else:
        print(f"✗ Upload failed: {response.status_code}")
except Exception as e:
    print(f"✗ Upload test failed: {e}")

# Test 3: Check HTML page
print("\nChecking HTML page...")
try:
    response = requests.get("http://localhost:5001/")
    if response.status_code == 200 and '<title>Guitar Tab Visualizer' in response.text:
        print("✓ HTML page is being served correctly")
    else:
        print("✗ HTML page not found or incorrect")
except Exception as e:
    print(f"✗ HTML check failed: {e}")

print("\n" + "=" * 50)
print("Integration test complete!")