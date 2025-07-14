#!/usr/bin/env python3
"""
Test script to verify the integration of all components
"""

import socket
import json
import time
import subprocess
import sys

def test_connector_communication():
    """Test communication with the hivemind connector"""
    print("Testing Hivemind Connector Communication...")
    
    try:
        # Connect to the hivemind connector
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(('localhost', 10001))
        
        # Test status command
        print("\n1. Testing STATUS command:")
        request = {'command': 'status'}
        client.send(json.dumps(request).encode('utf-8'))
        response = json.loads(client.recv(4096).decode('utf-8'))
        print(f"   Status: {response['status']}")
        print(f"   Scripts discovered: {response.get('discovered_scripts', {})}")
        
        # Test get_scripts command
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('localhost', 10001))
        print("\n2. Testing GET_SCRIPTS command:")
        request = {'command': 'get_scripts'}
        client.send(json.dumps(request).encode('utf-8'))
        response = json.loads(client.recv(4096).decode('utf-8'))
        print(f"   Available scripts: {response.get('scripts', {})}")
        
        # Test parameter access
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('localhost', 10001))
        print("\n3. Testing GET_PARAMETER command:")
        request = {
            'command': 'get_parameter',
            'script': 'speed-test',
            'parameter': 'PROCESSING_CONFIG'
        }
        client.send(json.dumps(request).encode('utf-8'))
        response = json.loads(client.recv(4096).decode('utf-8'))
        print(f"   Parameter value: {response.get('value')}")
        
        # Test parameter setting
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('localhost', 10001))
        print("\n4. Testing SET_PARAMETER command:")
        request = {
            'command': 'set_parameter',
            'script': 'speed-test',
            'parameter': 'PROCESSING_CONFIG',
            'value': {'force_cpu': True, 'confidence_threshold': 0.9}
        }
        client.send(json.dumps(request).encode('utf-8'))
        response = json.loads(client.recv(4096).decode('utf-8'))
        print(f"   Set parameter success: {response.get('success')}")
        
        print("\n✅ Connector communication test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Connector communication test failed: {e}")
        return False

def test_script_independence():
    """Test that scripts can run independently"""
    print("\n\nTesting Script Independence...")
    
    scripts = [
        ('polar_bear_master.py', ['--test']),
        ('speed-test.py', ['--test']),
    ]
    
    for script, args in scripts:
        print(f"\nTesting {script}:")
        try:
            # Run with a short timeout
            result = subprocess.run(
                [sys.executable, script] + args,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print(f"   ✅ {script} runs independently")
            else:
                print(f"   ⚠️  {script} returned error: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  {script} timed out (may be running interactively)")
        except Exception as e:
            print(f"   ❌ {script} failed: {e}")

def test_collaboration():
    """Test collaboration between scripts through the connector"""
    print("\n\nTesting Script Collaboration...")
    
    try:
        # Connect to connector
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('localhost', 10001))
        
        # Execute a function from speed-test
        print("\n1. Testing function execution across scripts:")
        request = {
            'command': 'execute_function',
            'script': 'speed-test',
            'function': 'get_pipeline_status',
            'args': [],
            'kwargs': {}
        }
        client.send(json.dumps(request).encode('utf-8'))
        response = json.loads(client.recv(4096).decode('utf-8'))
        print(f"   Function execution status: {response.get('status')}")
        print(f"   Result: {response.get('result', 'N/A')[:100]}...")
        
        print("\n✅ Collaboration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Collaboration test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("POLAR BEAR INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Check if connector is running
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.connect(('localhost', 10001))
        test_socket.close()
        print("✅ Hivemind connector is running on port 10001")
    except:
        print("❌ Hivemind connector is not running!")
        print("   Please start it with: python3 hivemind_connector.py")
        return
    
    # Run tests
    tests_passed = 0
    tests_total = 3
    
    if test_connector_communication():
        tests_passed += 1
    
    if test_script_independence():
        tests_passed += 1
    
    if test_collaboration():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"INTEGRATION TEST SUMMARY: {tests_passed}/{tests_total} passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("\n🎉 All integration tests passed! The system is fully integrated.")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")

if __name__ == "__main__":
    main()