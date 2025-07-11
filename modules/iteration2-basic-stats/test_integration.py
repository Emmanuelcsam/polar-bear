#!/usr/bin/env python3
"""
Integration test for the enhanced connector system
Tests both independent operation and connector-controlled operation
"""

import os
import sys
import json
import socket
import time
import subprocess
import threading
from pathlib import Path

def test_connector_system():
    """Test the main connector system"""
    print("\n=== Testing Connector System ===")
    
    # Import and test the connector
    from connector import ConnectorSystem
    
    connector = ConnectorSystem()
    connector.scan_scripts()
    connector.enable_collaboration()
    
    # Get system status
    status = connector.get_system_status()
    print(f"Scripts loaded: {status['scripts_loaded']}")
    print(f"Scripts: {list(status['scripts'].keys())}")
    print(f"Collaboration enabled: {status['collaboration_enabled']}")
    
    # Test script control
    print("\n--- Testing Script Control ---")
    
    # Test correlation-finder
    if 'correlation-finder' in connector.script_controllers:
        print("\nTesting correlation-finder:")
        
        # Execute with test data
        test_data = "1 10%\n2 20%\n3 15%\n4 30%\n5 25%"
        result = connector.control_script('correlation-finder', 'execute', {'input_data': test_data})
        print(f"Execution result: {result.get('success')}")
        if result.get('stdout'):
            print(f"Output: {result.get('stdout')}")
        
        # Get variables
        vars_result = connector.control_script('correlation-finder', 'get_variables')
        print(f"Variables: {vars_result}")
    
    # Test collaborative task
    print("\n--- Testing Collaborative Task ---")
    task = {
        'name': 'test_collaboration',
        'steps': [
            {
                'script': 'correlation-finder',
                'action': 'execute',
                'params': {'input_data': '1 10%\n2 20%\n3 30%\n4 40%\n5 50%'}
            }
        ]
    }
    
    task_result = connector.execute_collaborative_task(task)
    print(f"Collaborative task result: {json.dumps(task_result, indent=2)}")
    
    return True

def test_hivemind_connector():
    """Test the hivemind connector via socket communication"""
    print("\n=== Testing Hivemind Connector ===")
    
    try:
        # Connect to hivemind connector
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', 10113))
        
        # Test status command
        print("\n--- Testing Status Command ---")
        status_msg = json.dumps({'command': 'status'})
        sock.send(status_msg.encode())
        response = json.loads(sock.recv(4096).decode())
        print(f"Status response: {json.dumps(response, indent=2)}")
        
        # Test script info
        print("\n--- Testing Script Info ---")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 10113))
        info_msg = json.dumps({'command': 'get_script_info', 'script': 'correlation-finder'})
        sock.send(info_msg.encode())
        response = json.loads(sock.recv(4096).decode())
        print(f"Script info response: {json.dumps(response, indent=2)}")
        
        # Test script execution
        print("\n--- Testing Script Execution ---")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 10113))
        exec_msg = json.dumps({
            'command': 'control_script',
            'script': 'correlation-finder',
            'action': 'execute',
            'params': {'input_data': '1 5%\n2 10%\n3 15%\n4 20%\n5 25%'}
        })
        sock.send(exec_msg.encode())
        response = json.loads(sock.recv(4096).decode())
        print(f"Execution response: {json.dumps(response, indent=2)}")
        
        sock.close()
        return True
        
    except socket.error as e:
        print(f"Socket error: {e}")
        print("Make sure hivemind_connector.py is running")
        return False

def test_independent_scripts():
    """Test that scripts can run independently"""
    print("\n=== Testing Independent Script Operation ===")
    
    # Test correlation-finder
    print("\n--- Testing correlation-finder independently ---")
    test_input = "1 10%\n2 20%\n3 30%\n4 25%\n5 35%"
    
    try:
        result = subprocess.run(
            [sys.executable, 'correlation-finder.py'],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"Return code: {result.returncode}")
        print(f"Output:\n{result.stdout}")
        if result.stderr:
            print(f"Errors:\n{result.stderr}")
    except Exception as e:
        print(f"Error running correlation-finder: {e}")
    
    # Test intensity-matcher (with limited iterations)
    print("\n--- Testing intensity-matcher independently ---")
    # Create a test image if needed
    if not os.path.exists('test_image.jpg'):
        from PIL import Image
        import numpy as np
        test_img = Image.fromarray(np.random.randint(0, 256, (100, 100), dtype=np.uint8), 'L')
        test_img.save('test_image.jpg')
    
    try:
        # Test with specific intensity
        result = subprocess.run(
            [sys.executable, 'intensity-matcher.py', 'test_image.jpg', '128'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"Return code: {result.returncode}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Errors: {result.stderr}")
    except Exception as e:
        print(f"Error running intensity-matcher: {e}")
    
    return True

def test_collaboration():
    """Test collaboration between scripts"""
    print("\n=== Testing Script Collaboration ===")
    
    from connector import ConnectorSystem
    
    connector = ConnectorSystem()
    connector.scan_scripts()
    connector.enable_collaboration()
    
    # Test shared data
    print("\n--- Testing Shared Data ---")
    
    # Set shared data from one script
    connector.control_script('correlation-finder', 'execute', {
        'input_data': '1 10%\n2 20%\n3 30%\n4 40%\n5 50%'
    })
    
    # Check shared memory
    shared_data = connector.shared_memory['data']
    print(f"Shared data after correlation analysis: {json.dumps(shared_data, indent=2)}")
    
    # Test message passing
    print("\n--- Testing Message Passing ---")
    
    # Send a message
    connector.send_message('test_script', 'all', 'Test broadcast message')
    
    # Check messages
    messages = connector.receive_messages('correlation-finder')
    print(f"Messages received: {messages}")
    
    return True

def main():
    """Run all integration tests"""
    print("Starting Integration Tests for iteration2-basic-stats")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 4
    
    # Test 1: Connector System
    try:
        if test_connector_system():
            tests_passed += 1
            print("\n✓ Connector System test passed")
    except Exception as e:
        print(f"\n✗ Connector System test failed: {e}")
    
    # Test 2: Independent Scripts
    try:
        if test_independent_scripts():
            tests_passed += 1
            print("\n✓ Independent Scripts test passed")
    except Exception as e:
        print(f"\n✗ Independent Scripts test failed: {e}")
    
    # Test 3: Collaboration
    try:
        if test_collaboration():
            tests_passed += 1
            print("\n✓ Collaboration test passed")
    except Exception as e:
        print(f"\n✗ Collaboration test failed: {e}")
    
    # Test 4: Hivemind Connector (optional - requires running hivemind_connector.py)
    print("\n\nTo test hivemind connector:")
    print("1. Run in another terminal: python hivemind_connector.py")
    print("2. Then run: python test_integration.py --test-hivemind")
    
    if '--test-hivemind' in sys.argv:
        try:
            if test_hivemind_connector():
                tests_passed += 1
                print("\n✓ Hivemind Connector test passed")
        except Exception as e:
            print(f"\n✗ Hivemind Connector test failed: {e}")
    else:
        tests_total -= 1  # Don't count hivemind test if not requested
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    print("=" * 50)
    
    # Cleanup
    if os.path.exists('test_image.jpg'):
        os.remove('test_image.jpg')

if __name__ == "__main__":
    main()