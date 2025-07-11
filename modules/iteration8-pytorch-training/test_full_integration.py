#!/usr/bin/env python3
"""
Comprehensive Integration Test
Tests all integration features between connectors and scripts
"""

import socket
import json
import time
import subprocess
import threading
import os
import sys

def send_command(port, command):
    """Send command to connector and get response"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', port))
        sock.send(json.dumps(command).encode())
        response = sock.recv(8192).decode()
        sock.close()
        return json.loads(response)
    except Exception as e:
        return {'error': str(e)}

def test_connector_integration():
    """Test the enhanced connector integration"""
    print("="*60)
    print("TESTING ENHANCED CONNECTOR INTEGRATION")
    print("="*60)
    
    # Test 1: Basic status
    print("\n1. Testing basic status...")
    response = send_command(10051, {'type': 'status'})
    print(f"Status response: {json.dumps(response, indent=2)}")
    
    # Test 2: Enhanced integration status
    print("\n2. Testing enhanced integration...")
    response = send_command(10051, {
        'type': 'enhanced_control',
        'action': 'get_all_scripts'
    })
    print(f"Enhanced scripts: {json.dumps(response, indent=2)}")
    
    # Test 3: Set parameters
    print("\n3. Testing parameter control...")
    response = send_command(10051, {
        'type': 'enhanced_control',
        'action': 'set_parameter',
        'script': 'train.py',
        'parameter': 'LEARNING_RATE',
        'value': 0.05
    })
    print(f"Set parameter response: {json.dumps(response, indent=2)}")
    
    # Test 4: Get script info
    print("\n4. Testing script info retrieval...")
    response = send_command(10051, {
        'type': 'enhanced_control',
        'action': 'get_script_info',
        'script': 'train.py'
    })
    print(f"Script info: {json.dumps(response, indent=2)}")
    
    # Test 5: Call function
    print("\n5. Testing function execution...")
    response = send_command(10051, {
        'type': 'enhanced_control',
        'action': 'call_function',
        'script': 'preprocess.py',
        'function': 'initialize_generator'
    })
    print(f"Function call response: {json.dumps(response, indent=2)}")
    
    # Test 6: Publish event
    print("\n6. Testing event publishing...")
    response = send_command(10051, {
        'type': 'enhanced_control',
        'action': 'publish_event',
        'event_type': 'info',
        'data': {'message': 'Test event from integration test'}
    })
    print(f"Event publish response: {json.dumps(response, indent=2)}")

def test_hivemind_integration():
    """Test the hivemind connector integration"""
    print("\n" + "="*60)
    print("TESTING HIVEMIND CONNECTOR INTEGRATION")
    print("="*60)
    
    # Test 1: Basic status
    print("\n1. Testing hivemind status...")
    response = send_command(10050, {'command': 'status'})
    print(f"Hivemind status: {json.dumps(response, indent=2)}")
    
    # Test 2: Enhanced control
    print("\n2. Testing hivemind enhanced control...")
    response = send_command(10050, {
        'command': 'enhanced_control',
        'action': 'get_script_info',
        'script': 'train.py'
    })
    print(f"Hivemind script info: {json.dumps(response, indent=2)}")
    
    # Test 3: Broadcast event
    print("\n3. Testing event broadcast...")
    response = send_command(10050, {
        'command': 'enhanced_control',
        'action': 'broadcast_event',
        'event_type': 'status',
        'data': {'test': 'broadcast from hivemind'}
    })
    print(f"Broadcast response: {json.dumps(response, indent=2)}")

def test_script_independence():
    """Test that scripts can still run independently"""
    print("\n" + "="*60)
    print("TESTING SCRIPT INDEPENDENCE")
    print("="*60)
    
    # Test each script can run independently
    scripts = ['preprocess.py', 'load.py', 'train.py', 'final.py']
    
    for script in scripts:
        if os.path.exists(script):
            print(f"\n Testing {script} independence...")
            try:
                # Run with --help or a test flag
                result = subprocess.run(
                    [sys.executable, script, '--help'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 or 'usage:' in result.stdout.lower():
                    print(f"✓ {script} can run independently")
                else:
                    # Try running without arguments
                    print(f"  {script} may require input or setup")
            except subprocess.TimeoutExpired:
                print(f"  {script} timed out (may be waiting for input)")
            except Exception as e:
                print(f"✗ Error testing {script}: {e}")

def test_collaboration():
    """Test collaboration between scripts through connectors"""
    print("\n" + "="*60)
    print("TESTING SCRIPT COLLABORATION")
    print("="*60)
    
    # Test 1: Shared state
    print("\n1. Testing shared state between scripts...")
    
    # Set a parameter in one script
    response = send_command(10051, {
        'type': 'enhanced_control',
        'action': 'set_parameter',
        'script': 'train.py',
        'parameter': 'IMG_SIZE',
        'value': 256
    })
    print(f"Set IMG_SIZE in train.py: {response}")
    
    # Check if it's reflected in other scripts
    response = send_command(10051, {
        'type': 'enhanced_control',
        'action': 'get_script_info',
        'script': 'final.py'
    })
    if response.get('info', {}).get('parameters', {}).get('IMG_SIZE') == 256:
        print("✓ Shared state working - parameter propagated")
    else:
        print("✗ Shared state not working properly")
    
    # Test 2: Event propagation
    print("\n2. Testing event propagation...")
    response = send_command(10051, {
        'type': 'enhanced_control',
        'action': 'get_events'
    })
    print(f"Current events: {json.dumps(response, indent=2)}")

def main():
    """Run all integration tests"""
    print("FULL INTEGRATION TEST SUITE")
    print("="*60)
    print("This test assumes connectors are running on their default ports:")
    print("- Enhanced Connector: 10051")
    print("- Hivemind Connector: 10050")
    print("\nStarting tests in 3 seconds...")
    time.sleep(3)
    
    try:
        # Test enhanced connector
        test_connector_integration()
        
        # Test hivemind connector
        test_hivemind_integration()
        
        # Test script independence
        test_script_independence()
        
        # Test collaboration
        test_collaboration()
        
        print("\n" + "="*60)
        print("INTEGRATION TEST SUITE COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Test suite failed - {e}")
        print("Make sure connectors are running:")
        print("  python connector.py --server")
        print("  python hivemind_connector.py")

if __name__ == "__main__":
    main()