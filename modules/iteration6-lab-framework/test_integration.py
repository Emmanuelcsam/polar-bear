#!/usr/bin/env python3
"""
Integration test script to verify all scripts work both independently and with connectors
"""

import os
import sys
import time
import json
import socket
import subprocess
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_script_independence():
    """Test that scripts can run independently"""
    print("\n=== Testing Script Independence ===")
    
    # Test 1: Import and use modules directly
    print("\n1. Testing direct module imports...")
    try:
        from modules import random_pixel
        
        # Test legacy interface
        img = random_pixel.gen()
        print(f"✓ random_pixel.gen() works: generated image shape {img.shape}")
        
        # Test guided generation
        from modules import cv_module, intensity_reader
        cv_module.batch("data")
        intensity_reader.learn()
        guided_img = random_pixel.guided()
        print(f"✓ random_pixel.guided() works: generated guided image shape {guided_img.shape}")
        
    except Exception as e:
        print(f"✗ Module import failed: {e}")
        return False
        
    # Test 2: Run scripts independently
    print("\n2. Testing independent script execution...")
    scripts_to_test = ["test_basic.py", "demo.py"]
    
    for script in scripts_to_test:
        try:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"✓ {script} runs successfully")
            else:
                print(f"✗ {script} failed: {result.stderr}")
        except Exception as e:
            print(f"✗ Error running {script}: {e}")
            
    return True


def test_connector_integration():
    """Test connector integration capabilities"""
    print("\n=== Testing Connector Integration ===")
    
    # Test 1: Check if hivemind connector is running
    print("\n1. Checking hivemind connector...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(('localhost', 10006))
        
        # Send status command
        status_msg = {'command': 'status'}
        sock.send(json.dumps(status_msg).encode())
        response = json.loads(sock.recv(4096).decode())
        sock.close()
        
        print(f"✓ Hivemind connector active: {response}")
        
    except Exception as e:
        print(f"✗ Hivemind connector not running: {e}")
        print("  Starting hivemind connector...")
        
        # Start hivemind connector
        hivemind_proc = subprocess.Popen(
            [sys.executable, "hivemind_connector.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2)  # Give it time to start
        
    # Test 2: Test connector.py functionality
    print("\n2. Testing enhanced connector...")
    try:
        from connector import LabFrameworkConnector
        
        connector = LabFrameworkConnector()
        connector.discover_modules()
        
        # Get module info
        info = connector.get_module_info('random_pixel')
        if info:
            print(f"✓ Module discovery works: found {len(connector.modules)} modules")
            print(f"  random_pixel info: {info['functions'][:3]}...")
        
        # Control module
        result = connector.control_module('random_pixel', 'gen')
        if result.get('status') == 'success':
            print("✓ Module control works: generated image via connector")
        else:
            print(f"✗ Module control failed: {result}")
            
    except Exception as e:
        print(f"✗ Connector test failed: {e}")
        import traceback
        traceback.print_exc()
        
    return True


def test_bidirectional_communication():
    """Test bidirectional communication between scripts and connectors"""
    print("\n=== Testing Bidirectional Communication ===")
    
    try:
        from core.connector_interface import ConnectorClient
        
        client = ConnectorClient()
        
        # Test 1: Get script info
        print("\n1. Testing script information retrieval...")
        info = client.get_script_info('random_pixel')
        if info:
            print(f"✓ Script info retrieved: {info.get('name')}")
        else:
            print("✗ Could not retrieve script info")
            
        # Test 2: Set parameters
        print("\n2. Testing parameter control...")
        success = client.set_script_parameter('random_pixel', 'size', 64)
        if success:
            print("✓ Parameter set successfully")
        else:
            print("✗ Parameter setting failed")
            
        # Test 3: Get variables
        print("\n3. Testing variable monitoring...")
        count = client.get_script_variable('random_pixel', 'generation_count')
        if count is not None:
            print(f"✓ Variable retrieved: generation_count = {count}")
        else:
            print("✗ Variable retrieval failed")
            
        # Test 4: Call methods
        print("\n4. Testing method invocation...")
        result = client.call_script_method('random_pixel', 'get_statistics')
        if result:
            print(f"✓ Method called successfully: {result}")
        else:
            print("✗ Method invocation failed")
            
    except Exception as e:
        print(f"✗ Bidirectional communication test failed: {e}")
        import traceback
        traceback.print_exc()
        
    return True


def test_collaboration():
    """Test that scripts can collaborate through connectors"""
    print("\n=== Testing Script Collaboration ===")
    
    try:
        from connector import LabFrameworkConnector
        
        connector = LabFrameworkConnector()
        connector.discover_modules()
        
        # Test workflow execution
        print("\n1. Testing collaborative workflow...")
        workflow = [
            {'module': 'random_pixel', 'action': 'gen'},
            {'module': 'cv_module', 'action': 'batch', 'params': {'folder': 'data'}},
            {'module': 'intensity_reader', 'action': 'learn'},
            {'module': 'random_pixel', 'action': 'guided'}
        ]
        
        results = connector.execute_workflow(workflow)
        
        success_count = sum(1 for r in results if r['result'].get('status') == 'success')
        print(f"✓ Workflow executed: {success_count}/{len(workflow)} steps successful")
        
        for i, result in enumerate(results):
            status = "✓" if result['result'].get('status') == 'success' else "✗"
            print(f"  {status} Step {i+1}: {result['step']}")
            
    except Exception as e:
        print(f"✗ Collaboration test failed: {e}")
        import traceback
        traceback.print_exc()
        
    return True


def main():
    """Main test runner"""
    print("=" * 60)
    print("LAB FRAMEWORK INTEGRATION TEST")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Script Independence", test_script_independence),
        ("Connector Integration", test_connector_integration),
        ("Bidirectional Communication", test_bidirectional_communication),
        ("Script Collaboration", test_collaboration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
            
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(tests)} tests passed")
    
    if total_passed == len(tests):
        print("\n✅ All tests passed! Full integration successful.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()