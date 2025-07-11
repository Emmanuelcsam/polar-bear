#!/usr/bin/env python3
"""
Test script to verify connector integration
Tests both independent and collaborative capabilities
"""
import subprocess
import sys
import time
import json
import socket
import threading
import os

def test_independent_scripts():
    """Test that scripts can run independently"""
    print("\n=== Testing Independent Script Execution ===")
    
    # Test 1: Random Pixel Generator
    print("\n1. Testing Random Pixel Generator (5 values)...")
    try:
        result = subprocess.run(
            [sys.executable, "random_pixel_generator.py", "100", "200", "0.1", "5"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            print("✓ Random pixel generator works independently")
            print(f"  Output preview: {result.stdout[:100]}...")
        else:
            print("✗ Random pixel generator failed")
            print(f"  Error: {result.stderr}")
    except Exception as e:
        print(f"✗ Error testing random pixel generator: {e}")
        
    # Test 2: Intensity Reader (need a test image)
    print("\n2. Creating test image for intensity reader...")
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_data = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        img = Image.fromarray(test_data, mode='L')
        img.save('test_image.png')
        print("✓ Test image created")
        
        # Test intensity reader
        print("3. Testing Intensity Reader...")
        result = subprocess.run(
            [sys.executable, "intensity_reader.py", "test_image.png", "128"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            print("✓ Intensity reader works independently")
            print(f"  Found {result.stdout.count('\\n')} pixels above threshold")
        else:
            print("✗ Intensity reader failed")
            print(f"  Error: {result.stderr}")
            
        # Cleanup
        os.remove('test_image.png')
        
    except ImportError:
        print("✗ PIL not installed, skipping intensity reader test")
    except Exception as e:
        print(f"✗ Error testing intensity reader: {e}")

def test_connector_control():
    """Test connector control capabilities"""
    print("\n\n=== Testing Connector Control ===")
    
    # Start the connector in background
    print("\n1. Starting connector...")
    connector_process = subprocess.Popen(
        [sys.executable, "connector.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give it time to start
    time.sleep(2)
    
    try:
        # Test socket connection
        print("2. Testing socket connection...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', 10089))
        print("✓ Connected to connector")
        
        # Test 1: List scripts
        print("\n3. Testing list scripts command...")
        command = {'type': 'list_scripts'}
        sock.send(json.dumps(command).encode())
        response = json.loads(sock.recv(4096).decode())
        
        if response.get('status') == 'success':
            scripts = response.get('scripts', {})
            print(f"✓ Found {len(scripts)} loaded scripts:")
            for name, info in scripts.items():
                print(f"  - {name}: {len(info['functions'])} functions")
        else:
            print("✗ Failed to list scripts")
            
        # Test 2: Control script
        print("\n4. Testing script control...")
        command = {
            'type': 'control_script',
            'script': 'random_pixel_generator',
            'command': 'get_info',
            'params': {}
        }
        sock.send(json.dumps(command).encode())
        response = json.loads(sock.recv(4096).decode())
        
        if response.get('status') == 'success':
            print("✓ Successfully controlled script")
            info = response.get('result', {})
            print(f"  Functions: {info.get('functions', [])}")
        else:
            print("✗ Failed to control script")
            
        sock.close()
        
    except socket.timeout:
        print("✗ Connection timeout - connector may not be running")
    except ConnectionRefusedError:
        print("✗ Connection refused - connector not listening on port")
    except Exception as e:
        print(f"✗ Error testing connector: {e}")
    finally:
        # Stop the connector
        connector_process.terminate()
        connector_process.wait(timeout=5)
        print("\n✓ Connector stopped")

def test_hivemind_connector():
    """Test hivemind connector capabilities"""
    print("\n\n=== Testing Hivemind Connector ===")
    
    # Start the hivemind connector
    print("\n1. Starting hivemind connector...")
    hivemind_process = subprocess.Popen(
        [sys.executable, "hivemind_connector.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give it time to start
    time.sleep(2)
    
    try:
        # Test connection
        print("2. Testing hivemind connection...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', 10089))
        print("✓ Connected to hivemind connector")
        
        # Test status command
        print("\n3. Testing status command...")
        command = {'command': 'status'}
        sock.send(json.dumps(command).encode())
        response = json.loads(sock.recv(4096).decode())
        
        print(f"✓ Connector status:")
        print(f"  - ID: {response.get('connector_id')}")
        print(f"  - Scripts: {response.get('scripts')}")
        print(f"  - Depth: {response.get('depth')}")
        
        sock.close()
        
    except Exception as e:
        print(f"✗ Error testing hivemind: {e}")
    finally:
        # Stop the hivemind connector
        hivemind_process.terminate()
        hivemind_process.wait(timeout=5)
        print("\n✓ Hivemind connector stopped")

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Test 1: Independent execution
    test_independent_scripts()
    
    # Test 2: Connector control
    test_connector_control()
    
    # Test 3: Hivemind connector
    test_hivemind_connector()
    
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()