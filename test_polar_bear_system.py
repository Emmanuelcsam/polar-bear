#!/usr/bin/env python3
"""
Test script for Polar Bear System
Verifies basic functionality without full system launch
"""

import os
import sys
import json
import socket
import time
import threading
from pathlib import Path

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    try:
        import logging
        import queue
        import hashlib
        import subprocess
        import importlib.util
        print("✓ All standard imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directories...")
    dirs_found = []
    dirs_missing = []
    
    for directory in ['ruleset', 'modules', 'training', 'static', 'templates']:
        if os.path.exists(directory):
            dirs_found.append(directory)
        else:
            dirs_missing.append(directory)
    
    print(f"✓ Found directories: {', '.join(dirs_found)}")
    if dirs_missing:
        print(f"✗ Missing directories: {', '.join(dirs_missing)}")
    
    return len(dirs_missing) == 0

def test_port_availability():
    """Test if required ports are available"""
    print("\nTesting port availability...")
    ports = {
        9000: "Root Controller",
        9001: "Ruleset Connector", 
        9002: "Modules Connector",
        9003: "Training Connector",
        9004: "Static Connector",
        9005: "Templates Connector"
    }
    
    available = []
    in_use = []
    
    for port, name in ports.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            sock.close()
            available.append(f"{port} ({name})")
        except OSError:
            in_use.append(f"{port} ({name})")
    
    if available:
        print(f"✓ Available ports: {len(available)}")
    if in_use:
        print(f"✗ Ports in use: {', '.join(in_use)}")
    
    return len(in_use) == 0

def test_file_creation():
    """Test if we can create files in directories"""
    print("\nTesting file creation permissions...")
    test_file = "test_polar_bear_temp.txt"
    
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✓ File creation successful")
        return True
    except Exception as e:
        print(f"✗ File creation failed: {e}")
        return False

def test_json_operations():
    """Test JSON operations"""
    print("\nTesting JSON operations...")
    test_data = {
        "test": "data",
        "timestamp": time.time(),
        "nested": {"key": "value"}
    }
    
    try:
        # Test serialization
        json_str = json.dumps(test_data)
        # Test deserialization
        parsed = json.loads(json_str)
        assert parsed["test"] == "data"
        print("✓ JSON operations successful")
        return True
    except Exception as e:
        print(f"✗ JSON operations failed: {e}")
        return False

def test_socket_communication():
    """Test basic socket communication"""
    print("\nTesting socket communication...")
    
    success = False
    server_socket = None
    
    def server_thread():
        nonlocal success, server_socket
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('localhost', 9999))
            server_socket.listen(1)
            server_socket.settimeout(2.0)
            
            conn, addr = server_socket.accept()
            data = conn.recv(1024).decode('utf-8')
            if data == "test":
                conn.send(b"success")
                success = True
            conn.close()
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            if server_socket:
                server_socket.close()
    
    # Start server
    server = threading.Thread(target=server_thread)
    server.daemon = True
    server.start()
    
    # Give server time to start
    time.sleep(0.5)
    
    # Client test
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(2.0)
        client.connect(('localhost', 9999))
        client.send(b"test")
        response = client.recv(1024).decode('utf-8')
        client.close()
        
        if response == "success":
            print("✓ Socket communication successful")
            return True
    except Exception as e:
        print(f"✗ Socket communication failed: {e}")
    
    return False

def test_config_file():
    """Test configuration file operations"""
    print("\nTesting configuration file...")
    
    config_exists = os.path.exists('polar_bear_config.json')
    if config_exists:
        try:
            with open('polar_bear_config.json', 'r') as f:
                config = json.load(f)
            print("✓ Configuration file exists and is valid")
            print(f"  Project: {config.get('project_name', 'Not set')}")
            print(f"  Log level: {config.get('log_level', 'Not set')}")
            return True
        except Exception as e:
            print(f"✗ Configuration file error: {e}")
            return False
    else:
        print("→ No configuration file found (will be created on first run)")
        return True

def test_python_version():
    """Test Python version"""
    print("\nTesting Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 7):
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.7+ required")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("POLAR BEAR SYSTEM - DIAGNOSTIC TEST")
    print("="*60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Import Test", test_imports),
        ("Directory Test", test_directories),
        ("Port Availability", test_port_availability),
        ("File Creation", test_file_creation),
        ("JSON Operations", test_json_operations),
        ("Socket Communication", test_socket_communication),
        ("Configuration File", test_config_file)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name:<25} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready to run.")
        print("\nTo start the system, run:")
        print("  python launch_polar_bear.py")
    else:
        print("\n✗ Some tests failed. Please fix the issues before running.")
        print("\nCommon fixes:")
        print("- Ensure all directories exist")
        print("- Check if ports 9000-9005 are free")
        print("- Verify Python 3.7+ is installed")
    
    return passed == total

def main():
    """Main test function"""
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()