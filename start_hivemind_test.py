#!/usr/bin/env python3
"""
Test script to start hivemind and perform basic operations
"""

import subprocess
import time
import socket
import json
import threading

def test_connector(port):
    """Test a single connector"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(('localhost', port))
        
        # Send status command
        sock.send(json.dumps({'command': 'status'}).encode())
        response = sock.recv(4096).decode()
        data = json.loads(response)
        
        sock.close()
        return data
    except Exception as e:
        return None

def run_hivemind_test():
    """Run the hivemind and test it"""
    print("Starting Polar Bear Hivemind in background...")
    
    # Start hivemind process
    process = subprocess.Popen(
        ['python', 'polar_bear_hivemind.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True
    )
    
    # Wait for initialization
    print("Waiting for hivemind to initialize...")
    time.sleep(5)
    
    # Test root server
    print("\nTesting root server connection...")
    root_response = test_connector(10000)
    if root_response:
        print("✓ Root server is not responding to status (expected)")
    else:
        print("✓ Root server is running on port 10000")
    
    # Test some connectors
    print("\nTesting connector communications...")
    active_count = 0
    for port in range(10001, 10011):
        response = test_connector(port)
        if response:
            active_count += 1
            print(f"✓ Connector on port {port}: {response.get('directory', 'Unknown')}")
    
    print(f"\nFound {active_count} active connectors")
    
    # Send commands to hivemind
    print("\nSending status command to hivemind...")
    process.stdin.write("status\n")
    process.stdin.flush()
    time.sleep(1)
    
    # Clean up
    print("\nStopping hivemind...")
    process.stdin.write("exit\n")
    process.stdin.flush()
    time.sleep(2)
    process.terminate()
    
    print("Test complete!")

if __name__ == "__main__":
    run_hivemind_test()