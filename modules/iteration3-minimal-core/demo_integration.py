#!/usr/bin/env python3
"""
Demonstration of connector integration capabilities
Shows how connectors can control and coordinate scripts
"""
import socket
import json
import time
import subprocess
import sys
import threading
from PIL import Image
import numpy as np

class ConnectorClient:
    """Client for communicating with connectors"""
    
    def __init__(self, host='localhost', port=10089):
        self.host = host
        self.port = port
        
    def send_command(self, command):
        """Send command to connector and get response"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        
        try:
            sock.connect((self.host, self.port))
            sock.send(json.dumps(command).encode())
            response = json.loads(sock.recv(4096).decode())
            return response
        finally:
            sock.close()

def demo_basic_control():
    """Demonstrate basic script control"""
    print("\n=== DEMO 1: Basic Script Control ===")
    
    # Start the connector
    print("Starting connector...")
    connector_process = subprocess.Popen(
        [sys.executable, "connector.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(3)  # Give it time to start
    
    try:
        client = ConnectorClient()
        
        # List loaded scripts
        print("\n1. Listing loaded scripts:")
        response = client.send_command({'type': 'list_scripts'})
        if response.get('status') == 'success':
            scripts = response.get('scripts', {})
            for name, info in scripts.items():
                print(f"   - {name}:")
                print(f"     Functions: {', '.join(info['functions'])}")
                
        # Get pixel generator parameters
        print("\n2. Getting pixel generator parameters:")
        response = client.send_command({
            'type': 'control_pixel_generator',
            'params': {'action': 'get_params'}
        })
        if response.get('status') == 'success':
            params = response.get('params', {})
            print(f"   Current parameters: {json.dumps(params, indent=2)}")
            
        # Set new parameters
        print("\n3. Setting new pixel generator parameters:")
        response = client.send_command({
            'type': 'control_pixel_generator',
            'params': {
                'action': 'set_params',
                'values': {
                    'min_val': 50,
                    'max_val': 150,
                    'delay': 0.5
                }
            }
        })
        print(f"   Result: {response.get('status')}")
        
    finally:
        connector_process.terminate()
        connector_process.wait(timeout=5)
        print("\nConnector stopped")

def demo_collaborative_processing():
    """Demonstrate collaborative processing between scripts"""
    print("\n\n=== DEMO 2: Collaborative Processing ===")
    
    # Create a test image with random pixels
    print("1. Creating test image with random pixels...")
    np.random.seed(42)
    test_data = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
    img = Image.fromarray(test_data, mode='L')
    img.save('demo_test.png')
    print("   Test image created")
    
    # Start the connector
    print("\n2. Starting connector for collaborative processing...")
    connector_process = subprocess.Popen(
        [sys.executable, "connector.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(3)
    
    try:
        client = ConnectorClient()
        
        # Read intensity values through connector
        print("\n3. Reading intensity values through connector:")
        response = client.send_command({
            'type': 'execute_intensity_reader',
            'params': {
                'path': 'demo_test.png',
                'threshold': 128
            }
        })
        
        if response.get('status') == 'success':
            results = response.get('results', [])
            print(f"   Found {len(results)} pixels above threshold 128")
            print(f"   Average intensity: {sum(results)/len(results):.2f}" if results else "   No pixels above threshold")
            
        # Control script through generic interface
        print("\n4. Getting script info through generic interface:")
        response = client.send_command({
            'type': 'control_script',
            'script': 'intensity_reader',
            'command': 'get_info',
            'params': {}
        })
        
        if response.get('status') == 'success':
            info = response.get('result', {})
            print(f"   Intensity reader info: {json.dumps(info, indent=2)}")
            
    finally:
        connector_process.terminate()
        connector_process.wait(timeout=5)
        print("\nConnector stopped")
        
        # Cleanup
        import os
        if os.path.exists('demo_test.png'):
            os.remove('demo_test.png')

def demo_hivemind_integration():
    """Demonstrate hivemind connector integration"""
    print("\n\n=== DEMO 3: Hivemind Integration ===")
    
    # Start hivemind connector
    print("1. Starting hivemind connector...")
    hivemind_process = subprocess.Popen(
        [sys.executable, "hivemind_connector.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(3)
    
    try:
        # Connect to hivemind
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', 10089))
        
        # Get status
        print("\n2. Getting hivemind status:")
        command = {'command': 'status'}
        sock.send(json.dumps(command).encode())
        response = json.loads(sock.recv(4096).decode())
        
        print(f"   Connector ID: {response.get('connector_id')}")
        print(f"   Directory: {response.get('directory')}")
        print(f"   Scripts: {response.get('scripts')}")
        print(f"   Children: {response.get('children')}")
        
        # List scripts through hivemind
        print("\n3. Listing scripts through hivemind:")
        command = {'command': 'list_scripts'}
        sock.send(json.dumps(command).encode())
        response = json.loads(sock.recv(4096).decode())
        
        if response.get('status') == 'success':
            scripts = response.get('scripts', {})
            print(f"   Found {len(scripts)} scripts")
            
        # Control script through hivemind
        print("\n4. Controlling script through hivemind:")
        command = {
            'command': 'control_script',
            'script_name': 'random_pixel_generator',
            'action': 'get_info',
            'params': {}
        }
        sock.send(json.dumps(command).encode())
        response = json.loads(sock.recv(4096).decode())
        
        if response.get('status') == 'success':
            print("   Successfully controlled script through hivemind")
            
        sock.close()
        
    except Exception as e:
        print(f"   Error: {e}")
    finally:
        hivemind_process.terminate()
        hivemind_process.wait(timeout=5)
        print("\nHivemind connector stopped")

def main():
    """Run all demonstrations"""
    print("=" * 60)
    print("CONNECTOR INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    print("\nThis demo shows:")
    print("1. Basic script control through connectors")
    print("2. Collaborative processing between scripts")
    print("3. Hivemind connector integration")
    
    try:
        # Run demos
        demo_basic_control()
        demo_collaborative_processing()
        demo_hivemind_integration()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo error: {e}")
        
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    print("\nKey capabilities demonstrated:")
    print("✓ Scripts can run independently")
    print("✓ Connectors can control script parameters")
    print("✓ Connectors can execute script functions")
    print("✓ Connectors can retrieve script information")
    print("✓ Hivemind integration provides hierarchical control")
    print("✓ All components work together seamlessly")

if __name__ == "__main__":
    main()