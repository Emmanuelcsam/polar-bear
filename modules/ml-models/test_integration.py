#!/usr/bin/env python3
"""
Integration Test Script for ML Models Connector System
Tests collaboration and parameter control between scripts
"""

import sys
import socket
import json
import time
import threading
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from script_interface import ScriptInterface, ConnectorClient

class CollaborationTestScript(ScriptInterface):
    """Test script that demonstrates collaboration capabilities"""
    
    def __init__(self, script_id):
        super().__init__(f"test_script_{script_id}", f"Collaboration Test Script {script_id}")
        self.script_id = script_id
        self.client = ConnectorClient(self)
        
        # Register parameters
        self.register_parameter("test_value", 0, range(0, 101))
        self.register_parameter("sync_enabled", True, [True, False])
        self.register_parameter("broadcast_interval", 5, range(1, 61))
        
        # Register variables
        self.register_variable("messages_sent", 0)
        self.register_variable("messages_received", 0)
        self.register_variable("last_collaboration_time", None)
        self.register_variable("collaborators", [])
        
        # Collaboration data
        self.received_data = {}
        
    def run(self):
        """Main execution"""
        self.logger.info(f"Test Script {self.script_id} starting...")
        
        # Register with connector
        self.client.register_script()
        
        # Start collaboration thread
        collab_thread = threading.Thread(target=self._collaboration_loop, daemon=True)
        collab_thread.start()
        
        # Main loop
        start_time = time.time()
        while self.running and (time.time() - start_time) < 30:  # Run for 30 seconds
            # Update test value
            current_value = self.get_parameter("test_value")
            new_value = (current_value + self.script_id) % 100
            self.set_parameter("test_value", new_value)
            
            # Broadcast our state if sync enabled
            if self.get_parameter("sync_enabled"):
                self._broadcast_state()
                
            time.sleep(self.get_parameter("broadcast_interval"))
            
        self.logger.info(f"Test Script {self.script_id} completed")
        self._print_summary()
        
    def _collaboration_loop(self):
        """Handle incoming collaboration requests"""
        while self.running:
            # Check for collaboration messages
            response = self.send_to_connector({
                "command": "get_collaborations",
                "script_name": self.script_name
            })
            
            if response and response.get("requests"):
                for request in response["requests"]:
                    self._handle_collaboration(request)
                    
            time.sleep(1)
            
    def _handle_collaboration(self, request):
        """Process collaboration request"""
        source = request.get("source")
        data = request.get("data")
        
        self.logger.info(f"Received collaboration from {source}")
        
        # Store received data
        self.received_data[source] = data
        
        # Update variables
        messages = self.get_variable("messages_received")
        self.set_variable("messages_received", messages + 1)
        self.set_variable("last_collaboration_time", time.time())
        
        # Update collaborators list
        collaborators = self.get_variable("collaborators")
        if source not in collaborators:
            collaborators.append(source)
            self.set_variable("collaborators", collaborators)
            
        # Process data
        if data.get("type") == "state_sync":
            # Sync our test value with average
            other_value = data.get("test_value", 0)
            my_value = self.get_parameter("test_value")
            avg_value = (my_value + other_value) // 2
            self.set_parameter("test_value", avg_value)
            
    def _broadcast_state(self):
        """Broadcast current state to other scripts"""
        state_data = {
            "type": "state_sync",
            "script_id": self.script_id,
            "test_value": self.get_parameter("test_value"),
            "timestamp": time.time()
        }
        
        self.client.broadcast_data(state_data)
        
        # Update sent counter
        messages = self.get_variable("messages_sent")
        self.set_variable("messages_sent", messages + 1)
        
    def _print_summary(self):
        """Print collaboration summary"""
        print(f"\n=== Script {self.script_id} Summary ===")
        print(f"Messages sent: {self.get_variable('messages_sent')}")
        print(f"Messages received: {self.get_variable('messages_received')}")
        print(f"Collaborators: {self.get_variable('collaborators')}")
        print(f"Final test value: {self.get_parameter('test_value')}")
        print(f"Received data from: {list(self.received_data.keys())}")


def test_parameter_control():
    """Test remote parameter control"""
    print("\n=== Testing Parameter Control ===")
    
    try:
        # Connect to script control port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 10118))
        
        # Request script list
        message = {"command": "list_scripts"}
        sock.send(json.dumps(message).encode())
        response = json.loads(sock.recv(4096).decode())
        
        print(f"Available scripts: {response.get('scripts', [])}")
        
        # If test script is registered, try to control it
        if any("test_script" in s for s in response.get('scripts', [])):
            # Set parameter
            control_msg = {
                "command": "set_parameter",
                "script": "test_script_1",
                "parameter": "test_value",
                "value": 42
            }
            sock.send(json.dumps(control_msg).encode())
            response = json.loads(sock.recv(4096).decode())
            print(f"Parameter set result: {response}")
            
        sock.close()
        return True
        
    except Exception as e:
        print(f"Parameter control test failed: {e}")
        return False


def test_collaboration():
    """Test collaboration between two scripts"""
    print("\n=== Testing Script Collaboration ===")
    
    # Create two test scripts
    script1 = CollaborationTestScript(1)
    script2 = CollaborationTestScript(2)
    
    # Run them in threads
    thread1 = threading.Thread(target=script1.run_with_connector)
    thread2 = threading.Thread(target=script2.run_with_connector)
    
    thread1.start()
    thread2.start()
    
    # Wait for completion
    thread1.join(timeout=35)
    thread2.join(timeout=35)
    
    # Check if they collaborated
    collab1 = len(script1.get_variable("collaborators")) > 0
    collab2 = len(script2.get_variable("collaborators")) > 0
    
    return collab1 and collab2


def test_connector_status():
    """Test connector status and script info"""
    print("\n=== Testing Connector Status ===")
    
    try:
        # Connect to hivemind connector
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 10117))
        
        # Get status
        message = {"command": "status"}
        sock.send(json.dumps(message).encode())
        response = json.loads(sock.recv(4096).decode())
        
        print(f"Connector Status: {response.get('status')}")
        print(f"Scripts available: {response.get('scripts')}")
        print(f"Scripts registered: {response.get('registered_scripts')}")
        print(f"Scripts running: {response.get('running_scripts')}")
        
        # Get detailed script info
        message = {"command": "get_scripts_info"}
        sock.send(json.dumps(message).encode())
        response = json.loads(sock.recv(4096).decode())
        
        print(f"\nDetailed Script Info:")
        for category, data in response.items():
            print(f"  {category}: {data}")
            
        sock.close()
        return True
        
    except Exception as e:
        print(f"Connector status test failed: {e}")
        return False


def main():
    """Run integration tests"""
    print("="*60)
    print("ML MODELS CONNECTOR - INTEGRATION TESTS")
    print("="*60)
    
    # Check if connectors are running
    print("\nChecking connector availability...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect(('localhost', 10117))
        sock.close()
        print("✓ Hivemind connector is running")
    except:
        print("✗ Hivemind connector not running")
        print("  Please run: python hivemind_connector.py")
        return
        
    # Run tests
    tests = [
        ("Connector Status", test_connector_status),
        ("Parameter Control", test_parameter_control),
        ("Script Collaboration", test_collaboration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test failed with error: {e}")
            results.append((test_name, False))
            
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<30} [{status}]")
        
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed ({(passed/total)*100:.0f}%)")
    
    if passed == total:
        print("\n✓ All integration tests passed!")
        print("The ML Models Connector System is fully functional.")
    else:
        print("\n✗ Some tests failed.")
        print("Check the connector logs for more details.")


if __name__ == "__main__":
    main()