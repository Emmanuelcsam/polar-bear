#!/usr/bin/env python3
"""
Demo client for interacting with the hivemind connector
Shows how to control scripts remotely via socket communication
"""

import socket
import json
import sys

def send_command(command):
    """Send a command to the hivemind connector and return response"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(('localhost', 10113))
        
        sock.send(json.dumps(command).encode())
        response = json.loads(sock.recv(8192).decode())
        
        sock.close()
        return response
        
    except socket.error as e:
        return {'error': f'Socket error: {e}'}
    except Exception as e:
        return {'error': f'Error: {e}'}

def main():
    """Demonstrate hivemind connector capabilities"""
    
    print("=== Hivemind Connector Client Demo ===\n")
    print("Connecting to hivemind connector on port 10113...")
    
    # 1. Check status
    print("\n1. Checking Status:")
    response = send_command({'command': 'status'})
    if 'error' in response:
        print(f"   Error: {response['error']}")
        print("   Make sure hivemind_connector.py is running!")
        return
    
    print(f"   Status: {response.get('status')}")
    print(f"   Connector ID: {response.get('connector_id')}")
    print(f"   Scripts available: {response.get('scripts')}")
    
    # 2. Get script info
    print("\n2. Getting Script Information:")
    response = send_command({
        'command': 'get_script_info',
        'script': 'correlation-finder'
    })
    
    if 'script_info' in response:
        info = response['script_info']
        print(f"   Script: {info.get('name')}")
        print(f"   Functions: {[f['name'] for f in info.get('functions', [])]}")
    
    # 3. Execute correlation analysis
    print("\n3. Running Correlation Analysis:")
    response = send_command({
        'command': 'control_script',
        'script': 'correlation-finder',
        'action': 'execute',
        'params': {
            'input_data': '1 10%\n2 20%\n3 30%\n4 40%\n5 50%'
        }
    })
    
    if response.get('success'):
        print("   Analysis completed successfully")
        if 'stdout' in response:
            print(f"   Output: {response['stdout'].strip()}")
    
    # 4. Set and get variables
    print("\n4. Variable Management:")
    
    # Set a variable
    response = send_command({
        'command': 'set_variable',
        'script': 'correlation-finder',
        'variable': 'analysis_label',
        'value': 'Demo Analysis via Hivemind'
    })
    print(f"   Set variable: {response.get('success')}")
    
    # Get variables
    response = send_command({
        'command': 'get_variables',
        'script': 'correlation-finder'
    })
    if 'variables' in response:
        print(f"   Current variables: {list(response['variables'].keys())}")
    
    # 5. Test shared data
    print("\n5. Shared Data Management:")
    
    # Set shared data
    response = send_command({
        'command': 'set_shared_data',
        'key': 'demo_value',
        'value': {'timestamp': '2025-07-11', 'data': [1, 2, 3, 4, 5]}
    })
    print(f"   Set shared data: {response.get('status')}")
    
    # Get shared data
    response = send_command({
        'command': 'get_shared_data'
    })
    if 'shared_data' in response:
        print(f"   Shared data keys: {list(response['shared_data'].keys())}")
    
    # 6. Run collaborative task
    print("\n6. Collaborative Task:")
    task_definition = {
        'name': 'demo_collaborative_task',
        'steps': [
            {
                'script': 'correlation-finder',
                'action': 'execute',
                'params': {'input_data': '1 5%\n2 15%\n3 25%\n4 35%\n5 45%'}
            }
        ]
    }
    
    response = send_command({
        'command': 'collaborative_task',
        'task_definition': task_definition
    })
    
    if 'task_result' in response:
        print("   Collaborative task completed")
        results = response['task_result']
        for key, value in results.items():
            if isinstance(value, dict) and value.get('success'):
                print(f"   - {key}: Success")
    
    # 7. Troubleshoot
    print("\n7. Troubleshooting:")
    response = send_command({'command': 'troubleshoot'})
    
    print(f"   Healthy: {response.get('healthy')}")
    print(f"   Scripts loaded: {response.get('scripts_loaded')}")
    print(f"   Collaboration enabled: {response.get('collaboration_enabled')}")
    
    if response.get('issues'):
        print(f"   Issues: {response['issues']}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()