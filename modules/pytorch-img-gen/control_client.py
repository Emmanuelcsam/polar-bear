

import socket
import json
import sys
import time

class HivemindClient:
    def __init__(self, connector_port=10047):
        self.connector_port = connector_port

    def send_command(self, command):
        """Sends a command to the main hivemind connector."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('localhost', self.connector_port))
                s.sendall(json.dumps(command).encode('utf-8'))
                response = s.recv(4096).decode('utf-8')
                return json.loads(response)
        except Exception as e:
            return {'error': f"Failed to connect to hivemind connector: {e}"}

def print_status(status):
    """Prints the status in a readable format."""
    print("\n--- Hivemind Status ---")
    print(f"Connector ID: {status.get('connector_id')}")
    print(f"Directory: {status.get('directory')}")
    print("\nManaged Scripts:")
    scripts = status.get('scripts', {})
    if not scripts:
        print("  No scripts found.")
    for name, info in scripts.items():
        print(f"  - {name}:")
        print(f"    Status: {info.get('status', 'unknown')}")
        if info.get('port'):
            print(f"    Sidecar Port: {info.get('port')}")
    print("-----------------------\n")


def main():
    """A simple CLI to interact with the hivemind connector."""
    client = HivemindClient()

    if len(sys.argv) > 1:
        command_name = sys.argv[1]
        
        if command_name == 'status':
            response = client.send_command({'command': 'status'})
            if 'error' in response:
                print(f"Error: {response['error']}")
            else:
                print_status(response)

        elif command_name == 'start':
            if len(sys.argv) < 3:
                print("Usage: python control_client.py start <script_name>")
                return
            script_name = sys.argv[2]
            response = client.send_command({'command': 'start_script', 'script': script_name})
            print(response)

        elif command_name == 'stop':
            if len(sys.argv) < 3:
                print("Usage: python control_client.py stop <script_name>")
                return
            script_name = sys.argv[2]
            response = client.send_command({'command': 'stop_script', 'script': script_name})
            print(response)

        elif command_name == 'get_param':
            if len(sys.argv) < 4:
                print("Usage: python control_client.py get_param <script_name> <param_key>")
                return
            script_name, key = sys.argv[2], sys.argv[3]
            response = client.send_command({'command': 'get_param', 'script': script_name, 'key': key})
            print(response)

        elif command_name == 'set_param':
            if len(sys.argv) < 5:
                print("Usage: python control_client.py set_param <script_name> <param_key> <param_value>")
                return
            script_name, key, value = sys.argv[2], sys.argv[3], sys.argv[4]
            # Try to convert value to float/int
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                pass # Keep as string
            response = client.send_command({'command': 'set_param', 'script': script_name, 'key': key, 'value': value})
            print(response)
        
        elif command_name == 'test_run':
            print("--- Starting Test Run ---")
            print("Getting initial status...")
            print_status(client.send_command({'command': 'status'}))

            print("\n(1) Running feature-extractor.py to generate features.pt...")
            print(client.send_command({'command': 'start_script', 'script': 'feature-extractor.py'}))
            print("Waiting for feature extractor to finish...")
            time.sleep(15) # Give it ample time to download model and run

            print("\n(2) Starting background scripts: noise & neural generators...")
            print(client.send_command({'command': 'start_script', 'script': 'noise-generator.py'}))
            time.sleep(1)
            print(client.send_command({'command': 'start_script', 'script': 'neural-generator.py'}))
            time.sleep(1)

            print("\n(3) Getting status...")
            print_status(client.send_command({'command': 'status'}))

            print("\n(4) Starting style-optimizer.py (should work now)...")
            print(client.send_command({'command': 'start_script', 'script': 'style-optimizer.py'}))
            time.sleep(2)

            print("\n(5) Verifying style-optimizer is running...")
            print_status(client.send_command({'command': 'status'}))

            print("\n(6) Getting learning_rate from style-optimizer.py...")
            print(client.send_command({'command': 'get_param', 'script': 'style-optimizer.py', 'key': 'learning_rate'}))

            print("\n(7) Setting learning_rate on style-optimizer.py to 0.005...")
            print(client.send_command({'command': 'set_param', 'script': 'style-optimizer.py', 'key': 'learning_rate', 'value': 0.005}))

            print("\n(8) Getting learning_rate from style-optimizer.py again...")
            print(client.send_command({'command': 'get_param', 'script': 'style-optimizer.py', 'key': 'learning_rate'}))

            print("\n(9) Stopping all scripts...")
            # Stop in reverse order of dependency
            print(client.send_command({'command': 'stop_script', 'script': 'style-optimizer.py'}))
            print(client.send_command({'command': 'stop_script', 'script': 'neural-generator.py'}))
            print(client.send_command({'command': 'stop_script', 'script': 'noise-generator.py'}))
            # feature-extractor should have stopped on its own

            print("\n(10) Final status...")
            print_status(client.send_command({'command': 'status'}))
            print("--- Test Run Complete ---")

        else:
            print(f"Unknown command: {command_name}")

    else:
        print("Usage: python control_client.py <command> [args...]")
        print("Commands: status, start, stop, get_param, set_param, test_run")

if __name__ == "__main__":
    main()

