#!/usr/bin/env python3
"""
Hivemind Connector for: modules/pytorch-img-gen
Connector ID: 8d1c2291
Depth Level: 2
"""

import socket
import json
import threading
import time
import os
import sys
import subprocess
from pathlib import Path
import logging

class HivemindConnector:
    def __init__(self):
        self.connector_id = "8d1c2291"
        self.directory = Path("/home/jarvis/Documents/GitHub/polar-bear/modules/pytorch-img-gen")
        self.port = 10047
        self.parent_port = 10004
        self.depth = 2
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {}
        self.managed_scripts = {}  # To store info about running scripts
        
        # Setup logging
        self.logger = logging.getLogger(f"Connector_{self.connector_id}")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def start(self):
        """Start the connector"""
        self.logger.info(f"Starting connector {self.connector_id} on port {self.port}")
        
        self.listen_thread = threading.Thread(target=self.listen_for_connections)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        self.connect_to_parent()
        self.scan_directory()
        self.heartbeat_loop()
        
    def listen_for_connections(self):
        """Listen for incoming connections"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('localhost', self.port))
        self.socket.listen(5)
        
        while self.running:
            try:
                conn, addr = self.socket.accept()
                threading.Thread(target=self.handle_connection, args=(conn,)).start()
            except Exception as e:
                if self.running:
                    self.logger.error(f"Error in listen loop: {e}")
                
    def handle_connection(self, conn):
        """Handle incoming connection"""
        try:
            with conn:
                data = conn.recv(4096).decode()
                if not data: return
                message = json.loads(data)
                response = self.process_message(message)
                conn.send(json.dumps(response).encode())
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
            
    def process_message(self, message):
        """Process incoming message"""
        cmd = message.get('command')
        script = message.get('script')

        if cmd == 'status':
            return self.get_status()
        elif cmd == 'scan':
            self.scan_directory()
            return {'status': 'scan_complete', 'scripts': list(self.managed_scripts.keys())}
        elif cmd == 'start_script':
            return self.start_script(script)
        elif cmd == 'stop_script':
            return self.stop_script(script)
        elif cmd == 'get_param':
            return self.send_command_to_script(script, message)
        elif cmd == 'set_param':
            return self.send_command_to_script(script, message)
        elif cmd == 'troubleshoot':
            return self.troubleshoot_connections()
        else:
            return {'error': 'Unknown command'}

    def get_status(self):
        """Get the status of the connector and its managed scripts."""
        script_statuses = {}
        for name, info in self.managed_scripts.items():
            # Check if the process is still running
            if info.get('process') and info['process'].poll() is None:
                status = 'running'
            else:
                status = 'stopped'
            script_statuses[name] = {
                'status': status,
                'port': info.get('port')
            }
        return {
            'status': 'active',
            'connector_id': self.connector_id,
            'directory': str(self.directory),
            'depth': self.depth,
            'scripts': script_statuses,
            'children': len(self.child_connectors)
        }

    def connect_to_parent(self):
        """Connect to parent connector"""
        try:
            self.parent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.parent_socket.connect(('localhost', self.parent_port))
            register_msg = {'command': 'register', 'connector_id': self.connector_id, 'port': self.port, 'directory': str(self.directory)}
            self.parent_socket.send(json.dumps(register_msg).encode())
            self.logger.info(f"Connected to parent on port {self.parent_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to parent: {e}")
            
    def scan_directory(self):
        """Scan directory for Python scripts and subdirectories"""
        self.managed_scripts.clear()
        try:
            for item in self.directory.iterdir():
                if item.is_file() and item.suffix == '.py' and 'connector.py' not in item.name and 'setup.py' not in item.name and 'test_runner.py' not in item.name and 'control_client.py' not in item.name:
                    self.managed_scripts[item.name] = {'path': str(item), 'process': None, 'port': None}
                elif item.is_dir() and not self.should_skip_directory(item):
                    child_connector = item / 'hivemind_connector.py'
                    if child_connector.exists():
                        self.child_connectors[item.name] = str(child_connector)
            self.logger.info(f"Found {len(self.managed_scripts)} scripts and {len(self.child_connectors)} child connectors")
        except Exception as e:
            self.logger.error(f"Error scanning directory: {e}")

    def start_script(self, script_name):
        """Start a managed script."""
        if script_name not in self.managed_scripts:
            return {'error': 'Script not found'}
        
        info = self.managed_scripts[script_name]
        if info.get('process') and info['process'].poll() is None:
            return {'status': 'already_running', 'script': script_name}

        # Use the python from the user-specified virtual environment
        python_executable = self.directory / "venv" / "bin" / "python"
        if not python_executable.exists():
            self.logger.error(f"Python executable not found at: {python_executable}")
            return {'error': f'venv python not found at {python_executable}'}

        try:
            self.logger.info(f"Starting script: {script_name} with interpreter {python_executable}")
            process = subprocess.Popen([str(python_executable), info['path']],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True)
            info['process'] = process
            
            # Wait for the port file to appear
            port_file = self.directory / f".{script_name}.port"
            max_wait = 30  # seconds
            start_time = time.time()
            while not port_file.exists():
                if time.time() - start_time > max_wait:
                    process.terminate()
                    stdout, stderr = process.communicate()
                    self.logger.error(f"Sidecar port file not found for {script_name} after {max_wait} seconds.")
                    self.logger.error(f"STDOUT: {stdout}")
                    self.logger.error(f"STDERR: {stderr}")
                    return {'error': f'Could not get sidecar port for script {script_name}', 'stdout': stdout, 'stderr': stderr}
                time.sleep(0.2)

            info['port'] = int(port_file.read_text())
            port_file.unlink() # Clean up port file
            self.logger.info(f"Started {script_name} (PID: {process.pid}) on sidecar port {info['port']}")
            return {'status': 'started', 'script': script_name, 'pid': process.pid, 'port': info['port']}
        except Exception as e:
            self.logger.error(f"Execution failed for {script_name}: {e}")
            return {'error': f'Execution failed: {str(e)}'}

    def stop_script(self, script_name):
        """Stop a managed script by sending a command to its sidecar."""
        response = self.send_command_to_script(script_name, {'command': 'stop'})
        if 'error' not in response:
             # Give the script a moment to shut down
            time.sleep(1)
            info = self.managed_scripts.get(script_name)
            if info and info.get('process'):
                info['process'].terminate() # Ensure it's stopped
                info['process'] = None
                info['port'] = None
        return response

    def send_command_to_script(self, script_name, command_message):
        """Send a command to a script's sidecar connector."""
        if script_name not in self.managed_scripts or not self.managed_scripts[script_name].get('port'):
            return {'error': 'Script not running or has no sidecar port'}

        port = self.managed_scripts[script_name]['port']
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('localhost', port))
                s.sendall(json.dumps(command_message).encode('utf-8'))
                response = s.recv(4096).decode('utf-8')
                return json.loads(response)
        except Exception as e:
            return {'error': f"Failed to communicate with script '{script_name}': {e}"}

    def should_skip_directory(self, path):
        """Check if directory should be skipped"""
        skip_dirs = {'venv', 'env', '.env', '__pycache__', '.git', 'node_modules', '.venv', 'virtualenv', '.tox', 'build', 'dist', '.pytest_cache', '.mypy_cache'}
        return path.name.startswith('.') or path.name.lower() in skip_dirs
        
    def troubleshoot_connections(self):
        """Troubleshoot connector connections"""
        # (Implementation remains the same)
        pass
        
    def heartbeat_loop(self):
        """Send heartbeat to parent"""
        while self.running:
            try:
                if self.parent_socket:
                    heartbeat = {'command': 'heartbeat', 'connector_id': self.connector_id, 'timestamp': time.time()}
                    self.parent_socket.send(json.dumps(heartbeat).encode())
            except:
                self.connect_to_parent()
            time.sleep(30)

    def stop(self):
        self.running = False
        for script_name in list(self.managed_scripts.keys()):
            self.stop_script(script_name)
        if self.socket:
            self.socket.close()
        if self.parent_socket:
            self.parent_socket.close()
        self.logger.info("Hivemind connector shut down.")

if __name__ == "__main__":
    connector = HivemindConnector()
    try:
        connector.start()
    except KeyboardInterrupt:
        connector.stop()
        print("\nConnector stopped")