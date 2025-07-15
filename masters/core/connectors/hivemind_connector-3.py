#!/usr/bin/env python3
"""
Hivemind Connector for: training/random-gen-cnn
Connector ID: 6786b0ec
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
        self.connector_id = "6786b0ec"
        self.directory = Path("/home/jarvis/Documents/GitHub/polar-bear/training/random-gen-cnn")
        self.port = 10314
        self.parent_port = 10184
        self.depth = 2
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {}
        self.scripts_in_directory = {}
        self.script_registry = {}  # Registry for scripts with their parameters
        self.script_status = {}    # Current status of scripts
        self.script_ports = {}     # Ports where scripts are listening
        
        # Setup logging
        self.logger = logging.getLogger(f"Connector_{self.connector_id}")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def start(self):
        """Start the connector"""
        self.logger.info(f"Starting connector {self.connector_id} on port {self.port}")
        
        # Start listening for connections
        self.listen_thread = threading.Thread(target=self.listen_for_connections)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        # Connect to parent
        self.connect_to_parent()
        
        # Scan directory
        self.scan_directory()
        
        # Heartbeat loop
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
            except:
                pass
                
    def handle_connection(self, conn):
        """Handle incoming connection"""
        try:
            data = conn.recv(4096).decode()
            message = json.loads(data)
            
            response = self.process_message(message)
            conn.send(json.dumps(response).encode())
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
        finally:
            conn.close()
            
    def process_message(self, message):
        """Process incoming message"""
        cmd = message.get('command')
        
        if cmd == 'status':
            return {
                'status': 'active',
                'connector_id': self.connector_id,
                'directory': str(self.directory),
                'depth': self.depth,
                'scripts': len(self.scripts_in_directory),
                'children': len(self.child_connectors)
            }
        elif cmd == 'scan':
            self.scan_directory()
            return {'status': 'scan_complete', 'scripts': list(self.scripts_in_directory.keys())}
        elif cmd == 'execute':
            script = message.get('script')
            if script in self.scripts_in_directory:
                return self.execute_script(script)
            return {'error': 'Script not found'}
        elif cmd == 'troubleshoot':
            return self.troubleshoot_connections()
        elif cmd == 'register_script':
            return self.register_script(message)
        elif cmd == 'update_status':
            return self.update_script_status(message)
        elif cmd == 'get_parameters':
            return self.get_script_parameters(message)
        elif cmd == 'script_listening':
            return self.register_script_port(message)
        elif cmd == 'control_script':
            return self.control_script(message)
        else:
            return {'error': 'Unknown command'}
            
    def connect_to_parent(self):
        """Connect to parent connector"""
        try:
            self.parent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.parent_socket.connect(('localhost', self.parent_port))
            
            # Register with parent
            register_msg = {
                'command': 'register',
                'connector_id': self.connector_id,
                'port': self.port,
                'directory': str(self.directory)
            }
            self.parent_socket.send(json.dumps(register_msg).encode())
            self.logger.info(f"Connected to parent on port {self.parent_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to parent: {e}")
            
    def scan_directory(self):
        """Scan directory for Python scripts and subdirectories"""
        self.scripts_in_directory.clear()
        
        try:
            for item in self.directory.iterdir():
                if item.is_file() and item.suffix == '.py' and item.name != 'hivemind_connector.py':
                    self.scripts_in_directory[item.name] = str(item)
                elif item.is_dir() and not self.should_skip_directory(item):
                    # Check for child connector
                    child_connector = item / 'hivemind_connector.py'
                    if child_connector.exists():
                        self.child_connectors[item.name] = str(child_connector)
                        
            self.logger.info(f"Found {len(self.scripts_in_directory)} scripts and {len(self.child_connectors)} child connectors")
        except Exception as e:
            self.logger.error(f"Error scanning directory: {e}")
            
    def should_skip_directory(self, path):
        """Check if directory should be skipped"""
        skip_dirs = {
            'venv', 'env', '.env', '__pycache__', '.git', 
            'node_modules', '.venv', 'virtualenv', '.tox',
            'build', 'dist', '.pytest_cache', '.mypy_cache'
        }
        return path.name.startswith('.') or path.name.lower() in skip_dirs
        
    def execute_script(self, script_name):
        """Execute a script in the directory"""
        if script_name not in self.scripts_in_directory:
            return {'error': 'Script not found'}
            
        try:
            result = subprocess.run(
                [sys.executable, self.scripts_in_directory[script_name]],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                'status': 'executed',
                'script': script_name,
                'returncode': result.returncode,
                'stdout': result.stdout[-1000:],  # Last 1000 chars
                'stderr': result.stderr[-1000:]
            }
        except Exception as e:
            return {'error': f'Execution failed: {str(e)}'}
            
    def troubleshoot_connections(self):
        """Troubleshoot connector connections"""
        issues = []
        
        # Check parent connection
        if not self.parent_socket:
            issues.append("Not connected to parent")
        
        # Check listening socket
        if not self.socket:
            issues.append("Not listening for connections")
            
        # Check child connectors
        for name, path in self.child_connectors.items():
            if not Path(path).exists():
                issues.append(f"Child connector missing: {name}")
                
        return {
            'connector_id': self.connector_id,
            'issues': issues,
            'healthy': len(issues) == 0
        }
        
    def register_script(self, message):
        """Register a script with its parameters"""
        script_name = message.get('script')
        parameters = message.get('parameters', {})
        
        self.script_registry[script_name] = {
            'parameters': parameters,
            'registered_at': time.time()
        }
        
        self.logger.info(f"Registered script: {script_name}")
        return {'status': 'registered', 'script': script_name}
        
    def update_script_status(self, message):
        """Update the status of a script"""
        script_name = message.get('script')
        status = message.get('status', {})
        
        self.script_status[script_name] = {
            'status': status,
            'updated_at': time.time()
        }
        
        return {'status': 'updated', 'script': script_name}
        
    def get_script_parameters(self, message):
        """Get parameters for a script"""
        script_name = message.get('script')
        
        # Check if we have override parameters for this script
        if script_name in self.script_registry:
            params = self.script_registry[script_name].get('parameters', {})
            # Here you could merge with global parameters or parent-provided parameters
            return {'parameters': params}
            
        return {'parameters': {}}
        
    def register_script_port(self, message):
        """Register the port where a script is listening"""
        script_name = message.get('script')
        port = message.get('port')
        
        self.script_ports[script_name] = port
        self.logger.info(f"Script {script_name} listening on port {port}")
        
        return {'status': 'registered', 'script': script_name, 'port': port}
        
    def control_script(self, message):
        """Send control command to a script"""
        script_name = message.get('script')
        command = message.get('command')
        
        if script_name not in self.script_ports:
            return {'error': 'Script not listening'}
            
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', self.script_ports[script_name]))
            sock.send(json.dumps(command).encode())
            
            response = sock.recv(4096).decode()
            result = json.loads(response)
            sock.close()
            
            return result
            
        except Exception as e:
            return {'error': f'Failed to control script: {str(e)}'}
    
    def heartbeat_loop(self):
        """Send heartbeat to parent"""
        while self.running:
            try:
                if self.parent_socket:
                    heartbeat = {
                        'command': 'heartbeat',
                        'connector_id': self.connector_id,
                        'timestamp': time.time()
                    }
                    self.parent_socket.send(json.dumps(heartbeat).encode())
            except:
                # Reconnect if connection lost
                self.connect_to_parent()
                
            time.sleep(30)  # Heartbeat every 30 seconds

if __name__ == "__main__":
    connector = HivemindConnector()
    try:
        connector.start()
    except KeyboardInterrupt:
        connector.running = False
        print("\nConnector stopped")
