#!/usr/bin/env python3
"""
Hivemind Connector for: modules/visualization
Connector ID: f3c2af27
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

from common_data_and_utils import log_message, load_json_data, ImageResult, InspectorConfig, DefectInfo, DetectedZoneInfo, ZoneDefinition, load_single_image

class HivemindConnector:
    def __init__(self):
        self.connector_id = "f3c2af27"
        self.directory = Path("/home/jarvis/Documents/GitHub/polar-bear/modules/visualization")
        self.port = 10119
        self.parent_port = 10004
        self.depth = 2
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {}
        self.scripts_in_directory = {}
        
        # Use common logging utility
        log_message(f"Initializing HivemindConnector {self.connector_id}", level="INFO")
        
    def start(self):
        """Start the connector"""
        log_message(f"Starting connector {self.connector_id} on port {self.port}", level="INFO")
        
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
            log_message(f"Error handling connection: {e}", level="ERROR")
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
            log_message(f"Connected to parent on port {self.parent_port}", level="INFO")
        except Exception as e:
            log_message(f"Failed to connect to parent: {e}", level="ERROR")
            
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
                        
            log_message(f"Found {len(self.scripts_in_directory)} scripts and {len(self.child_connectors)} child connectors", level="INFO")
        except Exception as e:
            log_message(f"Error scanning directory: {e}", level="ERROR")
            
    def should_skip_directory(self, path):
        """Check if directory should be skipped"""
        skip_dirs = {
            'venv', 'env', '.env', '__pycache__', '.git', 
            'node_modules', '.venv', 'virtualenv', '.tox',
            'build', 'dist', '.pytest_cache', '.mypy_cache'
        }
        return path.name.startswith('.') or path.name.lower() in skip_dirs
        
    def execute_script(self, script_name, **kwargs):
        """Execute a script in the directory by importing its main function."""
        if script_name not in self.scripts_in_directory:
            return {'error': 'Script not found'}
            
        try:
            module_name = script_name.replace(".py", "")
            # Dynamically import the module
            module = __import__(module_name)
            
            # Assuming each script has a callable function named after its purpose
            # For example, result-visualizer.py might have a visualize_results function
            if module_name == "result-visualizer":
                if 'image_file' not in kwargs or 'image_result_file' not in kwargs:
                    return {'error': 'Missing image_file or image_result_file for result-visualizer'}
                
                image_path = Path(kwargs['image_file'])
                image_result_path = Path(kwargs['image_result_file'])

                raw_image_result_data = load_json_data(image_result_path)
                if raw_image_result_data is None:
                    return {'error': f'Could not load JSON data from {image_result_path}'}
                image_result = ImageResult.from_dict(raw_image_result_data)
                if image_result is None:
                    return {'error': f'Could not parse ImageResult from {image_result_path}'}

                save_path = Path(kwargs.get('save_path', f"hivemind_output_{image_path.stem}.png"))

                log_message(f"Executing {script_name} with image: {image_path} and result: {image_result_path}", level="INFO")
                module.visualize_results(image_path, image_result, save_path)
                return {'status': 'executed', 'script': script_name, 'output_saved_to': str(save_path)}
            
            # Add similar blocks for other scripts as they are modularized
            # elif module_name == "annotated-image-generator":
            #     ...
            
            else:
                log_message(f"No specific execution logic for {script_name}. Attempting generic execution.", level="WARNING")
                # Fallback to subprocess for unhandled scripts, or raise error
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
            log_message(f"Execution of {script_name} failed: {e}", level="ERROR")
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
                
        log_message(f"Troubleshooting complete for {self.connector_id}. Issues: {issues}", level="INFO")
        return {
            'connector_id': self.connector_id,
            'issues': issues,
            'healthy': len(issues) == 0
        }
        
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
            except Exception as e:
                log_message(f"Error during heartbeat: {e}. Attempting to reconnect.", level="ERROR")
                # Reconnect if connection lost
                self.connect_to_parent()
                
            time.sleep(30)  # Heartbeat every 30 seconds

if __name__ == "__main__":
    connector = HivemindConnector()
    try:
        connector.start()
    except KeyboardInterrupt:
        connector.running = False
        log_message("Connector stopped by user.", level="INFO")
