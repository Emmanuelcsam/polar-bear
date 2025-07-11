#!/usr/bin/env python3
"""
Hivemind Connector for: modules/monitoring-systems
Connector ID: 1ed3a2a2
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
from connector_interface import ConnectorInterface, ScriptRegistry, ScriptState
from typing import Dict, Any, Optional, List, Tuple

class HivemindConnector:
    def __init__(self):
        self.connector_id = "1ed3a2a2"
        self.directory = Path("/home/jarvis/Documents/GitHub/polar-bear/modules/monitoring-systems")
        self.port = 10130
        self.parent_port = 10004
        self.depth = 2
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {}
        self.scripts_in_directory = {}
        self.script_registry = ScriptRegistry()
        self.script_connections: Dict[str, socket.socket] = {}
        self.script_states: Dict[str, ScriptState] = {}
        
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
        elif cmd == 'control_script':
            return self.control_script(message)
        elif cmd == 'get_script_state':
            script = message.get('script')
            return self.get_script_state(script)
        elif cmd == 'set_parameter':
            return self.set_script_parameter(message)
        elif cmd == 'get_all_states':
            return self.get_all_script_states()
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
            
    def control_script(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Control a script with advanced features"""
        script_name = message.get('script')
        action = message.get('action')
        params = message.get('params', {})
        
        if script_name not in self.scripts_in_directory:
            return {'error': f'Script {script_name} not found'}
            
        if action == 'start':
            return self.start_script_with_interface(script_name, params)
        elif action == 'stop':
            return self.stop_script(script_name)
        elif action == 'pause':
            return self.pause_script(script_name)
        elif action == 'resume':
            return self.resume_script(script_name)
        else:
            return {'error': f'Unknown action: {action}'}
            
    def start_script_with_interface(self, script_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start a script with connector interface enabled"""
        try:
            # Create a modified version of the script that includes the interface
            script_path = Path(self.scripts_in_directory[script_name])
            
            # Check if script already has interface
            with open(script_path, 'r') as f:
                content = f.read()
                
            if 'connector_interface' not in content:
                # Inject interface code at the beginning
                interface_code = f'''# Auto-injected connector interface
from connector_interface import ConnectorInterface
_connector_interface = ConnectorInterface('{script_name}', {self.port})
_connector_interface.set_status('initializing')
\n'''
                
                # Create temporary script with interface
                temp_script = script_path.parent / f'.{script_name}.connected'
                with open(temp_script, 'w') as f:
                    f.write(interface_code + content)
                    
                script_to_run = str(temp_script)
            else:
                script_to_run = str(script_path)
                
            # Start the script
            result = subprocess.Popen(
                [sys.executable, script_to_run],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, 'CONNECTOR_PORT': str(self.port)}
            )
            
            # Register the script
            self.script_registry.register_script(script_name, None)
            
            return {
                'status': 'started',
                'script': script_name,
                'pid': result.pid,
                'message': f'Script {script_name} started with connector interface'
            }
            
        except Exception as e:
            return {'error': f'Failed to start script: {str(e)}'}
            
    def stop_script(self, script_name: str) -> Dict[str, Any]:
        """Stop a running script"""
        if script_name in self.script_connections:
            try:
                # Send stop command through connection
                conn = self.script_connections[script_name]
                command = {'command': 'stop'}
                conn.send(json.dumps(command).encode())
                
                # Remove connection
                del self.script_connections[script_name]
                
                return {'status': 'stopped', 'script': script_name}
            except Exception as e:
                return {'error': f'Failed to stop script: {str(e)}'}
        else:
            return {'error': f'Script {script_name} not connected'}
            
    def pause_script(self, script_name: str) -> Dict[str, Any]:
        """Pause a running script"""
        if script_name in self.script_connections:
            try:
                conn = self.script_connections[script_name]
                command = {'command': 'pause'}
                conn.send(json.dumps(command).encode())
                return {'status': 'paused', 'script': script_name}
            except Exception as e:
                return {'error': f'Failed to pause script: {str(e)}'}
        else:
            return {'error': f'Script {script_name} not connected'}
            
    def resume_script(self, script_name: str) -> Dict[str, Any]:
        """Resume a paused script"""
        if script_name in self.script_connections:
            try:
                conn = self.script_connections[script_name]
                command = {'command': 'resume'}
                conn.send(json.dumps(command).encode())
                return {'status': 'resumed', 'script': script_name}
            except Exception as e:
                return {'error': f'Failed to resume script: {str(e)}'}
        else:
            return {'error': f'Script {script_name} not connected'}
            
    def get_script_state(self, script_name: str) -> Dict[str, Any]:
        """Get the current state of a script"""
        if script_name in self.script_states:
            state = self.script_states[script_name]
            return {
                'status': 'success',
                'state': {
                    'script_name': state.script_name,
                    'status': state.status,
                    'parameters': {k: v.__dict__ for k, v in state.parameters.items()},
                    'metrics': state.metrics,
                    'last_update': state.last_update,
                    'error_message': state.error_message
                }
            }
        else:
            return {'error': f'No state available for script {script_name}'}
            
    def set_script_parameter(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Set a parameter in a connected script"""
        script_name = message.get('script')
        param_name = message.get('parameter')
        value = message.get('value')
        
        if script_name not in self.script_connections:
            return {'error': f'Script {script_name} not connected'}
            
        try:
            conn = self.script_connections[script_name]
            command = {
                'command': 'set_parameter',
                'parameter': param_name,
                'value': value
            }
            conn.send(json.dumps(command).encode())
            
            # Wait for response
            conn.settimeout(5.0)
            response = conn.recv(4096)
            if response:
                return json.loads(response.decode())
            else:
                return {'error': 'No response from script'}
                
        except Exception as e:
            return {'error': f'Failed to set parameter: {str(e)}'}
            
    def get_all_script_states(self) -> Dict[str, Any]:
        """Get states of all connected scripts"""
        states = {}
        for script_name, state in self.script_states.items():
            states[script_name] = {
                'status': state.status,
                'parameters_count': len(state.parameters),
                'metrics': state.metrics,
                'last_update': state.last_update
            }
        return {
            'status': 'success',
            'scripts': states,
            'total_scripts': len(self.scripts_in_directory),
            'connected_scripts': len(self.script_connections)
        }

if __name__ == "__main__":
    connector = HivemindConnector()
    try:
        connector.start()
    except KeyboardInterrupt:
        connector.running = False
        print("\nConnector stopped")
