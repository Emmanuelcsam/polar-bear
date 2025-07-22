#!/usr/bin/env python3
"""
Hivemind Connector for: modules/iteration2-basic-stats
Connector ID: f2e60346
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
import importlib.util
import ast
import io
from contextlib import redirect_stdout, redirect_stderr
import queue

# Import the enhanced connector system
sys.path.insert(0, str(Path(__file__).parent))
from connector import ScriptController, ConnectorSystem

class HivemindConnector:
    def __init__(self):
        self.connector_id = "f2e60346"
        self.directory = Path("/home/jarvis/Documents/GitHub/polar-bear/modules/iteration2-basic-stats")
        self.port = 10113
        self.parent_port = 10004
        self.depth = 2
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {}
        self.scripts_in_directory = {}
        
        # Initialize the enhanced connector system
        self.connector_system = ConnectorSystem()
        self.connector_system.scan_scripts()
        self.connector_system.enable_collaboration()
        
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
        """Process incoming message with enhanced script control"""
        cmd = message.get('command')
        
        if cmd == 'status':
            # Return enhanced status with script details
            status = self.connector_system.get_system_status()
            return {
                'status': 'active',
                'connector_id': self.connector_id,
                'directory': str(self.directory),
                'depth': self.depth,
                'scripts': len(self.scripts_in_directory),
                'children': len(self.child_connectors),
                'system_status': status
            }
            
        elif cmd == 'scan':
            self.scan_directory()
            return {'status': 'scan_complete', 'scripts': list(self.scripts_in_directory.keys())}
            
        elif cmd == 'execute':
            script = message.get('script')
            if script in self.scripts_in_directory:
                return self.execute_script(script, message.get('params', {}))
            return {'error': 'Script not found'}
            
        elif cmd == 'control_script':
            # New enhanced control commands
            script_name = message.get('script')
            action = message.get('action')
            params = message.get('params', {})
            return self.connector_system.control_script(script_name, action, params)
            
        elif cmd == 'get_script_info':
            script_name = message.get('script')
            info = self.connector_system.get_script_info(script_name)
            return {'script_info': info} if info else {'error': 'Script not found'}
            
        elif cmd == 'set_variable':
            script_name = message.get('script')
            variable = message.get('variable')
            value = message.get('value')
            result = self.connector_system.control_script(
                script_name, 'set_variable', 
                {'variable': variable, 'value': value}
            )
            return result
            
        elif cmd == 'get_variables':
            script_name = message.get('script')
            return self.connector_system.control_script(script_name, 'get_variables')
            
        elif cmd == 'execute_function':
            script_name = message.get('script')
            function_name = message.get('function')
            args = message.get('args', [])
            kwargs = message.get('kwargs', {})
            return self.connector_system.control_script(
                script_name, 'execute_function',
                {'function': function_name, 'args': args, 'kwargs': kwargs}
            )
            
        elif cmd == 'modify_script':
            script_name = message.get('script')
            modifications = message.get('modifications')
            return self.connector_system.control_script(
                script_name, 'modify',
                {'modifications': modifications}
            )
            
        elif cmd == 'collaborative_task':
            task_definition = message.get('task_definition')
            return {
                'task_result': self.connector_system.execute_collaborative_task(task_definition)
            }
            
        elif cmd == 'get_shared_data':
            return {
                'shared_data': self.connector_system.shared_memory.get('data', {})
            }
            
        elif cmd == 'set_shared_data':
            key = message.get('key')
            value = message.get('value')
            self.connector_system.shared_memory['data'][key] = value
            return {'status': 'shared_data_set', 'key': key}
            
        elif cmd == 'send_script_message':
            sender = message.get('sender', 'hivemind')
            target = message.get('target')
            msg = message.get('message')
            self.connector_system.send_message(sender, target, msg)
            return {'status': 'message_sent'}
            
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
                'directory': str(self.directory),
                'capabilities': [
                    'execute', 'control_script', 'set_variable', 'get_variables',
                    'execute_function', 'modify_script', 'collaborative_task',
                    'shared_data_management', 'script_messaging'
                ]
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
            
            # Also ensure the connector system is up to date
            self.connector_system.scan_scripts()
            
        except Exception as e:
            self.logger.error(f"Error scanning directory: {e}")
            
    def should_skip_directory(self, path):
        """Check if directory should be skipped"""
        skip_dirs = {
            'venv', 'env', '.env', '__pycache__', '.git', 
            'node_modules', '.venv', 'virtualenv', '.tox',
            'build', 'dist', '.pytest_cache', '.mypy_cache',
            'iteration2_basic_stats.egg-info'
        }
        return path.name.startswith('.') or path.name.lower() in skip_dirs
        
    def execute_script(self, script_name, params=None):
        """Execute a script using the enhanced connector system"""
        if script_name not in self.scripts_in_directory:
            return {'error': 'Script not found'}
            
        try:
            # Use the enhanced execution with parameter support
            if script_name.replace('.py', '') in self.connector_system.script_controllers:
                result = self.connector_system.control_script(
                    script_name.replace('.py', ''), 
                    'execute',
                    params
                )
                return {
                    'status': 'executed',
                    'script': script_name,
                    'result': result
                }
            else:
                # Fallback to subprocess execution
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
                
        # Check script controllers
        script_issues = []
        for name, controller in self.connector_system.script_controllers.items():
            if not controller.is_loaded:
                script_issues.append(f"Script {name} not loaded properly")
                
        return {
            'connector_id': self.connector_id,
            'issues': issues,
            'script_issues': script_issues,
            'healthy': len(issues) == 0 and len(script_issues) == 0,
            'scripts_loaded': len(self.connector_system.script_controllers),
            'collaboration_enabled': hasattr(self.connector_system, 'shared_memory')
        }
        
    def heartbeat_loop(self):
        """Send heartbeat to parent"""
        while self.running:
            try:
                if self.parent_socket:
                    heartbeat = {
                        'command': 'heartbeat',
                        'connector_id': self.connector_id,
                        'timestamp': time.time(),
                        'scripts_status': {
                            name: controller.is_loaded 
                            for name, controller in self.connector_system.script_controllers.items()
                        }
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