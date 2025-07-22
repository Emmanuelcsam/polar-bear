#!/usr/bin/env python3
"""
Enhanced Hivemind Connector for: modules/ml-models
Connector ID: 213c96bc
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
from typing import Dict, Optional, Any

# Import script interface
sys.path.insert(0, str(Path(__file__).parent))
try:
    from script_interface import ScriptInterface, ConnectorClient
except ImportError:
    ScriptInterface = None
    ConnectorClient = None

class EnhancedHivemindConnector:
    def __init__(self):
        self.connector_id = "213c96bc"
        self.directory = Path("/home/jarvis/Documents/GitHub/polar-bear/modules/ml-models")
        self.port = 10117
        self.parent_port = 10004
        self.script_control_port = 10118
        self.depth = 2
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {}
        self.scripts_in_directory = {}
        self.registered_scripts = {}
        self.script_processes = {}
        self.collaboration_requests = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"HivemindConnector_{self.connector_id}")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def start(self):
        """Start the enhanced hivemind connector"""
        self.logger.info(f"Starting enhanced connector {self.connector_id} on port {self.port}")
        
        # Start listening for connections
        self.listen_thread = threading.Thread(target=self.listen_for_connections)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        # Start script control server
        self.control_thread = threading.Thread(target=self.run_script_control_server)
        self.control_thread.daemon = True
        self.control_thread.start()
        
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
                
    def run_script_control_server(self):
        """Run control server for script communication"""
        control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        control_socket.bind(('localhost', self.script_control_port))
        control_socket.listen(10)
        
        self.logger.info(f"Script control server listening on port {self.script_control_port}")
        
        while self.running:
            try:
                conn, addr = control_socket.accept()
                threading.Thread(target=self.handle_script_control, args=(conn,)).start()
            except Exception as e:
                if self.running:
                    self.logger.error(f"Script control server error: {e}")
                    
    def handle_connection(self, conn):
        """Handle incoming connection from parent or other connectors"""
        try:
            data = conn.recv(4096).decode()
            message = json.loads(data)
            
            response = self.process_message(message)
            conn.send(json.dumps(response).encode())
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
        finally:
            conn.close()
            
    def handle_script_control(self, conn):
        """Handle control connection from scripts"""
        try:
            data = conn.recv(4096).decode()
            message = json.loads(data)
            response = self.process_script_message(message)
            conn.send(json.dumps(response).encode())
        except Exception as e:
            self.logger.error(f"Error handling script control: {e}")
        finally:
            conn.close()
            
    def process_message(self, message):
        """Process incoming message from parent/other connectors"""
        cmd = message.get('command')
        
        if cmd == 'status':
            return {
                'status': 'active',
                'connector_id': self.connector_id,
                'directory': str(self.directory),
                'depth': self.depth,
                'scripts': len(self.scripts_in_directory),
                'registered_scripts': len(self.registered_scripts),
                'running_scripts': len([p for p in self.script_processes.values() if p.poll() is None]),
                'children': len(self.child_connectors)
            }
        elif cmd == 'scan':
            self.scan_directory()
            return {'status': 'scan_complete', 'scripts': list(self.scripts_in_directory.keys())}
        elif cmd == 'execute':
            script = message.get('script')
            with_connector = message.get('with_connector', True)
            if script in self.scripts_in_directory:
                return self.execute_script(script, with_connector)
            return {'error': 'Script not found'}
        elif cmd == 'stop':
            script = message.get('script')
            return self.stop_script(script)
        elif cmd == 'control':
            # Forward control commands to specific scripts
            script = message.get('script')
            control_cmd = message.get('control_command')
            return self.control_script(script, control_cmd)
        elif cmd == 'get_scripts_info':
            return self.get_all_scripts_info()
        elif cmd == 'troubleshoot':
            return self.troubleshoot_connections()
        else:
            return {'error': 'Unknown command'}
            
    def process_script_message(self, message):
        """Process messages from scripts"""
        command = message.get("command")
        
        if command == "register_script":
            script_name = message.get("script_name")
            info = message.get("info")
            self.registered_scripts[script_name] = info
            self.logger.info(f"Script '{script_name}' registered")
            
            # Notify parent of new script registration
            self.notify_parent("script_registered", {
                "script": script_name,
                "info": info
            })
            
            return {"status": "registered"}
            
        elif command == "list_scripts":
            return {"scripts": list(self.scripts_in_directory.keys())}
            
        elif command == "collaborate":
            source = message.get("source")
            target = message.get("target")
            data = message.get("data")
            
            # Store collaboration request
            if target not in self.collaboration_requests:
                self.collaboration_requests[target] = []
            self.collaboration_requests[target].append({
                "source": source,
                "data": data,
                "timestamp": time.time()
            })
            
            # Notify target script if it's registered
            if target in self.registered_scripts:
                self.notify_script(target, "collaboration_request", {
                    "from": source,
                    "data": data
                })
                
            return {"status": "collaboration_requested"}
            
        elif command == "broadcast":
            source = message.get("source")
            data = message.get("data")
            
            # Broadcast to all registered scripts
            for script_name in self.registered_scripts:
                if script_name != source:
                    self.notify_script(script_name, "broadcast", {
                        "from": source,
                        "data": data
                    })
                    
            return {"status": "broadcast_sent"}
            
        elif command == "get_collaborations":
            script_name = message.get("script_name")
            if script_name in self.collaboration_requests:
                requests = self.collaboration_requests[script_name]
                self.collaboration_requests[script_name] = []
                return {"requests": requests}
            return {"requests": []}
            
        elif command == "notification":
            script = message.get("script")
            event = message.get("event")
            data = message.get("data", {})
            
            self.logger.info(f"Notification from '{script}': {event}")
            
            # Update registered script info
            if script in self.registered_scripts:
                if event == "results_updated":
                    self.registered_scripts[script]["results"] = data
                elif event == "state_changed":
                    self.registered_scripts[script]["state"] = data.get("state")
                    
            # Forward notification to parent
            self.notify_parent("script_notification", {
                "script": script,
                "event": event,
                "data": data
            })
                    
            return {"status": "acknowledged"}
            
        return {"error": "Unknown command"}
            
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
                'capabilities': ['script_control', 'parameter_control', 'collaboration']
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
                if item.is_file() and item.suffix == '.py':
                    if item.name not in ['hivemind_connector.py', 'connector.py', 'script_interface.py']:
                        self.scripts_in_directory[item.stem] = str(item)
                        
                        # Check if script implements ScriptInterface
                        if self.check_script_interface(item):
                            self.logger.info(f"Script '{item.stem}' supports ScriptInterface")
                            
                elif item.is_dir() and not self.should_skip_directory(item):
                    # Check for child connector
                    child_connector = item / 'hivemind_connector.py'
                    if child_connector.exists():
                        self.child_connectors[item.name] = str(child_connector)
                        
            self.logger.info(f"Found {len(self.scripts_in_directory)} scripts and {len(self.child_connectors)} child connectors")
        except Exception as e:
            self.logger.error(f"Error scanning directory: {e}")
            
    def check_script_interface(self, script_path):
        """Check if script implements ScriptInterface"""
        try:
            spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
            module = importlib.util.module_from_spec(spec)
            
            # Check module content without executing
            with open(script_path, 'r') as f:
                content = f.read()
                return 'ScriptInterface' in content and 'from script_interface import' in content
        except:
            return False
            
    def should_skip_directory(self, path):
        """Check if directory should be skipped"""
        skip_dirs = {
            'venv', 'env', '.env', '__pycache__', '.git', 
            'node_modules', '.venv', 'virtualenv', '.tox',
            'build', 'dist', '.pytest_cache', '.mypy_cache'
        }
        return path.name.startswith('.') or path.name.lower() in skip_dirs
        
    def execute_script(self, script_name, with_connector=True):
        """Execute a script in the directory"""
        if script_name not in self.scripts_in_directory:
            return {'error': 'Script not found'}
            
        if script_name in self.script_processes and self.script_processes[script_name].poll() is None:
            return {'error': 'Script already running'}
            
        try:
            cmd = [sys.executable, self.scripts_in_directory[script_name]]
            if with_connector:
                cmd.append("--with-connector")
                
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.script_processes[script_name] = process
            
            return {
                'status': 'executed',
                'script': script_name,
                'pid': process.pid,
                'with_connector': with_connector
            }
        except Exception as e:
            return {'error': f'Execution failed: {str(e)}'}
            
    def stop_script(self, script_name):
        """Stop a running script"""
        if script_name not in self.script_processes:
            return {'error': 'Script not running'}
            
        process = self.script_processes[script_name]
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                
        del self.script_processes[script_name]
        
        # Remove from registered scripts
        if script_name in self.registered_scripts:
            del self.registered_scripts[script_name]
            
        return {'status': 'stopped', 'script': script_name}
        
    def control_script(self, script_name, control_command):
        """Send control command to a script"""
        if script_name not in self.registered_scripts:
            return {'error': 'Script not registered'}
            
        # Forward control command to script
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', self.script_control_port))
            
            message = {
                'target_script': script_name,
                'command': control_command
            }
            
            sock.send(json.dumps(message).encode())
            response = sock.recv(4096).decode()
            sock.close()
            
            return json.loads(response)
        except Exception as e:
            return {'error': f'Control failed: {str(e)}'}
            
    def get_all_scripts_info(self):
        """Get information about all scripts"""
        info = {
            'available_scripts': list(self.scripts_in_directory.keys()),
            'registered_scripts': {},
            'running_scripts': []
        }
        
        # Add registered script details
        for script_name, script_info in self.registered_scripts.items():
            info['registered_scripts'][script_name] = {
                'state': script_info.get('state'),
                'parameters': list(script_info.get('parameters', {}).keys()),
                'variables': list(script_info.get('variables', {}).keys()),
                'results': script_info.get('results', {})
            }
            
        # Add running script PIDs
        for script_name, process in self.script_processes.items():
            if process.poll() is None:
                info['running_scripts'].append({
                    'name': script_name,
                    'pid': process.pid
                })
                
        return info
        
    def notify_parent(self, event, data):
        """Notify parent connector of events"""
        if self.parent_socket:
            try:
                message = {
                    'command': 'child_notification',
                    'connector_id': self.connector_id,
                    'event': event,
                    'data': data
                }
                self.parent_socket.send(json.dumps(message).encode())
            except:
                # Reconnect if needed
                self.connect_to_parent()
                
    def notify_script(self, script_name, event, data):
        """Notify a specific script of an event"""
        # This would be implemented based on how scripts register for notifications
        self.logger.info(f"Notifying script '{script_name}' of event '{event}'")
            
    def troubleshoot_connections(self):
        """Enhanced troubleshooting for all connections and scripts"""
        issues = []
        
        # Check parent connection
        if not self.parent_socket:
            issues.append("Not connected to parent")
        else:
            try:
                self.parent_socket.send(b'')
            except:
                issues.append("Parent connection lost")
        
        # Check listening socket
        if not self.socket:
            issues.append("Not listening for connections")
            
        # Check script control server
        try:
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.connect(('localhost', self.script_control_port))
            test_sock.close()
        except:
            issues.append("Script control server not responding")
            
        # Check child connectors
        for name, path in self.child_connectors.items():
            if not Path(path).exists():
                issues.append(f"Child connector missing: {name}")
                
        # Check running scripts
        dead_scripts = []
        for script_name, process in self.script_processes.items():
            if process.poll() is not None:
                dead_scripts.append(script_name)
                
        for script in dead_scripts:
            del self.script_processes[script]
            if script in self.registered_scripts:
                del self.registered_scripts[script]
                
        if dead_scripts:
            issues.append(f"Dead scripts cleaned up: {dead_scripts}")
                
        return {
            'connector_id': self.connector_id,
            'issues': issues,
            'healthy': len(issues) == 0,
            'scripts': {
                'available': len(self.scripts_in_directory),
                'registered': len(self.registered_scripts),
                'running': len(self.script_processes)
            }
        }
        
    def heartbeat_loop(self):
        """Send heartbeat to parent with enhanced status"""
        while self.running:
            try:
                if self.parent_socket:
                    heartbeat = {
                        'command': 'heartbeat',
                        'connector_id': self.connector_id,
                        'timestamp': time.time(),
                        'status': {
                            'scripts_running': len([p for p in self.script_processes.values() if p.poll() is None]),
                            'scripts_registered': len(self.registered_scripts),
                            'pending_collaborations': sum(len(reqs) for reqs in self.collaboration_requests.values())
                        }
                    }
                    self.parent_socket.send(json.dumps(heartbeat).encode())
            except:
                # Reconnect if connection lost
                self.connect_to_parent()
                
            time.sleep(30)  # Heartbeat every 30 seconds

if __name__ == "__main__":
    connector = EnhancedHivemindConnector()
    try:
        connector.start()
    except KeyboardInterrupt:
        connector.running = False
        print("\nEnhanced Hivemind Connector stopped")