#!/usr/bin/env python3
"""
Enhanced Hivemind Connector for: modules/iteration7-enterprise-full
Integrates with the enhanced connector system for full script control
Connector ID: 3e1c53d2
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

# Import the enhanced connector
try:
    from connector import get_connector, EnhancedConnector
except ImportError:
    # Fallback if connector is not available
    get_connector = None
    EnhancedConnector = None


class HivemindConnector:
    def __init__(self):
        self.connector_id = "3e1c53d2"
        self.directory = Path("/home/jarvis/Documents/GitHub/polar-bear/modules/iteration7-enterprise-full")
        self.port = 10087
        self.parent_port = 10004
        self.depth = 2
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {}
        self.scripts_in_directory = {}
        
        # Enhanced connector integration
        self.enhanced_connector = None
        if get_connector:
            self.enhanced_connector = get_connector()
        
        # Setup logging
        self.logger = logging.getLogger(f"HivemindConnector_{self.connector_id}")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def start(self):
        """Start the connector"""
        self.logger.info(f"Starting hivemind connector {self.connector_id} on port {self.port}")
        
        # Initialize enhanced connector if available
        if self.enhanced_connector:
            self.enhanced_connector.start()
            self.logger.info("Enhanced connector integration enabled")
        
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
        """Process incoming message with enhanced capabilities"""
        cmd = message.get('command')
        
        if cmd == 'status':
            return self._get_status()
        elif cmd == 'scan':
            self.scan_directory()
            return {'status': 'scan_complete', 'scripts': list(self.scripts_in_directory.keys())}
        elif cmd == 'execute':
            script = message.get('script')
            mode = message.get('mode', 'independent')
            return self._execute_script(script, mode)
        elif cmd == 'set_parameter':
            return self._set_parameter(message)
        elif cmd == 'get_parameter':
            return self._get_parameter(message)
        elif cmd == 'call_function':
            return self._call_function(message)
        elif cmd == 'get_script_info':
            return self._get_script_info(message.get('script'))
        elif cmd == 'control':
            return self._control_scripts(message)
        elif cmd == 'troubleshoot':
            return self.troubleshoot_connections()
        else:
            return {'error': 'Unknown command'}
    
    def _get_status(self):
        """Get enhanced status including script details"""
        status = {
            'status': 'active',
            'connector_id': self.connector_id,
            'directory': str(self.directory),
            'depth': self.depth,
            'scripts': len(self.scripts_in_directory),
            'children': len(self.child_connectors),
            'enhanced_connector': self.enhanced_connector is not None
        }
        
        if self.enhanced_connector:
            enhanced_status = self.enhanced_connector.get_status()
            status['enhanced_status'] = enhanced_status
            
        return status
    
    def _execute_script(self, script_name, mode='independent'):
        """Execute a script with enhanced control"""
        if script_name not in self.scripts_in_directory:
            return {'error': 'Script not found'}
        
        if self.enhanced_connector and mode == 'collaborative':
            # Use enhanced connector for collaborative execution
            script = self.enhanced_connector.get_script(script_name)
            if script:
                script.run_collaborative()
                return {
                    'status': 'running_collaborative',
                    'script': script_name,
                    'mode': 'collaborative'
                }
        
        # Fallback to independent execution
        try:
            result = subprocess.run(
                [sys.executable, self.scripts_in_directory[script_name]],
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                'status': 'executed',
                'script': script_name,
                'mode': 'independent',
                'returncode': result.returncode,
                'stdout': result.stdout[-1000:],
                'stderr': result.stderr[-1000:]
            }
        except Exception as e:
            return {'error': f'Execution failed: {str(e)}'}
    
    def _set_parameter(self, message):
        """Set a parameter in a script"""
        script_name = message.get('script')
        param_name = message.get('parameter')
        value = message.get('value')
        
        if not all([script_name, param_name, value is not None]):
            return {'error': 'Missing required fields'}
        
        if self.enhanced_connector:
            try:
                self.enhanced_connector.set_parameter(script_name, param_name, value)
                return {
                    'status': 'parameter_set',
                    'script': script_name,
                    'parameter': param_name,
                    'value': value
                }
            except Exception as e:
                return {'error': f'Failed to set parameter: {str(e)}'}
        
        return {'error': 'Enhanced connector not available'}
    
    def _get_parameter(self, message):
        """Get a parameter from a script"""
        script_name = message.get('script')
        param_name = message.get('parameter')
        
        if not all([script_name, param_name]):
            return {'error': 'Missing required fields'}
        
        if self.enhanced_connector:
            try:
                value = self.enhanced_connector.get_parameter(script_name, param_name)
                return {
                    'status': 'parameter_retrieved',
                    'script': script_name,
                    'parameter': param_name,
                    'value': value
                }
            except Exception as e:
                return {'error': f'Failed to get parameter: {str(e)}'}
        
        return {'error': 'Enhanced connector not available'}
    
    def _call_function(self, message):
        """Call a function in a script"""
        script_name = message.get('script')
        function_name = message.get('function')
        args = message.get('args', [])
        kwargs = message.get('kwargs', {})
        
        if not all([script_name, function_name]):
            return {'error': 'Missing required fields'}
        
        if self.enhanced_connector:
            try:
                result = self.enhanced_connector.call_script_function(
                    script_name, function_name, *args, **kwargs
                )
                return {
                    'status': 'function_called',
                    'script': script_name,
                    'function': function_name,
                    'result': result
                }
            except Exception as e:
                return {'error': f'Failed to call function: {str(e)}'}
        
        return {'error': 'Enhanced connector not available'}
    
    def _get_script_info(self, script_name):
        """Get detailed information about a script"""
        if not script_name:
            return {'error': 'Script name required'}
        
        if self.enhanced_connector:
            script = self.enhanced_connector.get_script(script_name)
            if script:
                return {
                    'script': script_name,
                    'loaded': script.module is not None,
                    'running': script.running,
                    'functions': list(script.functions.keys()),
                    'classes': list(script.classes.keys()),
                    'attributes': list(script.attributes.keys()),
                    'state': script.state
                }
        
        return {'error': 'Script not found or enhanced connector not available'}
    
    def _control_scripts(self, message):
        """Control script execution"""
        action = message.get('action')
        target = message.get('target', 'all')
        
        if not self.enhanced_connector:
            return {'error': 'Enhanced connector not available'}
        
        try:
            if action == 'run_all':
                if target == 'collaborative':
                    self.enhanced_connector.run_all_collaborative()
                    return {'status': 'all_scripts_running_collaborative'}
                else:
                    results = self.enhanced_connector.run_all_independent()
                    return {
                        'status': 'all_scripts_executed',
                        'results': {k: v.returncode for k, v in results.items()}
                    }
            elif action == 'stop_all':
                self.enhanced_connector.stop_all()
                return {'status': 'all_scripts_stopped'}
            elif action == 'save_state':
                self.enhanced_connector.save_state()
                return {'status': 'state_saved'}
            elif action == 'load_state':
                self.enhanced_connector.load_state()
                return {'status': 'state_loaded'}
            else:
                return {'error': 'Unknown control action'}
        except Exception as e:
            return {'error': f'Control action failed: {str(e)}'}
            
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
                'capabilities': {
                    'enhanced_connector': self.enhanced_connector is not None,
                    'script_control': True,
                    'parameter_control': True,
                    'function_calls': True
                }
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
            
            # If enhanced connector is available, ensure it has scanned too
            if self.enhanced_connector and not self.enhanced_connector.scripts:
                self.enhanced_connector.discover_scripts()
                        
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
            
    def troubleshoot_connections(self):
        """Enhanced troubleshooting with connector details"""
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
        
        # Check enhanced connector
        if not self.enhanced_connector:
            issues.append("Enhanced connector not initialized")
        else:
            ec_status = self.enhanced_connector.get_status()
            if not ec_status['running']:
                issues.append("Enhanced connector not running")
                
        return {
            'connector_id': self.connector_id,
            'issues': issues,
            'healthy': len(issues) == 0,
            'scripts_available': len(self.scripts_in_directory),
            'enhanced_features': self.enhanced_connector is not None
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
                        'scripts_loaded': len(self.scripts_in_directory),
                        'enhanced_active': self.enhanced_connector is not None
                    }
                    self.parent_socket.send(json.dumps(heartbeat).encode())
            except:
                # Reconnect if connection lost
                self.connect_to_parent()
                
            time.sleep(30)  # Heartbeat every 30 seconds
    
    def stop(self):
        """Stop the connector and cleanup"""
        self.running = False
        
        # Stop enhanced connector if available
        if self.enhanced_connector:
            self.enhanced_connector.stop_all()
            self.enhanced_connector.save_state()
        
        # Close sockets
        if self.socket:
            self.socket.close()
        if self.parent_socket:
            self.parent_socket.close()
            
        self.logger.info("Hivemind connector stopped")


if __name__ == "__main__":
    connector = HivemindConnector()
    try:
        connector.start()
    except KeyboardInterrupt:
        connector.stop()
        print("\nConnector stopped")