#!/usr/bin/env python3
"""
Hivemind Connector for: modules/analysis-reporting
Connector ID: 499cbf4c
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
from typing import Dict, Any, List, Optional

# Import the script interface for enhanced control
try:
    from script_interface import get_script_manager, get_connector_interface
except ImportError:
    get_script_manager = None
    get_connector_interface = None

class HivemindConnector:
    def __init__(self):
        self.connector_id = "499cbf4c"
        self.directory = Path("/home/jarvis/Documents/GitHub/polar-bear/modules/analysis-reporting")
        self.port = 10109
        self.parent_port = 10004
        self.depth = 2
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {}
        self.scripts_in_directory = {}
        self.script_manager = None
        self.connector_interface = None
        self.execution_history = []  # Track execution history
        self.parameter_overrides = {}  # Store parameter overrides
        
        # Setup logging
        self.logger = logging.getLogger(f"Connector_{self.connector_id}")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize script interface if available
        if get_script_manager:
            self.script_manager = get_script_manager()
            self.connector_interface = get_connector_interface()
            self.logger.info("Enhanced script control enabled")
        
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
        """Process incoming message with enhanced capabilities"""
        cmd = message.get('command')
        
        # Enhanced status with script management info
        if cmd == 'status':
            status = {
                'status': 'active',
                'connector_id': self.connector_id,
                'directory': str(self.directory),
                'depth': self.depth,
                'scripts': len(self.scripts_in_directory),
                'children': len(self.child_connectors),
                'enhanced_control': self.script_manager is not None,
                'execution_history': len(self.execution_history)
            }
            
            # Add script statuses if available
            if self.script_manager:
                scripts = self.script_manager.get_all_scripts()
                status['script_statuses'] = {
                    s.name: s.status for s in scripts
                }
            
            return status
            
        elif cmd == 'scan':
            self.scan_directory()
            return {'status': 'scan_complete', 'scripts': list(self.scripts_in_directory.keys())}
            
        elif cmd == 'execute':
            script = message.get('script')
            parameters = message.get('parameters', {})
            return self.execute_script_enhanced(script, parameters)
            
        elif cmd == 'troubleshoot':
            return self.troubleshoot_connections()
            
        # New enhanced commands
        elif cmd == 'list_parameters':
            script = message.get('script')
            return self.list_script_parameters(script)
            
        elif cmd == 'update_parameter':
            script = message.get('script')
            param = message.get('parameter')
            value = message.get('value')
            return self.update_script_parameter(script, param, value)
            
        elif cmd == 'get_history':
            limit = message.get('limit', 10)
            return self.get_execution_history(limit)
            
        elif cmd == 'monitor':
            return self.get_monitoring_data()
            
        # Delegate to script interface if available
        elif self.connector_interface and cmd in [
            'list_scripts', 'get_script_info', 'get_parameter', 
            'get_results', 'reload_config'
        ]:
            return self.connector_interface.handle_command(message)
            
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
        
    def execute_script_enhanced(self, script_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a script with enhanced control and parameter support"""
        if script_name not in self.scripts_in_directory:
            return {'error': 'Script not found'}
        
        execution_record = {
            'script': script_name,
            'parameters': parameters or {},
            'timestamp': time.time(),
            'status': 'started'
        }
        
        try:
            # Use script manager if available for better control
            if self.script_manager and script_name in [s.name for s in self.script_manager.get_all_scripts()]:
                # Merge with any stored parameter overrides
                merged_params = {}
                if script_name in self.parameter_overrides:
                    merged_params.update(self.parameter_overrides[script_name])
                if parameters:
                    merged_params.update(parameters)
                
                # Execute through script manager
                result = self.script_manager.execute_script(
                    script_name, 
                    merged_params, 
                    async_mode=False
                )
                
                execution_record['status'] = 'completed'
                execution_record['result'] = result
                
            else:
                # Fallback to subprocess execution
                cmd = [sys.executable, self.scripts_in_directory[script_name]]
                
                # Add parameters as command line arguments if provided
                if parameters:
                    for key, value in parameters.items():
                        cmd.extend([f'--{key}', str(value)])
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.directory
                )
                
                execution_record['status'] = 'completed'
                execution_record['result'] = {
                    'returncode': result.returncode,
                    'stdout': result.stdout[-1000:],
                    'stderr': result.stderr[-1000:]
                }
            
            # Store in history
            self.execution_history.append(execution_record)
            if len(self.execution_history) > 100:  # Keep last 100 executions
                self.execution_history.pop(0)
            
            return {
                'status': 'executed',
                'execution_id': len(self.execution_history),
                **execution_record
            }
            
        except Exception as e:
            execution_record['status'] = 'failed'
            execution_record['error'] = str(e)
            self.execution_history.append(execution_record)
            return {'error': f'Execution failed: {str(e)}'}
    
    def list_script_parameters(self, script_name: str) -> Dict[str, Any]:
        """List available parameters for a script"""
        if self.script_manager and script_name in [s.name for s in self.script_manager.get_all_scripts()]:
            script_info = self.script_manager.get_script_info(script_name)
            if script_info and script_info.parameters:
                return {
                    'script': script_name,
                    'parameters': script_info.parameters,
                    'overrides': self.parameter_overrides.get(script_name, {})
                }
        
        return {
            'script': script_name,
            'parameters': {},
            'note': 'Parameter information not available for this script'
        }
    
    def update_script_parameter(self, script_name: str, param_name: str, value: Any) -> Dict[str, Any]:
        """Update a script parameter override"""
        if script_name not in self.parameter_overrides:
            self.parameter_overrides[script_name] = {}
        
        self.parameter_overrides[script_name][param_name] = value
        
        # Also update in script manager if available
        if self.script_manager:
            self.script_manager.update_parameter(script_name, param_name, value)
        
        return {
            'status': 'updated',
            'script': script_name,
            'parameter': param_name,
            'value': value
        }
    
    def get_execution_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent execution history"""
        history = self.execution_history[-limit:] if limit > 0 else self.execution_history
        return {
            'history': history,
            'total_executions': len(self.execution_history)
        }
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data"""
        monitoring_data = {
            'connector_id': self.connector_id,
            'uptime': time.time() - getattr(self, 'start_time', time.time()),
            'scripts_available': len(self.scripts_in_directory),
            'child_connectors': len(self.child_connectors),
            'execution_stats': {
                'total': len(self.execution_history),
                'successful': sum(1 for e in self.execution_history if e.get('status') == 'completed'),
                'failed': sum(1 for e in self.execution_history if e.get('status') == 'failed')
            }
        }
        
        # Add script manager stats if available
        if self.script_manager:
            scripts = self.script_manager.get_all_scripts()
            monitoring_data['script_states'] = {
                'idle': sum(1 for s in scripts if s.status == 'idle'),
                'running': sum(1 for s in scripts if s.status == 'running'),
                'completed': sum(1 for s in scripts if s.status == 'completed'),
                'failed': sum(1 for s in scripts if s.status == 'failed')
            }
        
        return monitoring_data
            
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

if __name__ == "__main__":
    connector = HivemindConnector()
    connector.start_time = time.time()  # Track start time for monitoring
    
    try:
        connector.start()
    except KeyboardInterrupt:
        connector.running = False
        if connector.script_manager:
            connector.script_manager.stop()
        print("\nConnector stopped")
