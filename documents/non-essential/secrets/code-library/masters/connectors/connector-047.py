#!/usr/bin/env python3
"""
Hivemind Connector for Polar Bear Root Directory
This connector integrates all root-level scripts with the hivemind network
"""

import socket
import threading
import json
import os
import sys
import subprocess
import time
import logging
from pathlib import Path
import uuid
import importlib.util
import inspect
import ast

# Connector Configuration
CONNECTOR_ID = "00000001"  # Root connector ID
CONNECTOR_PORT = 10001
PARENT_PORT = 10000  # Connect to master
DEPTH = 1
TIMEOUT = 30

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'hivemind_connector_{CONNECTOR_ID}')

class ScriptInterface:
    """Interface for managing and controlling scripts in the root directory"""
    
    def __init__(self):
        self.scripts = {}
        self.parameters = {}
        self.shared_state = {}
        self.discover_scripts()
    
    def discover_scripts(self):
        """Discover all Python scripts in the current directory"""
        current_dir = Path(__file__).parent
        for file in current_dir.glob("*.py"):
            if file.name not in ['hivemind_connector.py', 'connector.py', '__pycache__']:
                self.analyze_script(file)
    
    def analyze_script(self, script_path):
        """Analyze a script to extract parameters and functions"""
        script_name = script_path.stem
        
        try:
            # Parse the script
            with open(script_path, 'r') as f:
                tree = ast.parse(f.read())
            
            # Extract module-level variables (parameters)
            parameters = {}
            functions = {}
            classes = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Try to evaluate simple assignments
                            try:
                                value = ast.literal_eval(node.value)
                                parameters[target.id] = {
                                    'value': value,
                                    'type': type(value).__name__
                                }
                            except:
                                parameters[target.id] = {
                                    'value': None,
                                    'type': 'complex'
                                }
                
                elif isinstance(node, ast.FunctionDef):
                    functions[node.name] = {
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node)
                    }
                
                elif isinstance(node, ast.ClassDef):
                    classes[node.name] = {
                        'docstring': ast.get_docstring(node),
                        'methods': []
                    }
                    # Extract class methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            classes[node.name]['methods'].append(item.name)
            
            self.scripts[script_name] = {
                'path': str(script_path),
                'parameters': parameters,
                'functions': functions,
                'classes': classes,
                'module': None
            }
            
            logger.info(f"Analyzed script: {script_name}")
            
        except Exception as e:
            logger.error(f"Error analyzing {script_path}: {e}")
    
    def load_script_module(self, script_name):
        """Dynamically load a script as a module"""
        if script_name not in self.scripts:
            return None
        
        script_info = self.scripts[script_name]
        if script_info['module'] is None:
            try:
                spec = importlib.util.spec_from_file_location(
                    script_name, 
                    script_info['path']
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                script_info['module'] = module
                logger.info(f"Loaded module: {script_name}")
            except Exception as e:
                logger.error(f"Error loading module {script_name}: {e}")
                return None
        
        return script_info['module']
    
    def get_parameter(self, script_name, param_name):
        """Get a parameter value from a script"""
        module = self.load_script_module(script_name)
        if module and hasattr(module, param_name):
            return getattr(module, param_name)
        return None
    
    def set_parameter(self, script_name, param_name, value):
        """Set a parameter value in a script"""
        module = self.load_script_module(script_name)
        if module:
            setattr(module, param_name, value)
            # Update shared state
            if script_name not in self.shared_state:
                self.shared_state[script_name] = {}
            self.shared_state[script_name][param_name] = value
            return True
        return False
    
    def execute_function(self, script_name, function_name, *args, **kwargs):
        """Execute a function from a script"""
        module = self.load_script_module(script_name)
        if module and hasattr(module, function_name):
            func = getattr(module, function_name)
            if callable(func):
                return func(*args, **kwargs)
        return None
    
    def get_script_info(self):
        """Get information about all discovered scripts"""
        info = {}
        for script_name, script_data in self.scripts.items():
            info[script_name] = {
                'parameters': list(script_data['parameters'].keys()),
                'functions': list(script_data['functions'].keys()),
                'classes': list(script_data['classes'].keys())
            }
        return info

class HivemindConnector:
    def __init__(self):
        self.id = CONNECTOR_ID
        self.port = CONNECTOR_PORT
        self.parent_port = PARENT_PORT
        self.depth = DEPTH
        self.server_socket = None
        self.parent_socket = None
        self.running = False
        self.scripts = []
        self.child_connectors = []
        self.script_interface = ScriptInterface()
        self.last_heartbeat = time.time()
        
    def start(self):
        """Start the connector server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('localhost', self.port))
            self.server_socket.listen(5)
            self.running = True
            
            logger.info(f"Hivemind Connector {self.id} started on port {self.port}")
            
            # Start server thread
            server_thread = threading.Thread(target=self.accept_connections)
            server_thread.daemon = True
            server_thread.start()
            
            # Register with parent
            self.register_with_parent()
            
            # Start heartbeat thread
            heartbeat_thread = threading.Thread(target=self.send_heartbeats)
            heartbeat_thread.daemon = True
            heartbeat_thread.start()
            
            # Discover local scripts
            self.discover_scripts()
            
            # Keep running
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error starting connector: {e}")
            self.shutdown()
    
    def accept_connections(self):
        """Accept incoming connections"""
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                client_socket, address = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket,)
                )
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {e}")
    
    def handle_client(self, client_socket):
        """Handle client requests"""
        try:
            data = client_socket.recv(4096).decode('utf-8')
            if data:
                request = json.loads(data)
                response = self.process_command(request)
                client_socket.send(json.dumps(response).encode('utf-8'))
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            error_response = {'status': 'error', 'message': str(e)}
            try:
                client_socket.send(json.dumps(error_response).encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()
    
    def process_command(self, request):
        """Process incoming commands"""
        command = request.get('command', '')
        
        if command == 'status':
            return self.get_status()
        elif command == 'scan':
            return self.scan_resources()
        elif command == 'execute':
            return self.execute_script(request)
        elif command == 'troubleshoot':
            return self.troubleshoot()
        elif command == 'heartbeat':
            self.last_heartbeat = time.time()
            return {'status': 'ok', 'timestamp': self.last_heartbeat}
        elif command == 'get_scripts':
            return {'status': 'ok', 'scripts': self.script_interface.get_script_info()}
        elif command == 'get_parameter':
            script = request.get('script', '')
            param = request.get('parameter', '')
            value = self.script_interface.get_parameter(script, param)
            return {'status': 'ok', 'value': value}
        elif command == 'set_parameter':
            script = request.get('script', '')
            param = request.get('parameter', '')
            value = request.get('value')
            success = self.script_interface.set_parameter(script, param, value)
            return {'status': 'ok' if success else 'error', 'success': success}
        elif command == 'execute_function':
            script = request.get('script', '')
            function = request.get('function', '')
            args = request.get('args', [])
            kwargs = request.get('kwargs', {})
            try:
                result = self.script_interface.execute_function(script, function, *args, **kwargs)
                return {'status': 'ok', 'result': str(result)}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        else:
            return {'status': 'error', 'message': f'Unknown command: {command}'}
    
    def get_status(self):
        """Get connector status"""
        return {
            'status': 'ok',
            'connector_id': self.id,
            'port': self.port,
            'depth': self.depth,
            'scripts': len(self.scripts),
            'child_connectors': len(self.child_connectors),
            'uptime': time.time() - self.last_heartbeat,
            'discovered_scripts': self.script_interface.get_script_info()
        }
    
    def scan_resources(self):
        """Scan for scripts and child connectors"""
        self.discover_scripts()
        self.discover_child_connectors()
        return {
            'status': 'ok',
            'scripts': self.scripts,
            'child_connectors': self.child_connectors
        }
    
    def discover_scripts(self):
        """Discover Python scripts in current directory"""
        self.scripts = []
        current_dir = Path(__file__).parent
        
        for file in current_dir.glob("*.py"):
            if file.name not in ['hivemind_connector.py', 'connector.py']:
                self.scripts.append(file.name)
        
        logger.info(f"Discovered {len(self.scripts)} scripts")
    
    def discover_child_connectors(self):
        """Discover child hivemind connectors"""
        self.child_connectors = []
        # In root directory, we don't have child connectors
        # The module connectors are managed by the master
    
    def execute_script(self, request):
        """Execute a Python script"""
        script_name = request.get('script', '')
        args = request.get('args', [])
        kwargs = request.get('kwargs', {})
        
        if not script_name or script_name not in self.scripts:
            return {'status': 'error', 'message': f'Script not found: {script_name}'}
        
        try:
            # Execute script
            script_path = Path(__file__).parent / script_name
            cmd = [sys.executable, str(script_path)] + [str(arg) for arg in args]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT
            )
            
            return {
                'status': 'ok',
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {'status': 'error', 'message': f'Script timeout: {script_name}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def register_with_parent(self):
        """Register with parent connector"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries and self.running:
            try:
                parent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                parent_socket.settimeout(5)
                parent_socket.connect(('localhost', self.parent_port))
                
                # Send registration
                registration = {
                    'command': 'register',
                    'connector_id': self.id,
                    'port': self.port,
                    'depth': self.depth
                }
                
                parent_socket.send(json.dumps(registration).encode('utf-8'))
                response = parent_socket.recv(4096).decode('utf-8')
                
                if response:
                    result = json.loads(response)
                    if result.get('status') == 'ok':
                        logger.info(f"Registered with parent on port {self.parent_port}")
                        self.parent_socket = parent_socket
                        return True
                
                parent_socket.close()
                
            except Exception as e:
                logger.warning(f"Failed to register with parent (attempt {retry_count + 1}): {e}")
            
            retry_count += 1
            time.sleep(2)
        
        logger.error("Failed to register with parent after all retries")
        return False
    
    def send_heartbeats(self):
        """Send periodic heartbeats to parent"""
        while self.running:
            time.sleep(10)  # Heartbeat every 10 seconds
            if self.parent_socket:
                try:
                    heartbeat = {
                        'command': 'heartbeat',
                        'connector_id': self.id,
                        'timestamp': time.time()
                    }
                    self.parent_socket.send(json.dumps(heartbeat).encode('utf-8'))
                except Exception as e:
                    logger.warning(f"Heartbeat failed: {e}")
                    # Try to reconnect
                    self.register_with_parent()
    
    def troubleshoot(self):
        """Run troubleshooting diagnostics"""
        diagnostics = {
            'status': 'ok',
            'diagnostics': {
                'connector_id': self.id,
                'port': self.port,
                'server_running': self.running,
                'parent_connected': self.parent_socket is not None,
                'scripts_found': len(self.scripts),
                'script_details': self.script_interface.get_script_info(),
                'last_heartbeat': time.time() - self.last_heartbeat
            }
        }
        return diagnostics
    
    def shutdown(self):
        """Shutdown the connector"""
        logger.info("Shutting down connector...")
        self.running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        if self.parent_socket:
            try:
                # Send deregistration
                dereg = {
                    'command': 'deregister',
                    'connector_id': self.id
                }
                self.parent_socket.send(json.dumps(dereg).encode('utf-8'))
                self.parent_socket.close()
            except:
                pass

def main():
    """Main entry point"""
    connector = HivemindConnector()
    
    try:
        connector.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        connector.shutdown()

if __name__ == "__main__":
    main()