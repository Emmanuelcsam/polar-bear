#!/usr/bin/env python3
"""
Connector Interface for bidirectional communication between scripts and connectors.
This module provides the base classes and utilities for full integration.
"""

import json
import socket
import threading
import time
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Callable, Optional, List
import logging

class ScriptConnectorInterface:
    """Base class for making scripts connector-aware while maintaining independence."""
    
    def __init__(self, script_name: str, script_path: Optional[str] = None):
        self.script_name = script_name
        self.script_path = script_path or str(Path(inspect.getfile(self.__class__)).resolve())
        self.parameters = {}
        self.variables = {}
        self.callbacks = {}
        self.is_connected = False
        self.connector_socket = None
        self.listen_port = None
        self.running = True
        
        # Setup logging
        self.logger = logging.getLogger(f"Script_{script_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Try to connect to connector if available
        self._try_connect_to_connector()
        
    def _try_connect_to_connector(self):
        """Attempt to connect to the hivemind connector if available."""
        try:
            # Look for connector port in environment or default
            connector_port = 10006  # Default port for this directory
            
            self.connector_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connector_socket.settimeout(1.0)
            self.connector_socket.connect(('localhost', connector_port))
            
            # Register with connector
            register_msg = {
                'command': 'register_script',
                'script_name': self.script_name,
                'script_path': self.script_path,
                'capabilities': self._get_capabilities()
            }
            self.connector_socket.send(json.dumps(register_msg).encode())
            
            # Start listening thread
            self.listen_thread = threading.Thread(target=self._listen_for_commands)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            
            self.is_connected = True
            self.logger.info(f"Connected to hivemind connector on port {connector_port}")
            
        except Exception as e:
            # Connection failed - script will run independently
            self.is_connected = False
            if self.connector_socket:
                self.connector_socket.close()
                self.connector_socket = None
            
    def _get_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of this script."""
        return {
            'parameters': list(self.parameters.keys()),
            'variables': list(self.variables.keys()),
            'methods': [name for name, method in inspect.getmembers(self, predicate=inspect.ismethod) 
                       if not name.startswith('_') and hasattr(method, 'connector_exposed')],
            'info': self.get_info()
        }
        
    def _listen_for_commands(self):
        """Listen for commands from the connector."""
        while self.running and self.is_connected:
            try:
                data = self.connector_socket.recv(4096)
                if not data:
                    break
                    
                message = json.loads(data.decode())
                response = self._process_command(message)
                
                if response is not None:
                    self.connector_socket.send(json.dumps(response).encode())
                    
            except socket.timeout:
                continue
            except Exception as e:
                self.logger.error(f"Error in command listener: {e}")
                break
                
        self.is_connected = False
        
    def _process_command(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a command from the connector."""
        cmd = message.get('command')
        
        try:
            if cmd == 'get_parameter':
                param = message.get('parameter')
                if param in self.parameters:
                    return {'status': 'success', 'value': self.parameters[param]}
                return {'status': 'error', 'message': 'Parameter not found'}
                
            elif cmd == 'set_parameter':
                param = message.get('parameter')
                value = message.get('value')
                if param in self.parameters:
                    self.parameters[param] = value
                    if param in self.callbacks:
                        self.callbacks[param](value)
                    return {'status': 'success'}
                return {'status': 'error', 'message': 'Parameter not found'}
                
            elif cmd == 'get_variable':
                var = message.get('variable')
                if var in self.variables:
                    return {'status': 'success', 'value': self.variables[var]}
                return {'status': 'error', 'message': 'Variable not found'}
                
            elif cmd == 'set_variable':
                var = message.get('variable')
                value = message.get('value')
                if var in self.variables:
                    self.variables[var] = value
                    return {'status': 'success'}
                return {'status': 'error', 'message': 'Variable not found'}
                
            elif cmd == 'call_method':
                method_name = message.get('method')
                args = message.get('args', [])
                kwargs = message.get('kwargs', {})
                
                method = getattr(self, method_name, None)
                if method and hasattr(method, 'connector_exposed'):
                    result = method(*args, **kwargs)
                    return {'status': 'success', 'result': result}
                return {'status': 'error', 'message': 'Method not found or not exposed'}
                
            elif cmd == 'get_info':
                return {'status': 'success', 'info': self.get_info()}
                
            elif cmd == 'get_state':
                return {
                    'status': 'success',
                    'state': {
                        'parameters': self.parameters,
                        'variables': self.variables,
                        'info': self.get_info()
                    }
                }
                
            else:
                return {'status': 'error', 'message': f'Unknown command: {cmd}'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def register_parameter(self, name: str, default_value: Any, callback: Optional[Callable] = None):
        """Register a parameter that can be controlled by the connector."""
        self.parameters[name] = default_value
        if callback:
            self.callbacks[name] = callback
            
    def register_variable(self, name: str, initial_value: Any):
        """Register a variable that can be monitored by the connector."""
        self.variables[name] = initial_value
        
    def update_variable(self, name: str, value: Any):
        """Update a variable value."""
        if name in self.variables:
            self.variables[name] = value
            
            # Notify connector if connected
            if self.is_connected:
                try:
                    notification = {
                        'command': 'variable_updated',
                        'script_name': self.script_name,
                        'variable': name,
                        'value': value
                    }
                    self.connector_socket.send(json.dumps(notification).encode())
                except:
                    pass
                    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this script. Override in subclasses."""
        return {
            'name': self.script_name,
            'path': self.script_path,
            'connected': self.is_connected
        }
        
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.connector_socket:
            self.connector_socket.close()


def connector_exposed(func):
    """Decorator to mark methods as exposed to the connector."""
    func.connector_exposed = True
    return func


class ConnectorClient:
    """Client for connectors to interact with scripts."""
    
    def __init__(self, connector_port: int = 10006):
        self.connector_port = connector_port
        self.scripts = {}
        
    def get_script_info(self, script_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered script."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', self.connector_port))
            
            msg = {
                'command': 'get_script_info',
                'script_name': script_name
            }
            sock.send(json.dumps(msg).encode())
            
            response = json.loads(sock.recv(4096).decode())
            sock.close()
            
            return response.get('info') if response.get('status') == 'success' else None
            
        except Exception as e:
            print(f"Error getting script info: {e}")
            return None
            
    def set_script_parameter(self, script_name: str, parameter: str, value: Any) -> bool:
        """Set a parameter in a script."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', self.connector_port))
            
            msg = {
                'command': 'control_script',
                'script_name': script_name,
                'action': 'set_parameter',
                'parameter': parameter,
                'value': value
            }
            sock.send(json.dumps(msg).encode())
            
            response = json.loads(sock.recv(4096).decode())
            sock.close()
            
            return response.get('status') == 'success'
            
        except Exception as e:
            print(f"Error setting parameter: {e}")
            return False
            
    def get_script_variable(self, script_name: str, variable: str) -> Optional[Any]:
        """Get a variable value from a script."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', self.connector_port))
            
            msg = {
                'command': 'control_script',
                'script_name': script_name,
                'action': 'get_variable',
                'variable': variable
            }
            sock.send(json.dumps(msg).encode())
            
            response = json.loads(sock.recv(4096).decode())
            sock.close()
            
            return response.get('value') if response.get('status') == 'success' else None
            
        except Exception as e:
            print(f"Error getting variable: {e}")
            return None
            
    def call_script_method(self, script_name: str, method: str, *args, **kwargs) -> Optional[Any]:
        """Call a method in a script."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', self.connector_port))
            
            msg = {
                'command': 'control_script',
                'script_name': script_name,
                'action': 'call_method',
                'method': method,
                'args': list(args),
                'kwargs': kwargs
            }
            sock.send(json.dumps(msg).encode())
            
            response = json.loads(sock.recv(4096).decode())
            sock.close()
            
            return response.get('result') if response.get('status') == 'success' else None
            
        except Exception as e:
            print(f"Error calling method: {e}")
            return None