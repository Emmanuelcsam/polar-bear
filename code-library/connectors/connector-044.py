#!/usr/bin/env python3
"""
Connector Interface Module
Provides a standardized interface for scripts to communicate with the hivemind connector system.
Allows scripts to run independently or as part of the hivemind collective.
"""

import json
import socket
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import inspect
import functools

class ConnectorInterface:
    """Interface for scripts to communicate with the hivemind connector system"""
    
    def __init__(self, script_name: str = None):
        """Initialize the connector interface"""
        if script_name is None:
            # Get the calling script name from the stack
            frame = inspect.stack()[1]
            script_name = Path(frame.filename).name
            
        self.script_name = script_name
        self.connector_port = 10314  # Default hivemind connector port
        self.logger = logging.getLogger(f"ConnectorInterface_{script_name}")
        self.is_connected = False
        self.parameters = {}
        self.callbacks = {}
        
    def connect(self) -> bool:
        """Connect to the hivemind connector"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)  # 1 second timeout
            sock.connect(('localhost', self.connector_port))
            
            # Register script with connector
            msg = {
                'command': 'register_script',
                'script': self.script_name,
                'parameters': self.parameters
            }
            sock.send(json.dumps(msg).encode())
            
            response = sock.recv(4096).decode()
            result = json.loads(response)
            
            sock.close()
            
            if result.get('status') == 'registered':
                self.is_connected = True
                self.logger.info(f"Successfully connected to hivemind connector")
                return True
                
        except Exception as e:
            self.logger.debug(f"Could not connect to hivemind: {e}")
            
        self.is_connected = False
        return False
        
    def send_status(self, status: Dict[str, Any]) -> bool:
        """Send status update to the hivemind"""
        if not self.is_connected:
            return False
            
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', self.connector_port))
            
            msg = {
                'command': 'update_status',
                'script': self.script_name,
                'status': status
            }
            sock.send(json.dumps(msg).encode())
            sock.close()
            return True
            
        except Exception:
            return False
            
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters from the hivemind"""
        if not self.is_connected:
            return {}
            
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', self.connector_port))
            
            msg = {
                'command': 'get_parameters',
                'script': self.script_name
            }
            sock.send(json.dumps(msg).encode())
            
            response = sock.recv(4096).decode()
            result = json.loads(response)
            sock.close()
            
            return result.get('parameters', {})
            
        except Exception:
            return {}
            
    def register_parameter(self, name: str, default: Any, description: str = ""):
        """Register a parameter that can be controlled by the hivemind"""
        self.parameters[name] = {
            'default': default,
            'value': default,
            'description': description,
            'type': type(default).__name__
        }
        
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value, checking hivemind first, then using default"""
        if self.is_connected:
            hivemind_params = self.get_parameters()
            if name in hivemind_params:
                return hivemind_params[name]
                
        if name in self.parameters:
            return self.parameters[name].get('value', default)
            
        return default
        
    def register_callback(self, name: str, callback: Callable):
        """Register a callback that can be triggered by the hivemind"""
        self.callbacks[name] = callback
        
    def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command from the hivemind"""
        cmd_type = command.get('type')
        
        if cmd_type == 'set_parameter':
            param_name = command.get('parameter')
            value = command.get('value')
            if param_name in self.parameters:
                self.parameters[param_name]['value'] = value
                return {'status': 'success', 'parameter': param_name, 'value': value}
                
        elif cmd_type == 'call_function':
            func_name = command.get('function')
            args = command.get('args', [])
            kwargs = command.get('kwargs', {})
            if func_name in self.callbacks:
                try:
                    result = self.callbacks[func_name](*args, **kwargs)
                    return {'status': 'success', 'result': result}
                except Exception as e:
                    return {'status': 'error', 'error': str(e)}
                    
        return {'status': 'error', 'error': 'Unknown command'}
        
    def listen_for_commands(self):
        """Listen for commands from the hivemind (blocking)"""
        if not self.is_connected:
            return
            
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', 0))  # Bind to any available port
        port = sock.getsockname()[1]
        
        # Notify hivemind of listening port
        notify_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        notify_sock.connect(('localhost', self.connector_port))
        msg = {
            'command': 'script_listening',
            'script': self.script_name,
            'port': port
        }
        notify_sock.send(json.dumps(msg).encode())
        notify_sock.close()
        
        sock.listen(1)
        
        while True:
            try:
                conn, addr = sock.accept()
                data = conn.recv(4096).decode()
                command = json.loads(data)
                
                response = self.execute_command(command)
                conn.send(json.dumps(response).encode())
                conn.close()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Error handling command: {e}")
                
        sock.close()

def connector_aware(func):
    """Decorator to make a function connector-aware"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if running under hivemind control
        connector = ConnectorInterface(func.__module__)
        if connector.connect():
            # Get parameters from hivemind
            params = connector.get_parameters()
            kwargs.update(params)
            
        return func(*args, **kwargs)
    return wrapper

# Convenience functions for scripts
def setup_connector(script_name: str = None) -> ConnectorInterface:
    """Setup connector interface for a script"""
    connector = ConnectorInterface(script_name)
    connector.connect()
    return connector

def get_hivemind_parameter(name: str, default: Any = None, connector: ConnectorInterface = None) -> Any:
    """Get a parameter from hivemind or use default"""
    if connector is None:
        connector = ConnectorInterface()
        connector.connect()
    return connector.get_parameter(name, default)

def send_hivemind_status(status: Dict[str, Any], connector: ConnectorInterface = None) -> bool:
    """Send status to hivemind"""
    if connector is None:
        connector = ConnectorInterface()
        connector.connect()
    return connector.send_status(status)