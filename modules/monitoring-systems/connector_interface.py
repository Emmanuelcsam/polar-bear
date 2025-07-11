#!/usr/bin/env python3
"""
Connector Interface Module
Provides bidirectional communication and control between scripts and connectors
"""

import json
import socket
import threading
import queue
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from pathlib import Path
import inspect
import importlib.util
import sys

@dataclass
class ScriptParameter:
    """Represents a controllable parameter in a script"""
    name: str
    value: Any
    type: str
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    
@dataclass
class ScriptState:
    """Represents the current state of a script"""
    script_name: str
    status: str  # 'idle', 'running', 'error', 'paused'
    parameters: Dict[str, ScriptParameter]
    metrics: Dict[str, Any]
    last_update: float
    error_message: Optional[str] = None

class ConnectorInterface:
    """Interface for scripts to communicate with connectors"""
    
    def __init__(self, script_name: str, port: int = 10130):
        self.script_name = script_name
        self.port = port
        self.parameters: Dict[str, ScriptParameter] = {}
        self.metrics: Dict[str, Any] = {}
        self.callbacks: Dict[str, Callable] = {}
        self.status = 'idle'
        self.running = True
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.logger = logging.getLogger(f"ConnectorInterface_{script_name}")
        
        # Start communication thread
        self.comm_thread = threading.Thread(target=self._communication_loop, daemon=True)
        self.comm_thread.start()
        
    def register_parameter(self, name: str, initial_value: Any, 
                         param_type: str = None, description: str = "",
                         min_value: Optional[float] = None,
                         max_value: Optional[float] = None,
                         choices: Optional[List[Any]] = None,
                         callback: Optional[Callable] = None):
        """Register a parameter that can be controlled by the connector"""
        if param_type is None:
            param_type = type(initial_value).__name__
            
        self.parameters[name] = ScriptParameter(
            name=name,
            value=initial_value,
            type=param_type,
            description=description,
            min_value=min_value,
            max_value=max_value,
            choices=choices
        )
        
        if callback:
            self.callbacks[name] = callback
            
        self._notify_connector('parameter_registered', {
            'parameter': asdict(self.parameters[name])
        })
        
    def update_parameter(self, name: str, value: Any):
        """Update a parameter value"""
        if name not in self.parameters:
            raise ValueError(f"Parameter {name} not registered")
            
        param = self.parameters[name]
        
        # Validate value
        if param.min_value is not None and value < param.min_value:
            raise ValueError(f"Value {value} below minimum {param.min_value}")
        if param.max_value is not None and value > param.max_value:
            raise ValueError(f"Value {value} above maximum {param.max_value}")
        if param.choices is not None and value not in param.choices:
            raise ValueError(f"Value {value} not in allowed choices {param.choices}")
            
        # Update value
        old_value = param.value
        param.value = value
        
        # Call callback if registered
        if name in self.callbacks:
            try:
                self.callbacks[name](old_value, value)
            except Exception as e:
                self.logger.error(f"Error in parameter callback: {e}")
                
        self._notify_connector('parameter_updated', {
            'parameter': name,
            'old_value': old_value,
            'new_value': value
        })
        
    def get_parameter(self, name: str) -> Any:
        """Get current parameter value"""
        if name not in self.parameters:
            raise ValueError(f"Parameter {name} not registered")
        return self.parameters[name].value
        
    def update_metric(self, name: str, value: Any):
        """Update a metric value that connectors can monitor"""
        self.metrics[name] = value
        self._notify_connector('metric_updated', {
            'metric': name,
            'value': value
        })
        
    def set_status(self, status: str, error_message: Optional[str] = None):
        """Update script status"""
        self.status = status
        self._notify_connector('status_changed', {
            'status': status,
            'error_message': error_message
        })
        
    def get_state(self) -> ScriptState:
        """Get current script state"""
        return ScriptState(
            script_name=self.script_name,
            status=self.status,
            parameters=self.parameters,
            metrics=self.metrics,
            last_update=time.time()
        )
        
    def _notify_connector(self, event_type: str, data: Dict[str, Any]):
        """Send notification to connector"""
        message = {
            'event': event_type,
            'script': self.script_name,
            'timestamp': time.time(),
            'data': data
        }
        self.command_queue.put(message)
        
    def _communication_loop(self):
        """Handle communication with connector"""
        while self.running:
            try:
                # Connect to connector
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                
                try:
                    sock.connect(('localhost', self.port))
                    
                    # Send registration message
                    register_msg = {
                        'command': 'register_script',
                        'script_name': self.script_name,
                        'state': asdict(self.get_state())
                    }
                    sock.send(json.dumps(register_msg).encode())
                    
                    # Communication loop
                    while self.running:
                        # Send queued messages
                        try:
                            message = self.command_queue.get(timeout=0.1)
                            sock.send(json.dumps(message).encode())
                        except queue.Empty:
                            pass
                            
                        # Check for incoming commands
                        sock.settimeout(0.1)
                        try:
                            data = sock.recv(4096)
                            if data:
                                command = json.loads(data.decode())
                                self._handle_command(command)
                        except socket.timeout:
                            pass
                        except json.JSONDecodeError:
                            self.logger.error("Invalid JSON received from connector")
                            
                except (socket.error, ConnectionRefusedError):
                    self.logger.debug("Connector not available, retrying...")
                    time.sleep(5)
                finally:
                    sock.close()
                    
            except Exception as e:
                self.logger.error(f"Communication error: {e}")
                time.sleep(5)
                
    def _handle_command(self, command: Dict[str, Any]):
        """Handle command from connector"""
        cmd_type = command.get('command')
        
        if cmd_type == 'set_parameter':
            param_name = command.get('parameter')
            value = command.get('value')
            try:
                self.update_parameter(param_name, value)
                response = {'status': 'success', 'parameter': param_name, 'value': value}
            except Exception as e:
                response = {'status': 'error', 'message': str(e)}
                
        elif cmd_type == 'get_state':
            response = {'status': 'success', 'state': asdict(self.get_state())}
            
        elif cmd_type == 'pause':
            self.set_status('paused')
            response = {'status': 'success', 'message': 'Script paused'}
            
        elif cmd_type == 'resume':
            self.set_status('running')
            response = {'status': 'success', 'message': 'Script resumed'}
            
        elif cmd_type == 'stop':
            self.running = False
            self.set_status('stopped')
            response = {'status': 'success', 'message': 'Script stopped'}
            
        else:
            response = {'status': 'error', 'message': f'Unknown command: {cmd_type}'}
            
        self.response_queue.put(response)
        
    def wait_for_connector(self, timeout: float = 10.0) -> bool:
        """Wait for connector to be available"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                sock.connect(('localhost', self.port))
                sock.close()
                return True
            except:
                time.sleep(0.5)
        return False
        
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.set_status('stopped')
        if self.comm_thread.is_alive():
            self.comm_thread.join(timeout=2.0)


class ScriptRegistry:
    """Registry for managing script modules and their interfaces"""
    
    def __init__(self):
        self.scripts: Dict[str, Dict[str, Any]] = {}
        self.interfaces: Dict[str, ConnectorInterface] = {}
        
    def register_script(self, script_path: str, interface: ConnectorInterface):
        """Register a script with its interface"""
        script_name = Path(script_path).name
        self.scripts[script_name] = {
            'path': script_path,
            'module': None,
            'loaded': False
        }
        self.interfaces[script_name] = interface
        
    def load_script(self, script_name: str) -> Optional[Any]:
        """Dynamically load a script module"""
        if script_name not in self.scripts:
            return None
            
        script_info = self.scripts[script_name]
        if script_info['loaded']:
            return script_info['module']
            
        try:
            spec = importlib.util.spec_from_file_location(
                script_name.replace('.py', ''),
                script_info['path']
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            
            script_info['module'] = module
            script_info['loaded'] = True
            return module
            
        except Exception as e:
            logging.error(f"Failed to load script {script_name}: {e}")
            return None
            
    def get_script_functions(self, script_name: str) -> Dict[str, Callable]:
        """Get all callable functions from a script"""
        module = self.load_script(script_name)
        if not module:
            return {}
            
        functions = {}
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                functions[name] = obj
        return functions
        
    def call_script_function(self, script_name: str, function_name: str, 
                           *args, **kwargs) -> Any:
        """Call a function in a script"""
        functions = self.get_script_functions(script_name)
        if function_name not in functions:
            raise ValueError(f"Function {function_name} not found in {script_name}")
        return functions[function_name](*args, **kwargs)


# Decorator for easy integration
def connector_enabled(connector_port: int = 10130):
    """Decorator to enable connector interface for a script"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            script_name = Path(inspect.getfile(func)).name
            interface = ConnectorInterface(script_name, connector_port)
            
            # Inject interface into function's globals
            func.__globals__['connector_interface'] = interface
            
            try:
                interface.set_status('running')
                result = func(*args, **kwargs)
                interface.set_status('idle')
                return result
            except Exception as e:
                interface.set_status('error', str(e))
                raise
            finally:
                interface.cleanup()
                
        return wrapper
    return decorator