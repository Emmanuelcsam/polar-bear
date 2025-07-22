#!/usr/bin/env python3
"""
Script Wrapper - Provides connector integration for all scripts
Import this at the beginning of each script to enable connector features
"""

import sys
import os
import functools
import threading
import queue
import json
from typing import Any, Callable, Dict, Optional

# Track if we're running under a connector
_connector = None
_script_interface = None
_message_queue = queue.Queue()
_shared_state = {}
_event_handlers = {}


def get_connector():
    """Get the connector instance if available"""
    global _connector
    if _connector is None:
        # Check if connector injected this
        import __main__
        if hasattr(__main__, '_connector'):
            _connector = __main__._connector
    return _connector


def get_script_interface():
    """Get the script interface if available"""
    global _script_interface
    if _script_interface is None:
        # Check if connector injected this
        import __main__
        if hasattr(__main__, '_script_interface'):
            _script_interface = __main__._script_interface
    return _script_interface


def is_connected():
    """Check if running under connector control"""
    return get_connector() is not None


def expose_function(func: Callable) -> Callable:
    """Decorator to expose a function to the connector"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Log function call if connected
        if is_connected():
            connector = get_connector()
            if connector:
                connector.trigger_event('function_called', {
                    'script': os.path.basename(sys.argv[0]),
                    'function': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                })
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Log result if connected
        if is_connected():
            connector = get_connector()
            if connector:
                connector.trigger_event('function_completed', {
                    'script': os.path.basename(sys.argv[0]),
                    'function': func.__name__,
                    'result': result
                })
        
        return result
    
    # Mark as exposed for connector discovery
    wrapper._exposed = True
    return wrapper


def expose_variable(name: str, value: Any, writable: bool = True):
    """Expose a variable to the connector"""
    if is_connected():
        interface = get_script_interface()
        if interface:
            interface.set_variable(name, value)
            if not writable:
                # Track read-only variables
                if not hasattr(interface, '_readonly_vars'):
                    interface._readonly_vars = set()
                interface._readonly_vars.add(name)


def get_shared_variable(name: str, default: Any = None) -> Any:
    """Get a variable from the shared connector state"""
    if is_connected():
        connector = get_connector()
        if connector:
            return connector.shared_state.get(name, default)
    return _shared_state.get(name, default)


def set_shared_variable(name: str, value: Any):
    """Set a variable in the shared connector state"""
    if is_connected():
        connector = get_connector()
        if connector:
            connector.shared_state[name] = value
            connector.trigger_event('shared_variable_changed', {
                'name': name,
                'value': value,
                'script': os.path.basename(sys.argv[0])
            })
    else:
        _shared_state[name] = value


def send_message(message_type: str, data: Any):
    """Send a message through the connector"""
    message = {
        'type': message_type,
        'data': data,
        'source': os.path.basename(sys.argv[0]),
        'timestamp': time.time()
    }
    
    if is_connected():
        connector = get_connector()
        if connector:
            connector.broadcast_message(message)
    else:
        _message_queue.put(message)


def receive_messages(timeout: float = 0) -> Optional[Dict]:
    """Receive messages from the connector"""
    if is_connected():
        connector = get_connector()
        if connector and not connector.message_queue.empty():
            try:
                return connector.message_queue.get(timeout=timeout)
            except queue.Empty:
                return None
    else:
        try:
            return _message_queue.get(timeout=timeout)
        except queue.Empty:
            return None


def register_event_handler(event_type: str, handler: Callable):
    """Register an event handler"""
    if is_connected():
        connector = get_connector()
        if connector:
            connector.register_event_handler(event_type, handler)
    else:
        if event_type not in _event_handlers:
            _event_handlers[event_type] = []
        _event_handlers[event_type].append(handler)


def collaborative_mode(func: Callable) -> Callable:
    """Decorator to mark a function as collaborative"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_connected():
            # Running in collaborative mode
            connector = get_connector()
            if connector:
                connector.trigger_event('collaborative_start', {
                    'script': os.path.basename(sys.argv[0]),
                    'function': func.__name__
                })
            
            result = func(*args, **kwargs)
            
            if connector:
                connector.trigger_event('collaborative_end', {
                    'script': os.path.basename(sys.argv[0]),
                    'function': func.__name__,
                    'result': result
                })
            
            return result
        else:
            # Running independently
            return func(*args, **kwargs)
    
    wrapper._collaborative = True
    return wrapper


class ConnectorConfig:
    """Configuration for connector-aware scripts"""
    
    def __init__(self):
        self.script_name = os.path.basename(sys.argv[0])
        self.connected = is_connected()
        self.parameters = {}
        
    def set_parameter(self, name: str, value: Any):
        """Set a configuration parameter"""
        self.parameters[name] = value
        expose_variable(f"config_{name}", value)
        
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a configuration parameter"""
        if is_connected():
            interface = get_script_interface()
            if interface:
                value = interface.get_variable(f"config_{name}")
                if value is not None:
                    return value
        return self.parameters.get(name, default)
    
    def sync_parameters(self):
        """Sync parameters with connector"""
        if is_connected():
            interface = get_script_interface()
            if interface:
                for name, value in self.parameters.items():
                    interface.set_variable(f"config_{name}", value)


# Utility functions for common patterns
def log_to_connector(message: str, level: str = "INFO"):
    """Log a message through the connector"""
    send_message("log", {
        "message": message,
        "level": level
    })


def wait_for_signal(signal_name: str, timeout: float = None) -> bool:
    """Wait for a specific signal from the connector"""
    import time
    start_time = time.time()
    
    while True:
        msg = receive_messages(timeout=0.1)
        if msg and msg.get('type') == 'signal' and msg.get('data', {}).get('name') == signal_name:
            return True
        
        if timeout and (time.time() - start_time) > timeout:
            return False
        
        time.sleep(0.1)


def broadcast_result(result_name: str, result_value: Any):
    """Broadcast a result to other scripts"""
    send_message("result", {
        "name": result_name,
        "value": result_value
    })


# Auto-discovery helper
def auto_expose_module():
    """Automatically expose all suitable functions and variables in the calling module"""
    import inspect
    import __main__
    
    # Get the calling module
    frame = inspect.currentframe()
    if frame and frame.f_back:
        module_globals = frame.f_back.f_globals
        
        # Expose functions
        for name, obj in module_globals.items():
            if not name.startswith('_') and callable(obj) and inspect.isfunction(obj):
                # Check if it's defined in the main module
                if obj.__module__ == '__main__':
                    module_globals[name] = expose_function(obj)
        
        # Expose global variables (carefully)
        for name, obj in list(module_globals.items()):
            if not name.startswith('_') and not callable(obj) and not inspect.ismodule(obj):
                # Only expose simple types
                if isinstance(obj, (int, float, str, bool, list, dict)):
                    expose_variable(name, obj)


# Import time for timestamps
import time

# Create a global config instance
config = ConnectorConfig()