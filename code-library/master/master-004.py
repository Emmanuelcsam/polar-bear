#!/usr/bin/env python3
"""
Enhanced Integration System for Bidirectional Communication
Provides full control capabilities while maintaining script independence
"""

import threading
import queue
import json
import time
import inspect
import importlib.util
import sys
import os
from typing import Any, Dict, Callable, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events in the system"""
    PROGRESS = "progress"
    STATUS = "status"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DATA = "data"
    PARAMETER_CHANGE = "parameter_change"
    STATE_UPDATE = "state_update"
    FUNCTION_CALL = "function_call"
    RESULT = "result"


@dataclass
class Event:
    """Event object for communication"""
    type: EventType
    source: str
    target: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(time.time()))


class EventBus:
    """Central event bus for bidirectional communication"""
    
    def __init__(self):
        self._subscribers = {}
        self._event_queue = queue.Queue()
        self._lock = threading.Lock()
        self._running = True
        self._worker = threading.Thread(target=self._process_events, daemon=True)
        self._worker.start()
    
    def subscribe(self, event_type: EventType, callback: Callable, subscriber_id: str):
        """Subscribe to events of a specific type"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = {}
            self._subscribers[event_type][subscriber_id] = callback
            logger.info(f"Subscriber {subscriber_id} subscribed to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, subscriber_id: str):
        """Unsubscribe from events"""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].pop(subscriber_id, None)
    
    def publish(self, event: Event):
        """Publish an event"""
        self._event_queue.put(event)
    
    def _process_events(self):
        """Process events in the background"""
        while self._running:
            try:
                event = self._event_queue.get(timeout=0.1)
                with self._lock:
                    subscribers = self._subscribers.get(event.type, {}).copy()
                
                for subscriber_id, callback in subscribers.items():
                    if event.target is None or event.target == subscriber_id:
                        try:
                            callback(event)
                        except Exception as e:
                            logger.error(f"Error in subscriber {subscriber_id}: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def stop(self):
        """Stop the event bus"""
        self._running = False
        self._worker.join()


class ScriptController:
    """Enhanced script controller with full bidirectional control"""
    
    def __init__(self, script_path: str, event_bus: EventBus):
        self.script_path = script_path
        self.script_name = os.path.basename(script_path)
        self.event_bus = event_bus
        self.module = None
        self.parameters = {}
        self.state = {}
        self.callbacks = {}
        self._lock = threading.Lock()
        self._running_tasks = {}
        
        # Subscribe to relevant events
        self.event_bus.subscribe(EventType.PARAMETER_CHANGE, self._handle_parameter_change, self.script_name)
        self.event_bus.subscribe(EventType.FUNCTION_CALL, self._handle_function_call, self.script_name)
        
        # Load the script module
        self._load_module()
    
    def _load_module(self):
        """Load the script as a module"""
        try:
            spec = importlib.util.spec_from_file_location(
                self.script_name.replace('.py', ''),
                self.script_path
            )
            self.module = importlib.util.module_from_spec(spec)
            
            # Inject integration capabilities
            self._inject_integration()
            
            # Execute the module
            spec.loader.exec_module(self.module)
            
            # Extract parameters and functions
            self._extract_interface()
            
            logger.info(f"Loaded module: {self.script_name}")
        except Exception as e:
            logger.error(f"Error loading module {self.script_name}: {e}")
            raise
    
    def _inject_integration(self):
        """Inject integration capabilities into the module"""
        # Create integration object
        integration = ScriptIntegration(self)
        
        # Inject into module namespace
        self.module.__dict__['_integration'] = integration
        self.module.__dict__['publish_event'] = integration.publish_event
        self.module.__dict__['get_parameter'] = integration.get_parameter
        self.module.__dict__['set_parameter'] = integration.set_parameter
        self.module.__dict__['update_state'] = integration.update_state
        self.module.__dict__['get_state'] = integration.get_state
        self.module.__dict__['register_callback'] = integration.register_callback
        self.module.__dict__['report_progress'] = integration.report_progress
    
    def _extract_interface(self):
        """Extract parameters and functions from the module"""
        # Extract global variables as parameters
        for name, value in self.module.__dict__.items():
            if not name.startswith('_') and not callable(value) and not inspect.ismodule(value):
                if isinstance(value, (int, float, str, bool, list, dict)):
                    self.parameters[name] = value
        
        # Extract functions
        self.functions = {}
        for name, obj in inspect.getmembers(self.module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                self.functions[name] = obj
    
    def _handle_parameter_change(self, event: Event):
        """Handle parameter change events"""
        with self._lock:
            param_name = event.data.get('parameter')
            value = event.data.get('value')
            
            if param_name and hasattr(self.module, param_name):
                setattr(self.module, param_name, value)
                self.parameters[param_name] = value
                
                # Notify callbacks
                if 'parameter_change' in self.callbacks:
                    for callback in self.callbacks['parameter_change']:
                        callback(param_name, value)
                
                logger.info(f"{self.script_name}: Updated parameter {param_name} = {value}")
    
    def _handle_function_call(self, event: Event):
        """Handle function call events"""
        func_name = event.data.get('function')
        args = event.data.get('args', [])
        kwargs = event.data.get('kwargs', {})
        call_id = event.data.get('call_id', event.id)
        
        if func_name in self.functions:
            # Execute in separate thread
            thread = threading.Thread(
                target=self._execute_function,
                args=(func_name, args, kwargs, call_id, event.source)
            )
            thread.daemon = True
            thread.start()
            self._running_tasks[call_id] = thread
        else:
            # Send error response
            self.event_bus.publish(Event(
                type=EventType.ERROR,
                source=self.script_name,
                target=event.source,
                data={
                    'call_id': call_id,
                    'error': f"Function {func_name} not found"
                }
            ))
    
    def _execute_function(self, func_name: str, args: list, kwargs: dict, call_id: str, requester: str):
        """Execute a function and return the result"""
        try:
            # Get the function
            func = self.functions[func_name]
            
            # Execute
            result = func(*args, **kwargs)
            
            # Send result
            self.event_bus.publish(Event(
                type=EventType.RESULT,
                source=self.script_name,
                target=requester,
                data={
                    'call_id': call_id,
                    'function': func_name,
                    'result': result
                }
            ))
        except Exception as e:
            # Send error
            self.event_bus.publish(Event(
                type=EventType.ERROR,
                source=self.script_name,
                target=requester,
                data={
                    'call_id': call_id,
                    'function': func_name,
                    'error': str(e)
                }
            ))
        finally:
            # Clean up
            self._running_tasks.pop(call_id, None)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the script"""
        return {
            'name': self.script_name,
            'path': self.script_path,
            'parameters': self.parameters,
            'functions': list(self.functions.keys()),
            'state': self.state,
            'running_tasks': list(self._running_tasks.keys())
        }


class ScriptIntegration:
    """Integration interface injected into scripts"""
    
    def __init__(self, controller: ScriptController):
        self.controller = controller
    
    def publish_event(self, event_type: str, data: Dict[str, Any], target: Optional[str] = None):
        """Publish an event from the script"""
        try:
            event_type_enum = EventType(event_type)
        except ValueError:
            event_type_enum = EventType.INFO
        
        self.controller.event_bus.publish(Event(
            type=event_type_enum,
            source=self.controller.script_name,
            target=target,
            data=data
        ))
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value"""
        return self.controller.parameters.get(name, default)
    
    def set_parameter(self, name: str, value: Any):
        """Set a parameter value"""
        with self.controller._lock:
            self.controller.parameters[name] = value
            if hasattr(self.controller.module, name):
                setattr(self.controller.module, name, value)
        
        # Publish parameter change event
        self.publish_event('parameter_change', {'parameter': name, 'value': value})
    
    def update_state(self, key: str, value: Any):
        """Update shared state"""
        with self.controller._lock:
            self.controller.state[key] = value
        
        # Publish state update event
        self.publish_event('state_update', {'key': key, 'value': value})
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get shared state value"""
        return self.controller.state.get(key, default)
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for events"""
        with self.controller._lock:
            if event_type not in self.controller.callbacks:
                self.controller.callbacks[event_type] = []
            self.controller.callbacks[event_type].append(callback)
    
    def report_progress(self, progress: float, message: str = ""):
        """Report progress (0.0 to 1.0)"""
        self.publish_event('progress', {
            'progress': max(0.0, min(1.0, progress)),
            'message': message
        })


class EnhancedConnector:
    """Enhanced connector with full bidirectional control"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.controllers = {}
        self.shared_state = {}
        self._lock = threading.Lock()
        
        # Subscribe to events
        self.event_bus.subscribe(EventType.STATE_UPDATE, self._handle_state_update, 'connector')
        self.event_bus.subscribe(EventType.RESULT, self._handle_result, 'connector')
        self.event_bus.subscribe(EventType.ERROR, self._handle_error, 'connector')
        self.event_bus.subscribe(EventType.PROGRESS, self._handle_progress, 'connector')
        
        self._results = {}
        self._result_events = {}
    
    def load_script(self, script_path: str):
        """Load a script with full integration"""
        try:
            controller = ScriptController(script_path, self.event_bus)
            self.controllers[controller.script_name] = controller
            logger.info(f"Loaded script: {controller.script_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading script {script_path}: {e}")
            return False
    
    def call_function(self, script_name: str, function_name: str, *args, **kwargs) -> Any:
        """Call a function in a script and wait for result"""
        if script_name not in self.controllers:
            raise ValueError(f"Script {script_name} not loaded")
        
        call_id = str(time.time())
        event = threading.Event()
        self._result_events[call_id] = event
        
        # Publish function call event
        self.event_bus.publish(Event(
            type=EventType.FUNCTION_CALL,
            source='connector',
            target=script_name,
            data={
                'function': function_name,
                'args': args,
                'kwargs': kwargs,
                'call_id': call_id
            }
        ))
        
        # Wait for result (with timeout)
        if event.wait(timeout=300):  # 5 minute timeout
            result = self._results.pop(call_id, None)
            self._result_events.pop(call_id, None)
            
            if isinstance(result, Exception):
                raise result
            return result
        else:
            self._result_events.pop(call_id, None)
            raise TimeoutError(f"Function call {function_name} timed out")
    
    def set_parameter(self, script_name: str, parameter: str, value: Any):
        """Set a parameter in a script"""
        if script_name not in self.controllers:
            raise ValueError(f"Script {script_name} not loaded")
        
        self.event_bus.publish(Event(
            type=EventType.PARAMETER_CHANGE,
            source='connector',
            target=script_name,
            data={'parameter': parameter, 'value': value}
        ))
    
    def get_script_info(self, script_name: str) -> Dict[str, Any]:
        """Get information about a script"""
        if script_name not in self.controllers:
            raise ValueError(f"Script {script_name} not loaded")
        
        return self.controllers[script_name].get_info()
    
    def get_all_scripts_info(self) -> Dict[str, Any]:
        """Get information about all loaded scripts"""
        return {name: controller.get_info() for name, controller in self.controllers.items()}
    
    def _handle_state_update(self, event: Event):
        """Handle state update events"""
        key = event.data.get('key')
        value = event.data.get('value')
        
        if key:
            with self._lock:
                self.shared_state[key] = value
            logger.info(f"State updated: {key} = {value}")
    
    def _handle_result(self, event: Event):
        """Handle function result events"""
        call_id = event.data.get('call_id')
        if call_id in self._result_events:
            self._results[call_id] = event.data.get('result')
            self._result_events[call_id].set()
    
    def _handle_error(self, event: Event):
        """Handle error events"""
        call_id = event.data.get('call_id')
        if call_id in self._result_events:
            self._results[call_id] = Exception(event.data.get('error', 'Unknown error'))
            self._result_events[call_id].set()
        else:
            logger.error(f"Error from {event.source}: {event.data}")
    
    def _handle_progress(self, event: Event):
        """Handle progress events"""
        logger.info(f"Progress from {event.source}: {event.data.get('progress', 0) * 100:.1f}% - {event.data.get('message', '')}")
    
    def stop(self):
        """Stop the connector"""
        self.event_bus.stop()


# Example usage for independent script running
if __name__ == "__main__":
    # Scripts can still run independently
    print("Enhanced Integration System")
    print("This module provides bidirectional communication for script integration")
    
    # Demo
    connector = EnhancedConnector()
    print("\nEnhanced Connector initialized")
    print("Available methods:")
    print("- load_script(path): Load a script with full integration")
    print("- call_function(script, function, *args, **kwargs): Call a function")
    print("- set_parameter(script, parameter, value): Set a parameter")
    print("- get_script_info(script): Get script information")
    
    connector.stop()