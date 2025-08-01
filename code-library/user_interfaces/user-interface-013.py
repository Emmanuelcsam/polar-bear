#!/usr/bin/env python3
"""
Unified Script Interface for ML Models Module
Provides a base class and utilities for integrating scripts with the connector system
"""

import json
import logging
import socket
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import queue
import sys
import os

class ScriptInterface(ABC):
    """Base class for all ML model scripts to enable connector integration"""
    
    def __init__(self, script_name: str, description: str = ""):
        self.script_name = script_name
        self.description = description
        self.parameters = {}
        self.variables = {}
        self.state = "initialized"
        self.results = {}
        self.callbacks = {}
        self.message_queue = queue.Queue()
        self.running = False
        self.connector_port = 10117  # Default ML models connector port
        
        # Setup logging
        self.logger = logging.getLogger(script_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Register default parameters
        self._register_default_parameters()
        
        # Start message handling thread
        self.message_thread = threading.Thread(target=self._handle_messages, daemon=True)
        self.message_thread.start()
        
    def _register_default_parameters(self):
        """Register default parameters that all scripts should have"""
        self.register_parameter("log_level", "INFO", ["DEBUG", "INFO", "WARNING", "ERROR"])
        self.register_parameter("output_format", "json", ["json", "csv", "text"])
        self.register_parameter("max_iterations", 100, range(1, 10000))
        self.register_parameter("enabled", True, [True, False])
        
    def register_parameter(self, name: str, default_value: Any, valid_values: Optional[List] = None):
        """Register a controllable parameter"""
        self.parameters[name] = {
            "value": default_value,
            "default": default_value,
            "valid_values": valid_values,
            "type": type(default_value).__name__
        }
        
    def register_variable(self, name: str, value: Any, read_only: bool = False):
        """Register a monitorable variable"""
        self.variables[name] = {
            "value": value,
            "read_only": read_only,
            "type": type(value).__name__
        }
        
    def get_parameter(self, name: str) -> Any:
        """Get current parameter value"""
        if name in self.parameters:
            return self.parameters[name]["value"]
        raise KeyError(f"Parameter '{name}' not found")
        
    def set_parameter(self, name: str, value: Any) -> bool:
        """Set parameter value with validation"""
        if name not in self.parameters:
            return False
            
        param = self.parameters[name]
        
        # Type validation
        expected_type = param["type"]
        if expected_type == "int" and not isinstance(value, int):
            try:
                value = int(value)
            except:
                return False
        elif expected_type == "float" and not isinstance(value, (int, float)):
            try:
                value = float(value)
            except:
                return False
        elif expected_type == "bool" and not isinstance(value, bool):
            value = str(value).lower() in ["true", "1", "yes"]
            
        # Value validation
        if param["valid_values"] is not None:
            if hasattr(param["valid_values"], "__contains__"):
                if value not in param["valid_values"]:
                    return False
                    
        param["value"] = value
        self.logger.info(f"Parameter '{name}' set to {value}")
        
        # Trigger callback if registered
        if name in self.callbacks:
            self.callbacks[name](value)
            
        return True
        
    def get_variable(self, name: str) -> Any:
        """Get current variable value"""
        if name in self.variables:
            return self.variables[name]["value"]
        raise KeyError(f"Variable '{name}' not found")
        
    def set_variable(self, name: str, value: Any) -> bool:
        """Set variable value if not read-only"""
        if name not in self.variables:
            return False
            
        if self.variables[name]["read_only"]:
            return False
            
        self.variables[name]["value"] = value
        return True
        
    def register_callback(self, parameter_name: str, callback: Callable):
        """Register callback for parameter changes"""
        self.callbacks[parameter_name] = callback
        
    def get_info(self) -> Dict:
        """Get script information for connector"""
        return {
            "name": self.script_name,
            "description": self.description,
            "state": self.state,
            "parameters": {
                name: {
                    "value": param["value"],
                    "type": param["type"],
                    "valid_values": param["valid_values"] if isinstance(param["valid_values"], list) else None
                }
                for name, param in self.parameters.items()
            },
            "variables": {
                name: {
                    "value": var["value"],
                    "type": var["type"],
                    "read_only": var["read_only"]
                }
                for name, var in self.variables.items()
            },
            "results": self.results
        }
        
    def send_to_connector(self, message: Dict) -> Optional[Dict]:
        """Send message to connector and wait for response"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', self.connector_port))
            
            sock.send(json.dumps(message).encode())
            response = sock.recv(4096).decode()
            sock.close()
            
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Failed to communicate with connector: {e}")
            return None
            
    def notify_connector(self, event: str, data: Dict = None):
        """Notify connector of an event"""
        message = {
            "command": "notification",
            "script": self.script_name,
            "event": event,
            "data": data or {}
        }
        self.send_to_connector(message)
        
    def _handle_messages(self):
        """Handle incoming messages from queue"""
        while True:
            try:
                message = self.message_queue.get(timeout=1)
                self._process_message(message)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error handling message: {e}")
                
    def _process_message(self, message: Dict):
        """Process control message"""
        command = message.get("command")
        
        if command == "get_info":
            return self.get_info()
        elif command == "set_parameter":
            param_name = message.get("parameter")
            value = message.get("value")
            success = self.set_parameter(param_name, value)
            return {"success": success}
        elif command == "get_parameter":
            param_name = message.get("parameter")
            try:
                value = self.get_parameter(param_name)
                return {"value": value}
            except KeyError:
                return {"error": "Parameter not found"}
        elif command == "set_variable":
            var_name = message.get("variable")
            value = message.get("value")
            success = self.set_variable(var_name, value)
            return {"success": success}
        elif command == "get_variable":
            var_name = message.get("variable")
            try:
                value = self.get_variable(var_name)
                return {"value": value}
            except KeyError:
                return {"error": "Variable not found"}
        elif command == "start":
            self.start()
            return {"success": True}
        elif command == "stop":
            self.stop()
            return {"success": True}
        elif command == "get_results":
            return {"results": self.results}
            
    def start(self):
        """Start the script execution"""
        if self.state != "running":
            self.state = "running"
            self.running = True
            self.notify_connector("started")
            
    def stop(self):
        """Stop the script execution"""
        if self.state == "running":
            self.running = False
            self.state = "stopped"
            self.notify_connector("stopped")
            
    def update_results(self, key: str, value: Any):
        """Update results that can be accessed by connector"""
        self.results[key] = value
        self.notify_connector("results_updated", {"key": key, "value": value})
        
    @abstractmethod
    def run(self):
        """Main execution method - must be implemented by each script"""
        pass
        
    def run_with_connector(self):
        """Run script with connector integration"""
        try:
            self.notify_connector("script_started", {"pid": os.getpid()})
            self.start()
            self.run()
        except Exception as e:
            self.logger.error(f"Script execution failed: {e}")
            self.notify_connector("error", {"error": str(e)})
        finally:
            self.stop()
            self.notify_connector("script_finished", {"results": self.results})


class ConnectorClient:
    """Client for scripts to communicate with the connector"""
    
    def __init__(self, script_interface: ScriptInterface):
        self.script = script_interface
        self.port = script_interface.connector_port
        
    def register_script(self):
        """Register script with connector"""
        message = {
            "command": "register_script",
            "script_name": self.script.script_name,
            "info": self.script.get_info()
        }
        return self.script.send_to_connector(message)
        
    def request_collaboration(self, target_script: str, data: Dict) -> Optional[Dict]:
        """Request collaboration with another script"""
        message = {
            "command": "collaborate",
            "source": self.script.script_name,
            "target": target_script,
            "data": data
        }
        return self.script.send_to_connector(message)
        
    def broadcast_data(self, data: Dict):
        """Broadcast data to all scripts"""
        message = {
            "command": "broadcast",
            "source": self.script.script_name,
            "data": data
        }
        self.script.send_to_connector(message)
        
    def get_script_list(self) -> List[str]:
        """Get list of available scripts"""
        response = self.script.send_to_connector({"command": "list_scripts"})
        if response:
            return response.get("scripts", [])
        return []


def create_standalone_wrapper(script_class):
    """Decorator to make scripts work both standalone and with connectors"""
    def wrapper(*args, **kwargs):
        # Check if running standalone or with connector
        if "--with-connector" in sys.argv:
            instance = script_class(*args, **kwargs)
            client = ConnectorClient(instance)
            client.register_script()
            instance.run_with_connector()
        else:
            # Run standalone
            instance = script_class(*args, **kwargs)
            instance.run()
    return wrapper