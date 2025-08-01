#!/usr/bin/env python3
"""
Script Interface for connector integration
Provides a unified interface for controlling all scripts in the directory
"""

import importlib.util
import sys
import os
import threading
import queue
import json
from typing import Any, Dict, Optional, Callable
import inspect

class ScriptInterface:
    """Interface for managing and controlling scripts"""
    
    def __init__(self):
        self.scripts = {}
        self.script_modules = {}
        self.script_states = {}
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def load_script(self, script_name: str, script_path: str) -> bool:
        """Load a script dynamically"""
        try:
            spec = importlib.util.spec_from_file_location(script_name, script_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[script_name] = module
            spec.loader.exec_module(module)
            
            self.script_modules[script_name] = module
            self.scripts[script_name] = {
                'path': script_path,
                'module': module,
                'functions': self._extract_functions(module),
                'variables': self._extract_variables(module),
                'state': 'loaded'
            }
            
            return True
        except Exception as e:
            print(f"Error loading script {script_name}: {e}")
            return False
            
    def _extract_functions(self, module) -> Dict[str, Callable]:
        """Extract all callable functions from a module"""
        functions = {}
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                functions[name] = obj
        return functions
        
    def _extract_variables(self, module) -> Dict[str, Any]:
        """Extract all variables from a module"""
        variables = {}
        for name, obj in inspect.getmembers(module):
            if not name.startswith('_') and not inspect.isfunction(obj) and not inspect.isclass(obj) and not inspect.ismodule(obj):
                variables[name] = obj
        return variables
        
    def execute_function(self, script_name: str, function_name: str, *args, **kwargs) -> Any:
        """Execute a function in a script with given parameters"""
        if script_name not in self.scripts:
            raise ValueError(f"Script {script_name} not loaded")
            
        functions = self.scripts[script_name]['functions']
        if function_name not in functions:
            raise ValueError(f"Function {function_name} not found in {script_name}")
            
        return functions[function_name](*args, **kwargs)
        
    def get_variable(self, script_name: str, variable_name: str) -> Any:
        """Get a variable value from a script"""
        if script_name not in self.scripts:
            raise ValueError(f"Script {script_name} not loaded")
            
        module = self.scripts[script_name]['module']
        if hasattr(module, variable_name):
            return getattr(module, variable_name)
        else:
            raise ValueError(f"Variable {variable_name} not found in {script_name}")
            
    def set_variable(self, script_name: str, variable_name: str, value: Any) -> bool:
        """Set a variable value in a script"""
        if script_name not in self.scripts:
            raise ValueError(f"Script {script_name} not loaded")
            
        module = self.scripts[script_name]['module']
        setattr(module, variable_name, value)
        return True
        
    def get_script_info(self, script_name: str) -> Dict[str, Any]:
        """Get information about a loaded script"""
        if script_name not in self.scripts:
            return None
            
        script = self.scripts[script_name]
        return {
            'path': script['path'],
            'functions': list(script['functions'].keys()),
            'variables': list(self._extract_variables(script['module']).keys()),
            'state': script['state']
        }
        
    def list_scripts(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded scripts with their info"""
        result = {}
        for name in self.scripts:
            result[name] = self.get_script_info(name)
        return result
        
    def control_script(self, script_name: str, command: str, params: Dict[str, Any] = None) -> Any:
        """Send control commands to scripts"""
        if script_name not in self.scripts:
            raise ValueError(f"Script {script_name} not loaded")
            
        if command == 'execute':
            func_name = params.get('function')
            args = params.get('args', [])
            kwargs = params.get('kwargs', {})
            return self.execute_function(script_name, func_name, *args, **kwargs)
            
        elif command == 'get_var':
            var_name = params.get('variable')
            return self.get_variable(script_name, var_name)
            
        elif command == 'set_var':
            var_name = params.get('variable')
            value = params.get('value')
            return self.set_variable(script_name, var_name, value)
            
        elif command == 'get_info':
            return self.get_script_info(script_name)
            
        else:
            raise ValueError(f"Unknown command: {command}")
            
    def process_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a command from the connector"""
        try:
            script_name = command_data.get('script')
            command = command_data.get('command')
            params = command_data.get('params', {})
            
            result = self.control_script(script_name, command, params)
            
            return {
                'status': 'success',
                'result': result
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

# Singleton instance
_interface = ScriptInterface()

def get_interface():
    """Get the singleton script interface instance"""
    return _interface