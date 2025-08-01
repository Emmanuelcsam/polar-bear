#!/usr/bin/env python3
"""
Script Interface for PyTorch Production Module
Provides a unified interface for connectors to control all scripts
"""

import os
import sys
import json
import importlib.util
import inspect
import torch
import numpy as np
from pathlib import Path
import logging
import traceback
from typing import Any, Dict, List, Optional, Callable

class ScriptInterface:
    """Unified interface for script control and parameter management"""
    
    def __init__(self):
        self.scripts = {}
        self.script_configs = {}
        self.shared_state = {
            'model': None,
            'optimizer': None,
            'target_data': None,
            'img_size': 128,
            'learning_rate': 0.1,
            'model_filename': 'generator_model.pth',
            'output_filename': 'final_generated_image.png',
            'target_filename': 'target_data.pt'
        }
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging for the interface"""
        logger = logging.getLogger('ScriptInterface')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def register_scripts(self):
        """Register all available scripts with their configurations"""
        self.script_configs = {
            'preprocess.py': {
                'description': 'Initialize PyTorch generator model',
                'parameters': {
                    'img_size': {'type': 'int', 'default': 128, 'min': 32, 'max': 512},
                    'model_filename': {'type': 'str', 'default': 'generator_model.pth'}
                },
                'functions': ['initialize_generator'],
                'independent': True
            },
            'load.py': {
                'description': 'Load and prepare image data',
                'parameters': {
                    'img_size': {'type': 'int', 'default': 128, 'min': 32, 'max': 512},
                    'output_file': {'type': 'str', 'default': 'target_data.pt'}
                },
                'functions': ['load_and_prepare_data'],
                'independent': True,
                'interactive': True
            },
            'train.py': {
                'description': 'Train the PyTorch model',
                'parameters': {
                    'img_size': {'type': 'int', 'default': 128, 'min': 32, 'max': 512},
                    'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.0001, 'max': 1.0},
                    'model_filename': {'type': 'str', 'default': 'generator_model.pth'},
                    'target_filename': {'type': 'str', 'default': 'target_data.pt'}
                },
                'functions': ['train_model'],
                'independent': True,
                'requires': ['generator_model.pth', 'target_data.pt']
            },
            'final.py': {
                'description': 'Generate final image from trained model',
                'parameters': {
                    'img_size': {'type': 'int', 'default': 128, 'min': 32, 'max': 512},
                    'model_filename': {'type': 'str', 'default': 'generator_model.pth'},
                    'output_filename': {'type': 'str', 'default': 'final_generated_image.png'}
                },
                'functions': ['main', 'generate_final_image'],
                'independent': True,
                'requires': ['generator_model.pth']
            }
        }
        
    def load_script_module(self, script_name: str):
        """Dynamically load a script module"""
        try:
            script_path = Path(__file__).parent / script_name
            if not script_path.exists():
                return None
                
            spec = importlib.util.spec_from_file_location(
                script_name.replace('.py', ''),
                script_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self.scripts[script_name] = module
            self.logger.info(f"Loaded script: {script_name}")
            return module
            
        except Exception as e:
            self.logger.error(f"Failed to load {script_name}: {str(e)}")
            return None
            
    def get_script_info(self, script_name: str) -> Dict[str, Any]:
        """Get information about a script"""
        if script_name not in self.script_configs:
            return {'error': 'Script not found'}
            
        config = self.script_configs[script_name]
        module = self.scripts.get(script_name)
        
        info = {
            'name': script_name,
            'description': config['description'],
            'parameters': config['parameters'],
            'functions': config['functions'],
            'loaded': module is not None,
            'independent': config.get('independent', True),
            'requires': config.get('requires', [])
        }
        
        if module:
            # Get actual functions from module
            available_functions = [
                name for name, obj in inspect.getmembers(module)
                if inspect.isfunction(obj) and not name.startswith('_')
            ]
            info['available_functions'] = available_functions
            
        return info
        
    def set_parameter(self, script_name: str, param_name: str, value: Any) -> Dict[str, Any]:
        """Set a parameter value for a script"""
        if script_name not in self.script_configs:
            return {'error': 'Script not found'}
            
        params = self.script_configs[script_name]['parameters']
        if param_name not in params:
            return {'error': f'Parameter {param_name} not found'}
            
        param_config = params[param_name]
        
        # Validate parameter type
        if param_config['type'] == 'int':
            try:
                value = int(value)
                if 'min' in param_config and value < param_config['min']:
                    return {'error': f'Value below minimum: {param_config["min"]}'}
                if 'max' in param_config and value > param_config['max']:
                    return {'error': f'Value above maximum: {param_config["max"]}'}
            except ValueError:
                return {'error': 'Invalid integer value'}
                
        elif param_config['type'] == 'float':
            try:
                value = float(value)
                if 'min' in param_config and value < param_config['min']:
                    return {'error': f'Value below minimum: {param_config["min"]}'}
                if 'max' in param_config and value > param_config['max']:
                    return {'error': f'Value above maximum: {param_config["max"]}'}
            except ValueError:
                return {'error': 'Invalid float value'}
                
        # Update shared state
        self.shared_state[param_name] = value
        
        # Update in loaded module if applicable
        if script_name in self.scripts:
            module = self.scripts[script_name]
            if hasattr(module, param_name.upper()):
                setattr(module, param_name.upper(), value)
                
        return {
            'status': 'success',
            'parameter': param_name,
            'value': value,
            'script': script_name
        }
        
    def execute_function(self, script_name: str, function_name: str, 
                        args: List[Any] = None, kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a specific function from a script"""
        if script_name not in self.script_configs:
            return {'error': 'Script not found'}
            
        # Load module if not already loaded
        if script_name not in self.scripts:
            if not self.load_script_module(script_name):
                return {'error': 'Failed to load script'}
                
        module = self.scripts[script_name]
        
        if not hasattr(module, function_name):
            return {'error': f'Function {function_name} not found in {script_name}'}
            
        func = getattr(module, function_name)
        
        try:
            # Apply shared state parameters to module
            for param, value in self.shared_state.items():
                if hasattr(module, param.upper()):
                    setattr(module, param.upper(), value)
                    
            # Execute function
            args = args or []
            kwargs = kwargs or {}
            
            result = func(*args, **kwargs)
            
            return {
                'status': 'success',
                'script': script_name,
                'function': function_name,
                'result': str(result) if result is not None else 'Function executed successfully'
            }
            
        except Exception as e:
            return {
                'error': f'Execution failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
            
    def get_shared_state(self) -> Dict[str, Any]:
        """Get current shared state"""
        return self.shared_state.copy()
        
    def update_shared_state(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update shared state with multiple values"""
        self.shared_state.update(updates)
        return {'status': 'success', 'state': self.shared_state}
        
    def check_dependencies(self, script_name: str) -> Dict[str, Any]:
        """Check if script dependencies are met"""
        if script_name not in self.script_configs:
            return {'error': 'Script not found'}
            
        requires = self.script_configs[script_name].get('requires', [])
        missing = []
        
        for req in requires:
            if not Path(req).exists():
                missing.append(req)
                
        return {
            'script': script_name,
            'requires': requires,
            'missing': missing,
            'ready': len(missing) == 0
        }
        
    def list_scripts(self) -> List[Dict[str, Any]]:
        """List all available scripts with their status"""
        scripts = []
        for name, config in self.script_configs.items():
            info = self.get_script_info(name)
            deps = self.check_dependencies(name)
            info['ready'] = deps['ready']
            info['missing_deps'] = deps.get('missing', [])
            scripts.append(info)
        return scripts


# Singleton instance
interface = ScriptInterface()
interface.register_scripts()


def handle_connector_command(command: Dict[str, Any]) -> Dict[str, Any]:
    """Handle commands from connectors"""
    cmd = command.get('command')
    
    if cmd == 'list_scripts':
        return {'scripts': interface.list_scripts()}
        
    elif cmd == 'get_script_info':
        script = command.get('script')
        return interface.get_script_info(script)
        
    elif cmd == 'set_parameter':
        script = command.get('script')
        param = command.get('parameter')
        value = command.get('value')
        return interface.set_parameter(script, param, value)
        
    elif cmd == 'execute_function':
        script = command.get('script')
        function = command.get('function')
        args = command.get('args', [])
        kwargs = command.get('kwargs', {})
        return interface.execute_function(script, function, args, kwargs)
        
    elif cmd == 'get_state':
        return {'state': interface.get_shared_state()}
        
    elif cmd == 'update_state':
        updates = command.get('updates', {})
        return interface.update_shared_state(updates)
        
    elif cmd == 'check_dependencies':
        script = command.get('script')
        return interface.check_dependencies(script)
        
    else:
        return {'error': 'Unknown command'}
        

if __name__ == "__main__":
    # Test interface
    print("Script Interface initialized")
    print("\nAvailable scripts:")
    for script in interface.list_scripts():
        print(f"- {script['name']}: {script['description']}")