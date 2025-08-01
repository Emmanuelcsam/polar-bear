#!/usr/bin/env python3
"""
Script Interface for Connector Control
Provides a unified interface for connectors to control scripts
"""

import importlib.util
import inspect
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple
import threading
import queue

class ScriptInterface:
    """Interface for controlling scripts through connectors"""
    
    def __init__(self, script_path: str):
        self.script_path = Path(script_path)
        self.script_name = self.script_path.stem
        self.module = None
        self.functions = {}
        self.variables = {}
        self.parameters = {}
        self.original_values = {}
        self.execution_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.running = False
        self.thread = None
        
    def load_script(self) -> bool:
        """Load the script as a module"""
        try:
            spec = importlib.util.spec_from_file_location(self.script_name, self.script_path)
            self.module = importlib.util.module_from_spec(spec)
            sys.modules[self.script_name] = self.module
            spec.loader.exec_module(self.module)
            self._extract_components()
            return True
        except Exception as e:
            print(f"Error loading script {self.script_name}: {e}")
            return False
            
    def _extract_components(self):
        """Extract functions, variables, and parameters from the module"""
        for name, obj in inspect.getmembers(self.module):
            if name.startswith('_'):
                continue
                
            if inspect.isfunction(obj):
                self.functions[name] = obj
            elif inspect.isclass(obj):
                self.functions[name] = obj
            elif not inspect.ismodule(obj) and not inspect.isbuiltin(obj):
                self.variables[name] = obj
                self.original_values[name] = obj
                
    def get_info(self) -> Dict[str, Any]:
        """Get information about the script"""
        return {
            'script_name': self.script_name,
            'script_path': str(self.script_path),
            'functions': list(self.functions.keys()),
            'variables': {k: type(v).__name__ for k, v in self.variables.items()},
            'has_main': 'main' in self.functions,
            'docstring': self.module.__doc__ if self.module else None
        }
        
    def set_variable(self, name: str, value: Any) -> bool:
        """Set a variable in the script"""
        try:
            if hasattr(self.module, name):
                setattr(self.module, name, value)
                self.variables[name] = value
                return True
            return False
        except Exception as e:
            print(f"Error setting variable {name}: {e}")
            return False
            
    def get_variable(self, name: str) -> Any:
        """Get a variable from the script"""
        if hasattr(self.module, name):
            return getattr(self.module, name)
        return None
        
    def call_function(self, name: str, *args, **kwargs) -> Any:
        """Call a function in the script"""
        if name in self.functions:
            try:
                return self.functions[name](*args, **kwargs)
            except Exception as e:
                return {'error': str(e)}
        return {'error': f'Function {name} not found'}
        
    def run_main(self) -> Any:
        """Run the main function if it exists"""
        if 'main' in self.functions:
            return self.call_function('main')
        elif hasattr(self.module, '__name__'):
            # Try to execute the script's main block
            try:
                exec(compile(open(self.script_path).read(), self.script_path, 'exec'), 
                     {'__name__': '__main__', '__file__': str(self.script_path)})
                return {'status': 'executed'}
            except Exception as e:
                return {'error': str(e)}
        return {'error': 'No main function found'}
        
    def start_background_execution(self):
        """Start background execution thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._execution_loop)
            self.thread.daemon = True
            self.thread.start()
            
    def stop_background_execution(self):
        """Stop background execution thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            
    def _execution_loop(self):
        """Background execution loop"""
        while self.running:
            try:
                task = self.execution_queue.get(timeout=1)
                if task:
                    result = self._execute_task(task)
                    self.results_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                self.results_queue.put({'error': str(e)})
                
    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task"""
        task_type = task.get('type')
        
        if task_type == 'function':
            return {
                'task_id': task.get('id'),
                'result': self.call_function(
                    task.get('name'),
                    *task.get('args', []),
                    **task.get('kwargs', {})
                )
            }
        elif task_type == 'variable':
            if task.get('action') == 'set':
                return {
                    'task_id': task.get('id'),
                    'success': self.set_variable(task.get('name'), task.get('value'))
                }
            else:
                return {
                    'task_id': task.get('id'),
                    'value': self.get_variable(task.get('name'))
                }
        return {'error': 'Unknown task type'}
        
    def reset_variables(self):
        """Reset all variables to their original values"""
        for name, value in self.original_values.items():
            self.set_variable(name, value)
            
    def get_function_signature(self, name: str) -> Dict[str, Any]:
        """Get function signature information"""
        if name in self.functions:
            sig = inspect.signature(self.functions[name])
            return {
                'name': name,
                'parameters': {
                    param.name: {
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None
                    }
                    for param in sig.parameters.values()
                },
                'docstring': self.functions[name].__doc__
            }
        return {}


class ScriptManager:
    """Manages multiple script interfaces"""
    
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.scripts = {}
        self.interfaces = {}
        
    def discover_scripts(self) -> List[str]:
        """Discover all Python scripts in the directory"""
        scripts = []
        for file in self.directory.glob('*.py'):
            if file.name not in ['connector.py', 'hivemind_connector.py', 'script_interface.py', '__init__.py']:
                scripts.append(file.name)
                self.scripts[file.stem] = str(file)
        return scripts
        
    def load_script(self, script_name: str) -> bool:
        """Load a specific script"""
        if script_name in self.scripts:
            interface = ScriptInterface(self.scripts[script_name])
            if interface.load_script():
                self.interfaces[script_name] = interface
                return True
        return False
        
    def load_all_scripts(self) -> Dict[str, bool]:
        """Load all discovered scripts"""
        results = {}
        for script_name in self.scripts:
            results[script_name] = self.load_script(script_name)
        return results
        
    def get_script_info(self, script_name: str) -> Dict[str, Any]:
        """Get information about a specific script"""
        if script_name in self.interfaces:
            return self.interfaces[script_name].get_info()
        return {}
        
    def get_all_scripts_info(self) -> Dict[str, Any]:
        """Get information about all loaded scripts"""
        return {
            name: interface.get_info()
            for name, interface in self.interfaces.items()
        }
        
    def execute_on_script(self, script_name: str, action: Dict[str, Any]) -> Any:
        """Execute an action on a specific script"""
        if script_name not in self.interfaces:
            return {'error': f'Script {script_name} not loaded'}
            
        interface = self.interfaces[script_name]
        action_type = action.get('type')
        
        if action_type == 'call_function':
            return interface.call_function(
                action.get('function_name'),
                *action.get('args', []),
                **action.get('kwargs', {})
            )
        elif action_type == 'set_variable':
            success = interface.set_variable(action.get('variable_name'), action.get('value'))
            return {'success': success}
        elif action_type == 'get_variable':
            value = interface.get_variable(action.get('variable_name'))
            return {'value': value}
        elif action_type == 'run_main':
            return interface.run_main()
        elif action_type == 'get_function_signature':
            return interface.get_function_signature(action.get('function_name'))
        else:
            return {'error': 'Unknown action type'}
            
    def broadcast_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast an action to all scripts"""
        results = {}
        for script_name in self.interfaces:
            results[script_name] = self.execute_on_script(script_name, action)
        return results
        
    def orchestrate_collaboration(self, workflow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Orchestrate collaboration between scripts"""
        results = []
        context = {}
        
        for step in workflow:
            script_name = step.get('script')
            action = step.get('action')
            
            # Replace context variables in action
            if 'args' in action:
                action['args'] = self._replace_context(action['args'], context)
            if 'kwargs' in action:
                action['kwargs'] = self._replace_context(action['kwargs'], context)
                
            result = self.execute_on_script(script_name, action)
            
            # Store result in context if specified
            if 'store_as' in step:
                context[step['store_as']] = result
                
            results.append({
                'step': step.get('name', 'unnamed'),
                'script': script_name,
                'result': result
            })
            
        return results
        
    def _replace_context(self, data: Any, context: Dict[str, Any]) -> Any:
        """Replace context variables in data"""
        if isinstance(data, str) and data.startswith('$'):
            var_name = data[1:]
            return context.get(var_name, data)
        elif isinstance(data, list):
            return [self._replace_context(item, context) for item in data]
        elif isinstance(data, dict):
            return {k: self._replace_context(v, context) for k, v in data.items()}
        return data


if __name__ == "__main__":
    # Demo usage
    manager = ScriptManager(os.getcwd())
    scripts = manager.discover_scripts()
    print(f"Discovered scripts: {scripts}")
    
    # Load all scripts
    load_results = manager.load_all_scripts()
    print(f"Load results: {load_results}")
    
    # Get info about all scripts
    all_info = manager.get_all_scripts_info()
    print(json.dumps(all_info, indent=2))