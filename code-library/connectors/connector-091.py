#!/usr/bin/env python3
"""
Universal Connector - Enhanced connector system for Polar Bear
Provides full control and integration for all scripts in the system
"""

import os
import sys
import json
import time
import logging
import threading
import importlib.util
import inspect
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import traceback

class UniversalConnector:
    """Universal connector that can control and integrate all scripts"""
    
    def __init__(self, logger=None):
        self.logger = logger or self._setup_logger()
        self.scripts = {}
        self.modules = {}
        self.shared_state = {}
        self.locks = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Discover and analyze all scripts
        self._discover_all_scripts()
        
    def _setup_logger(self):
        """Setup default logger"""
        logger = logging.getLogger("UniversalConnector")
        logger.setLevel(logging.DEBUG)
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _discover_all_scripts(self):
        """Discover and analyze all Python scripts"""
        self.logger.info("Discovering all scripts in directory...")
        
        # Get all Python files
        for script_path in Path('.').glob('*.py'):
            if script_path.name in ['universal_connector.py', '__init__.py']:
                continue
            
            script_name = script_path.stem
            self.logger.debug(f"Analyzing {script_name}...")
            
            # Analyze script structure
            script_info = self._analyze_script(script_path)
            self.scripts[script_name] = script_info
            
            # Create a lock for thread-safe access
            self.locks[script_name] = threading.Lock()
        
        self.logger.info(f"Discovered {len(self.scripts)} scripts")
    
    def _analyze_script(self, script_path: Path) -> Dict:
        """Deep analysis of a script's structure and capabilities"""
        info = {
            'path': str(script_path),
            'module': None,
            'functions': {},
            'classes': {},
            'variables': {},
            'imports': [],
            'has_main': False,
            'docstring': None,
            'dependencies': [],
            'error': None
        }
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            info['docstring'] = ast.get_docstring(tree)
            
            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        info['imports'].append(alias.name)
                        info['dependencies'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        info['imports'].append(node.module)
                        info['dependencies'].append(node.module)
                
                elif isinstance(node, ast.FunctionDef):
                    # Extract function info
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'defaults': len(node.args.defaults),
                        'docstring': ast.get_docstring(node),
                        'is_async': isinstance(node, ast.AsyncFunctionDef)
                    }
                    info['functions'][node.name] = func_info
                    
                    if node.name == 'main':
                        info['has_main'] = True
                
                elif isinstance(node, ast.ClassDef):
                    # Extract class info
                    class_info = {
                        'name': node.name,
                        'bases': [self._get_name(base) for base in node.bases],
                        'methods': {},
                        'docstring': ast.get_docstring(node)
                    }
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                'name': item.name,
                                'args': [arg.arg for arg in item.args.args],
                                'docstring': ast.get_docstring(item)
                            }
                            class_info['methods'][item.name] = method_info
                    
                    info['classes'][node.name] = class_info
                
                elif isinstance(node, ast.Assign):
                    # Extract module-level variables
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            try:
                                # Try to evaluate simple values
                                value = ast.literal_eval(node.value)
                                info['variables'][target.id] = {
                                    'value': value,
                                    'type': type(value).__name__
                                }
                            except:
                                info['variables'][target.id] = {
                                    'value': None,
                                    'type': 'complex'
                                }
            
            # Check for if __name__ == "__main__":
            info['has_main'] = 'if __name__ == "__main__":' in content
            
        except Exception as e:
            info['error'] = str(e)
            self.logger.error(f"Error analyzing {script_path}: {e}")
        
        return info
    
    def _get_name(self, node):
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def load_module(self, script_name: str) -> Any:
        """Dynamically load a script as a module"""
        if script_name not in self.scripts:
            raise ValueError(f"Script '{script_name}' not found")
        
        # Check if already loaded
        if script_name in self.modules:
            return self.modules[script_name]
        
        script_info = self.scripts[script_name]
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                script_name,
                script_info['path']
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                
                # Change to script's directory for proper imports
                original_dir = os.getcwd()
                script_dir = os.path.dirname(os.path.abspath(script_info['path']))
                os.chdir(script_dir)
                
                try:
                    spec.loader.exec_module(module)
                    self.modules[script_name] = module
                    script_info['module'] = module
                    self.logger.debug(f"Successfully loaded module: {script_name}")
                finally:
                    os.chdir(original_dir)
                
                return module
            
        except Exception as e:
            self.logger.error(f"Error loading module {script_name}: {e}")
            raise
    
    def get_function(self, script_name: str, function_name: str) -> Callable:
        """Get a function from a script"""
        module = self.load_module(script_name)
        
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            if callable(func):
                return func
        
        raise AttributeError(f"Function '{function_name}' not found in '{script_name}'")
    
    def get_class(self, script_name: str, class_name: str) -> type:
        """Get a class from a script"""
        module = self.load_module(script_name)
        
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            if inspect.isclass(cls):
                return cls
        
        raise AttributeError(f"Class '{class_name}' not found in '{script_name}'")
    
    def get_variable(self, script_name: str, var_name: str) -> Any:
        """Get a variable value from a script"""
        with self.locks[script_name]:
            # Check shared state first
            if script_name in self.shared_state and var_name in self.shared_state[script_name]:
                return self.shared_state[script_name][var_name]
            
            # Load from module
            module = self.load_module(script_name)
            if hasattr(module, var_name):
                return getattr(module, var_name)
            
            raise AttributeError(f"Variable '{var_name}' not found in '{script_name}'")
    
    def set_variable(self, script_name: str, var_name: str, value: Any) -> bool:
        """Set a variable value in a script"""
        with self.locks[script_name]:
            try:
                module = self.load_module(script_name)
                setattr(module, var_name, value)
                
                # Update shared state
                if script_name not in self.shared_state:
                    self.shared_state[script_name] = {}
                self.shared_state[script_name][var_name] = value
                
                self.logger.debug(f"Set {script_name}.{var_name} = {value}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error setting variable: {e}")
                return False
    
    def execute_function(self, script_name: str, function_name: str, 
                        *args, **kwargs) -> Any:
        """Execute a function from a script with full error handling"""
        try:
            func = self.get_function(script_name, function_name)
            
            # Log execution
            self.logger.info(f"Executing {script_name}.{function_name}()")
            
            # Execute with timeout protection
            result = func(*args, **kwargs)
            
            self.logger.debug(f"Function returned: {type(result).__name__}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing {script_name}.{function_name}: {e}")
            self.logger.debug(traceback.format_exc())
            raise
    
    def create_instance(self, script_name: str, class_name: str, 
                       *args, **kwargs) -> Any:
        """Create an instance of a class from a script"""
        try:
            cls = self.get_class(script_name, class_name)
            
            self.logger.info(f"Creating instance of {script_name}.{class_name}")
            instance = cls(*args, **kwargs)
            
            return instance
            
        except Exception as e:
            self.logger.error(f"Error creating instance: {e}")
            raise
    
    def call_method(self, instance: Any, method_name: str, 
                   *args, **kwargs) -> Any:
        """Call a method on an instance"""
        if hasattr(instance, method_name):
            method = getattr(instance, method_name)
            if callable(method):
                return method(*args, **kwargs)
        
        raise AttributeError(f"Method '{method_name}' not found")
    
    def get_script_info(self, script_name: str = None) -> Dict:
        """Get detailed information about scripts"""
        if script_name:
            return self.scripts.get(script_name, {})
        else:
            # Return summary of all scripts
            summary = {}
            for name, info in self.scripts.items():
                summary[name] = {
                    'functions': list(info['functions'].keys()),
                    'classes': list(info['classes'].keys()),
                    'variables': list(info['variables'].keys()),
                    'has_main': info['has_main'],
                    'imports': info['imports'][:5]  # First 5 imports
                }
            return summary
    
    def find_scripts_with_function(self, function_name: str) -> List[str]:
        """Find all scripts that have a specific function"""
        scripts = []
        for script_name, info in self.scripts.items():
            if function_name in info['functions']:
                scripts.append(script_name)
        return scripts
    
    def find_scripts_with_class(self, class_name: str) -> List[str]:
        """Find all scripts that have a specific class"""
        scripts = []
        for script_name, info in self.scripts.items():
            if class_name in info['classes']:
                scripts.append(script_name)
        return scripts
    
    def execute_pipeline(self, pipeline: List[Dict]) -> List[Any]:
        """Execute a pipeline of function calls"""
        results = []
        
        for step in pipeline:
            script = step.get('script')
            function = step.get('function')
            args = step.get('args', [])
            kwargs = step.get('kwargs', {})
            
            # Use previous result as input if specified
            if step.get('use_previous_result') and results:
                args = [results[-1]] + list(args)
            
            try:
                result = self.execute_function(script, function, *args, **kwargs)
                results.append(result)
                self.logger.info(f"Pipeline step completed: {script}.{function}")
            except Exception as e:
                self.logger.error(f"Pipeline step failed: {e}")
                if step.get('continue_on_error', False):
                    results.append(None)
                else:
                    raise
        
        return results
    
    def broadcast_command(self, function_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute the same function across all scripts that have it"""
        scripts = self.find_scripts_with_function(function_name)
        results = {}
        
        for script in scripts:
            try:
                result = self.execute_function(script, function_name, *args, **kwargs)
                results[script] = {'status': 'success', 'result': result}
            except Exception as e:
                results[script] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def get_shared_state(self) -> Dict:
        """Get the current shared state across all scripts"""
        return self.shared_state.copy()
    
    def save_state(self, filepath: str):
        """Save the current shared state to a file"""
        with open(filepath, 'w') as f:
            json.dump(self.shared_state, f, indent=2, default=str)
        self.logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load shared state from a file"""
        with open(filepath, 'r') as f:
            self.shared_state = json.load(f)
        self.logger.info(f"State loaded from {filepath}")
    
    def reset_state(self):
        """Reset all shared state"""
        self.shared_state.clear()
        self.modules.clear()
        self.logger.info("State reset completed")
    
    def shutdown(self):
        """Clean shutdown"""
        self.executor.shutdown(wait=True)
        self.logger.info("Universal Connector shutdown complete")

# Convenience functions for direct usage
_connector = None

def get_connector() -> UniversalConnector:
    """Get or create the global connector instance"""
    global _connector
    if _connector is None:
        _connector = UniversalConnector()
    return _connector

def execute(script: str, function: str, *args, **kwargs) -> Any:
    """Execute a function from any script"""
    return get_connector().execute_function(script, function, *args, **kwargs)

def get_var(script: str, variable: str) -> Any:
    """Get a variable from any script"""
    return get_connector().get_variable(script, variable)

def set_var(script: str, variable: str, value: Any) -> bool:
    """Set a variable in any script"""
    return get_connector().set_variable(script, variable, value)

def create(script: str, class_name: str, *args, **kwargs) -> Any:
    """Create an instance of a class from any script"""
    return get_connector().create_instance(script, class_name, *args, **kwargs)

def info(script: str = None) -> Dict:
    """Get information about scripts"""
    return get_connector().get_script_info(script)

# Example usage and testing
if __name__ == "__main__":
    # Create connector
    connector = UniversalConnector()
    
    # Show discovered scripts
    print("\n=== Discovered Scripts ===")
    all_info = connector.get_script_info()
    for script, details in all_info.items():
        print(f"\n{script}:")
        print(f"  Functions: {', '.join(details['functions'][:5])}")
        print(f"  Classes: {', '.join(details['classes'][:5])}")
        
    # Example: Find all scripts with 'detect' functions
    print("\n=== Scripts with Detection Functions ===")
    detect_scripts = connector.find_scripts_with_function('detect_defects')
    for script in detect_scripts:
        print(f"  - {script}")
    
    print("\nUniversal Connector ready!")