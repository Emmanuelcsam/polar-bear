import os
import logging
import sys
import json
import subprocess
import threading
import queue
import time
from pathlib import Path
import importlib.util
import inspect
import ast
import io
from contextlib import redirect_stdout, redirect_stderr

# --- Configuration ---
LOG_FILE = "connector.log"

# --- Setup Logging ---
logger = logging.getLogger(os.path.abspath(__file__))
logger.setLevel(logging.INFO)
logger.propagate = False

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except (IOError, OSError) as e:
    print(f"Could not write to log file {LOG_FILE}: {e}", file=sys.stderr)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


class ScriptController:
    """Controls and manages individual scripts"""
    
    def __init__(self, script_path):
        self.script_path = Path(script_path)
        self.script_name = self.script_path.name
        self.module = None
        self.script_globals = {}
        self.script_locals = {}
        self.is_loaded = False
        self.execution_thread = None
        self.output_queue = queue.Queue()
        
    def load_script(self):
        """Load script as a module"""
        try:
            with open(self.script_path, 'r') as f:
                self.script_content = f.read()
            
            # Parse the script to understand its structure
            self.ast_tree = ast.parse(self.script_content)
            
            # Create a custom namespace for the script
            self.script_globals = {
                '__name__': '__main__',
                '__file__': str(self.script_path),
                '__builtins__': __builtins__,
                '_connector_control': self
            }
            
            self.is_loaded = True
            logger.info(f"Loaded script: {self.script_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load script {self.script_name}: {e}")
            return False
    
    def get_variables(self):
        """Get all variables defined in the script"""
        variables = {}
        for key, value in self.script_globals.items():
            if not key.startswith('__') and key != '_connector_control':
                try:
                    # Convert to JSON serializable format
                    if isinstance(value, (int, float, str, bool, list, dict)):
                        variables[key] = value
                    else:
                        variables[key] = str(value)
                except:
                    variables[key] = str(type(value))
        return variables
    
    def set_variable(self, var_name, value):
        """Set a variable in the script's namespace"""
        try:
            self.script_globals[var_name] = value
            logger.info(f"Set {var_name} = {value} in {self.script_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to set variable {var_name}: {e}")
            return False
    
    def get_functions(self):
        """Get all functions defined in the script"""
        functions = []
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'lineno': node.lineno
                }
                functions.append(func_info)
        return functions
    
    def execute_function(self, func_name, *args, **kwargs):
        """Execute a specific function from the script"""
        try:
            # First execute the script to define functions
            exec(self.script_content, self.script_globals)
            
            if func_name in self.script_globals and callable(self.script_globals[func_name]):
                result = self.script_globals[func_name](*args, **kwargs)
                return {'success': True, 'result': result}
            else:
                return {'success': False, 'error': f"Function {func_name} not found"}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def execute_script(self, input_data=None):
        """Execute the entire script"""
        try:
            # Capture output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            # Mock input if needed
            if input_data:
                input_lines = input_data.strip().split('\n') if isinstance(input_data, str) else []
                input_iter = iter(input_lines)
                self.script_globals['input'] = lambda: next(input_iter)
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(self.script_content, self.script_globals)
            
            return {
                'success': True,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue(),
                'variables': self.get_variables()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stdout': stdout_capture.getvalue() if 'stdout_capture' in locals() else '',
                'stderr': stderr_capture.getvalue() if 'stderr_capture' in locals() else ''
            }
    
    def modify_script(self, modifications):
        """Modify the script content"""
        try:
            # Apply modifications to AST or direct text replacement
            if 'replace' in modifications:
                for old, new in modifications['replace'].items():
                    self.script_content = self.script_content.replace(old, new)
            
            # Re-parse after modifications
            self.ast_tree = ast.parse(self.script_content)
            
            # Save back to file
            with open(self.script_path, 'w') as f:
                f.write(self.script_content)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to modify script: {e}")
            return False


class ConnectorSystem:
    """Main connector system that manages all scripts"""
    
    def __init__(self):
        self.directory = Path(__file__).parent
        self.scripts = {}
        self.script_controllers = {}
        self.hivemind_connector = None
        self.collaboration_queue = queue.Queue()
        
    def scan_scripts(self):
        """Scan directory for Python scripts"""
        logger.info("Scanning directory for scripts...")
        
        for file_path in self.directory.glob("*.py"):
            if file_path.name not in ['connector.py', 'hivemind_connector.py', 'setup.py']:
                if file_path.name.endswith('.py'):
                    self.scripts[file_path.stem] = file_path
                    controller = ScriptController(file_path)
                    if controller.load_script():
                        self.script_controllers[file_path.stem] = controller
        
        logger.info(f"Found {len(self.scripts)} scripts: {list(self.scripts.keys())}")
        
    def get_script_info(self, script_name):
        """Get detailed information about a script"""
        if script_name in self.script_controllers:
            controller = self.script_controllers[script_name]
            return {
                'name': script_name,
                'path': str(controller.script_path),
                'variables': controller.get_variables(),
                'functions': controller.get_functions(),
                'loaded': controller.is_loaded
            }
        return None
    
    def control_script(self, script_name, action, params=None):
        """Control a script with various actions"""
        if script_name not in self.script_controllers:
            return {'error': f"Script {script_name} not found"}
        
        controller = self.script_controllers[script_name]
        
        if action == 'execute':
            return controller.execute_script(params.get('input_data') if params else None)
        
        elif action == 'set_variable':
            if params and 'variable' in params and 'value' in params:
                success = controller.set_variable(params['variable'], params['value'])
                return {'success': success}
        
        elif action == 'get_variables':
            return {'variables': controller.get_variables()}
        
        elif action == 'execute_function':
            if params and 'function' in params:
                return controller.execute_function(
                    params['function'],
                    *params.get('args', []),
                    **params.get('kwargs', {})
                )
        
        elif action == 'modify':
            if params and 'modifications' in params:
                success = controller.modify_script(params['modifications'])
                return {'success': success}
        
        else:
            return {'error': f"Unknown action: {action}"}
    
    def enable_collaboration(self):
        """Enable collaboration between scripts"""
        logger.info("Enabling script collaboration...")
        
        # Create shared memory space for scripts
        self.shared_memory = {
            'data': {},
            'messages': queue.Queue(),
            'locks': {}
        }
        
        # Inject collaboration functions into all scripts
        for name, controller in self.script_controllers.items():
            controller.script_globals['_shared_memory'] = self.shared_memory
            controller.script_globals['_get_shared'] = lambda key: self.shared_memory['data'].get(key)
            controller.script_globals['_set_shared'] = lambda key, value: self.shared_memory['data'].update({key: value})
            controller.script_globals['_send_message'] = lambda target, msg: self.send_message(name, target, msg)
            controller.script_globals['_receive_messages'] = lambda: self.receive_messages(name)
    
    def send_message(self, sender, target, message):
        """Send message between scripts"""
        self.shared_memory['messages'].put({
            'sender': sender,
            'target': target,
            'message': message,
            'timestamp': time.time()
        })
    
    def receive_messages(self, recipient):
        """Receive messages for a script"""
        messages = []
        temp_queue = queue.Queue()
        
        # Get all messages
        while not self.shared_memory['messages'].empty():
            msg = self.shared_memory['messages'].get()
            if msg['target'] == recipient or msg['target'] == 'all':
                messages.append(msg)
            else:
                temp_queue.put(msg)
        
        # Put back non-matching messages
        while not temp_queue.empty():
            self.shared_memory['messages'].put(temp_queue.get())
        
        return messages
    
    def execute_collaborative_task(self, task_definition):
        """Execute a task that requires multiple scripts to collaborate"""
        logger.info(f"Executing collaborative task: {task_definition.get('name', 'unnamed')}")
        
        results = {}
        
        # Execute scripts in defined order with data passing
        for step in task_definition.get('steps', []):
            script_name = step.get('script')
            action = step.get('action', 'execute')
            params = step.get('params', {})
            
            # Add previous results to params if needed
            if step.get('use_previous_result'):
                params['input_data'] = results.get('previous_output', '')
            
            result = self.control_script(script_name, action, params)
            results[f"{script_name}_{action}"] = result
            
            if result.get('success') and 'stdout' in result:
                results['previous_output'] = result['stdout']
        
        return results
    
    def get_system_status(self):
        """Get full system status"""
        return {
            'directory': str(self.directory),
            'scripts_loaded': len(self.script_controllers),
            'scripts': {
                name: {
                    'loaded': controller.is_loaded,
                    'variables': len(controller.get_variables()),
                    'functions': len(controller.get_functions())
                }
                for name, controller in self.script_controllers.items()
            },
            'collaboration_enabled': hasattr(self, 'shared_memory'),
            'shared_data': self.shared_memory.get('data', {}) if hasattr(self, 'shared_memory') else {}
        }


def main():
    """Main function for the enhanced connector script"""
    logger.info("--- Enhanced Connector Script Initialized ---")
    
    # Create connector system
    connector = ConnectorSystem()
    
    # Scan for scripts
    connector.scan_scripts()
    
    # Enable collaboration
    connector.enable_collaboration()
    
    # Display system status
    status = connector.get_system_status()
    logger.info(f"System Status: {json.dumps(status, indent=2)}")
    
    # Example: Control scripts
    logger.info("\n--- Script Control Examples ---")
    
    # Get info about correlation-finder
    if 'correlation-finder' in connector.script_controllers:
        info = connector.get_script_info('correlation-finder')
        logger.info(f"Correlation Finder Info: {json.dumps(info, indent=2)}")
    
    # Example collaborative task
    collaborative_task = {
        'name': 'analyze_data_flow',
        'steps': [
            {
                'script': 'intensity-matcher',
                'action': 'set_variable',
                'params': {'variable': 'target_value', 'value': 128}
            }
        ]
    }
    
    logger.info("\n--- Executing Collaborative Task ---")
    task_result = connector.execute_collaborative_task(collaborative_task)
    logger.info(f"Task Result: {json.dumps(task_result, indent=2)}")
    
    logger.info("\nConnector ready for external control via hivemind_connector.py")
    
    # Return the connector instance for programmatic use
    return connector


if __name__ == "__main__":
    main()