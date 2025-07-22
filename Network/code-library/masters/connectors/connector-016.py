#!/usr/bin/env python3
"""
Enhanced Connector - Central orchestration and control system
Provides full control over all scripts while maintaining their independence
"""

import os
import sys
import json
import logging
import importlib.util
import inspect
import threading
import queue
import time
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path
import subprocess
import pickle

# --- Configuration ---
LOG_FILE = "connector.log"
STATE_FILE = "connector_state.pkl"
CONFIG_FILE = "connector_config.json"

# --- Setup Logging ---
logger = logging.getLogger("EnhancedConnector")
logger.setLevel(logging.INFO)
logger.propagate = False

# Remove existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

# File Handler
try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except (IOError, OSError) as e:
    print(f"Could not write to log file {LOG_FILE}: {e}", file=sys.stderr)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


class ScriptInterface:
    """Interface for interacting with individual scripts"""
    
    def __init__(self, script_path: str, connector):
        self.script_path = script_path
        self.script_name = os.path.basename(script_path)
        self.connector = connector
        self.module = None
        self.attributes = {}
        self.functions = {}
        self.classes = {}
        self.state = {}
        self.running = False
        self.thread = None
        
    def load(self):
        """Load the script as a module"""
        try:
            spec = importlib.util.spec_from_file_location(
                self.script_name.replace('.py', ''), 
                self.script_path
            )
            self.module = importlib.util.module_from_spec(spec)
            
            # Inject connector reference
            self.module._connector = self.connector
            self.module._script_interface = self
            
            # Load the module
            spec.loader.exec_module(self.module)
            
            # Catalog module contents
            self._catalog_module()
            
            logger.info(f"Loaded script: {self.script_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {self.script_name}: {e}")
            return False
    
    def _catalog_module(self):
        """Catalog all accessible elements in the module"""
        for name, obj in inspect.getmembers(self.module):
            if name.startswith('_'):
                continue
                
            if inspect.isfunction(obj):
                self.functions[name] = obj
            elif inspect.isclass(obj):
                self.classes[name] = obj
            else:
                self.attributes[name] = obj
    
    def get_variable(self, name: str) -> Any:
        """Get a variable from the script"""
        if hasattr(self.module, name):
            return getattr(self.module, name)
        return None
    
    def set_variable(self, name: str, value: Any):
        """Set a variable in the script"""
        setattr(self.module, name, value)
        self.state[name] = value
        logger.info(f"Set {self.script_name}.{name} = {value}")
    
    def call_function(self, name: str, *args, **kwargs) -> Any:
        """Call a function in the script"""
        if name in self.functions:
            try:
                result = self.functions[name](*args, **kwargs)
                logger.info(f"Called {self.script_name}.{name}()")
                return result
            except Exception as e:
                logger.error(f"Error calling {self.script_name}.{name}: {e}")
                raise
        else:
            raise AttributeError(f"Function {name} not found in {self.script_name}")
    
    def run_independent(self) -> subprocess.CompletedProcess:
        """Run the script independently as a subprocess"""
        try:
            result = subprocess.run(
                [sys.executable, self.script_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            logger.info(f"Ran {self.script_name} independently: return code {result.returncode}")
            return result
        except Exception as e:
            logger.error(f"Error running {self.script_name} independently: {e}")
            raise
    
    def run_collaborative(self):
        """Run the script's main function in collaborative mode"""
        if 'main' in self.functions:
            self.running = True
            self.thread = threading.Thread(
                target=self._run_main,
                name=f"Script_{self.script_name}"
            )
            self.thread.start()
            logger.info(f"Started {self.script_name} in collaborative mode")
    
    def _run_main(self):
        """Internal method to run main function"""
        try:
            self.functions['main']()
        except Exception as e:
            logger.error(f"Error in {self.script_name} main: {e}")
        finally:
            self.running = False
    
    def stop(self):
        """Stop the script if running"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            logger.info(f"Stopped {self.script_name}")


class EnhancedConnector:
    """Enhanced connector with full control over all scripts"""
    
    def __init__(self):
        self.scripts: Dict[str, ScriptInterface] = {}
        self.shared_state = {}
        self.message_queue = queue.Queue()
        self.event_handlers = {}
        self.running = False
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "auto_discover": True,
            "script_timeout": 60,
            "enable_collaboration": True
        }
    
    def discover_scripts(self):
        """Discover all Python scripts in the directory"""
        script_dir = Path('.')
        excluded = {'connector.py', 'hivemind_connector.py', '__pycache__'}
        
        for script_path in script_dir.glob('*.py'):
            if script_path.name not in excluded:
                self.add_script(str(script_path))
    
    def add_script(self, script_path: str):
        """Add a script to the connector"""
        interface = ScriptInterface(script_path, self)
        if interface.load():
            self.scripts[interface.script_name] = interface
            logger.info(f"Added script: {interface.script_name}")
    
    def get_script(self, script_name: str) -> Optional[ScriptInterface]:
        """Get a script interface by name"""
        return self.scripts.get(script_name)
    
    def set_parameter(self, script_name: str, param_name: str, value: Any):
        """Set a parameter in a specific script"""
        script = self.get_script(script_name)
        if script:
            script.set_variable(param_name, value)
        else:
            logger.error(f"Script {script_name} not found")
    
    def get_parameter(self, script_name: str, param_name: str) -> Any:
        """Get a parameter from a specific script"""
        script = self.get_script(script_name)
        if script:
            return script.get_variable(param_name)
        return None
    
    def call_script_function(self, script_name: str, function_name: str, *args, **kwargs) -> Any:
        """Call a function in a specific script"""
        script = self.get_script(script_name)
        if script:
            return script.call_function(function_name, *args, **kwargs)
        else:
            logger.error(f"Script {script_name} not found")
            return None
    
    def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all scripts"""
        self.message_queue.put(message)
        logger.info(f"Broadcasted message: {message.get('type', 'unknown')}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def trigger_event(self, event_type: str, data: Any = None):
        """Trigger an event"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
    
    def run_all_independent(self) -> Dict[str, subprocess.CompletedProcess]:
        """Run all scripts independently"""
        results = {}
        for name, script in self.scripts.items():
            try:
                results[name] = script.run_independent()
            except Exception as e:
                logger.error(f"Failed to run {name}: {e}")
        return results
    
    def run_all_collaborative(self):
        """Run all scripts in collaborative mode"""
        for script in self.scripts.values():
            script.run_collaborative()
    
    def stop_all(self):
        """Stop all running scripts"""
        for script in self.scripts.values():
            script.stop()
    
    def save_state(self):
        """Save connector state to file"""
        state = {
            'shared_state': self.shared_state,
            'script_states': {name: script.state for name, script in self.scripts.items()},
            'config': self.config
        }
        try:
            with open(STATE_FILE, 'wb') as f:
                pickle.dump(state, f)
            logger.info("Saved connector state")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """Load connector state from file"""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'rb') as f:
                    state = pickle.load(f)
                    self.shared_state = state.get('shared_state', {})
                    # Restore script states
                    for script_name, script_state in state.get('script_states', {}).items():
                        if script_name in self.scripts:
                            for var, value in script_state.items():
                                self.scripts[script_name].set_variable(var, value)
                logger.info("Loaded connector state")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get connector status"""
        return {
            'running': self.running,
            'scripts_loaded': len(self.scripts),
            'scripts': {
                name: {
                    'loaded': script.module is not None,
                    'running': script.running,
                    'functions': list(script.functions.keys()),
                    'attributes': list(script.attributes.keys())
                }
                for name, script in self.scripts.items()
            },
            'shared_state': self.shared_state,
            'message_queue_size': self.message_queue.qsize()
        }
    
    def start(self):
        """Start the connector"""
        self.running = True
        logger.info("Enhanced Connector started")
        
        # Auto-discover scripts if enabled
        if self.config.get('auto_discover', True):
            self.discover_scripts()
        
        # Load previous state
        self.load_state()
        
        # Print status
        status = self.get_status()
        logger.info(f"Loaded {status['scripts_loaded']} scripts")
        for script, info in status['scripts'].items():
            logger.info(f"  - {script}: {len(info['functions'])} functions, {len(info['attributes'])} attributes")


# Global connector instance
connector = None


def get_connector() -> EnhancedConnector:
    """Get the global connector instance"""
    global connector
    if connector is None:
        connector = EnhancedConnector()
    return connector


def main():
    """Main function for the enhanced connector"""
    conn = get_connector()
    conn.start()
    
    # Example usage
    logger.info("Enhanced Connector ready for operations")
    
    # Demonstrate capabilities
    if conn.scripts:
        logger.info("\nAvailable operations:")
        logger.info("1. Run scripts independently")
        logger.info("2. Run scripts collaboratively")
        logger.info("3. Control script parameters")
        logger.info("4. Call script functions")
        logger.info("5. Share data between scripts")
        
        # Save state periodically
        conn.save_state()
    else:
        logger.warning("No scripts found in directory")
    
    return conn


if __name__ == "__main__":
    main()