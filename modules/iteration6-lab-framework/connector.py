#!/usr/bin/env python3
"""
Enhanced Connector Script for Lab Framework
Provides full integration capabilities with all scripts in the directory
"""

import os
import logging
import sys
import json
import socket
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.connector_interface import ConnectorClient, ScriptConnectorInterface

# --- Configuration ---
LOG_FILE = "connector.log"
CONNECTOR_PORT = 10006

# --- Setup Logging ---
logger = logging.getLogger(os.path.abspath(__file__))
logger.setLevel(logging.INFO)
logger.propagate = False

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


class LabFrameworkConnector:
    """Main connector for the lab framework"""
    
    def __init__(self):
        self.client = ConnectorClient(CONNECTOR_PORT)
        self.modules = {}
        self.scripts = {}
        self.base_path = Path(__file__).parent
        
    def discover_modules(self):
        """Discover all available modules and scripts"""
        logger.info("Discovering modules and scripts...")
        
        # Check modules directory
        modules_dir = self.base_path / "modules"
        if modules_dir.exists():
            for py_file in modules_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    module_name = py_file.stem
                    self.modules[module_name] = str(py_file)
                    logger.info(f"Found module: {module_name}")
                    
        # Check main directory scripts
        for py_file in self.base_path.glob("*.py"):
            if py_file.name not in ["connector.py", "hivemind_connector.py", "setup.py"]:
                script_name = py_file.stem
                self.scripts[script_name] = str(py_file)
                logger.info(f"Found script: {script_name}")
                
    def get_module_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a module"""
        if module_name in self.modules:
            try:
                spec = importlib.util.spec_from_file_location(module_name, self.modules[module_name])
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                info = {
                    'name': module_name,
                    'path': self.modules[module_name],
                    'functions': [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')],
                    'has_interface': hasattr(module, 'ScriptConnectorInterface')
                }
                
                # Check for module-level variables
                info['variables'] = [name for name in dir(module) if not name.startswith('_') and not callable(getattr(module, name))]
                
                return info
                
            except Exception as e:
                logger.error(f"Error loading module {module_name}: {e}")
                return None
                
        return None
        
    def control_module(self, module_name: str, action: str, **kwargs) -> Dict[str, Any]:
        """Control a module - set parameters, call functions, etc."""
        if module_name not in self.modules:
            return {'status': 'error', 'message': 'Module not found'}
            
        try:
            # Try to use connector interface first
            result = self.client.call_script_method(module_name, action, **kwargs)
            if result is not None:
                return {'status': 'success', 'result': result}
                
            # Fallback to direct module manipulation
            spec = importlib.util.spec_from_file_location(module_name, self.modules[module_name])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, action) and callable(getattr(module, action)):
                func = getattr(module, action)
                result = func(**kwargs)
                return {'status': 'success', 'result': result}
            else:
                return {'status': 'error', 'message': f'Action {action} not found in module'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def get_all_module_states(self) -> Dict[str, Any]:
        """Get the state of all modules"""
        states = {}
        
        for module_name in self.modules:
            info = self.get_module_info(module_name)
            if info:
                states[module_name] = info
                
        return states
        
    def execute_workflow(self, workflow: list) -> list:
        """Execute a workflow of module actions"""
        results = []
        
        for step in workflow:
            module = step.get('module')
            action = step.get('action')
            params = step.get('params', {})
            
            logger.info(f"Executing: {module}.{action} with params: {params}")
            result = self.control_module(module, action, **params)
            results.append({
                'step': f"{module}.{action}",
                'result': result
            })
            
            if result.get('status') == 'error':
                logger.error(f"Workflow stopped due to error: {result.get('message')}")
                break
                
        return results
        
    def send_to_hivemind(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a message to the hivemind connector"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', CONNECTOR_PORT))
            sock.send(json.dumps(message).encode())
            
            response = sock.recv(4096).decode()
            sock.close()
            
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Error communicating with hivemind: {e}")
            return None


def main():
    """Main function for the connector script"""
    logger.info(f"--- Lab Framework Connector Initialized ---")
    
    connector = LabFrameworkConnector()
    
    # Discover available modules
    connector.discover_modules()
    
    # Get status from hivemind
    logger.info("Checking hivemind connector status...")
    status = connector.send_to_hivemind({'command': 'status'})
    if status:
        logger.info(f"Hivemind status: {status}")
    else:
        logger.warning("Could not connect to hivemind connector")
        
    # List available modules
    logger.info("\nAvailable modules:")
    for module_name in connector.modules:
        info = connector.get_module_info(module_name)
        if info:
            logger.info(f"  - {module_name}: {len(info['functions'])} functions")
            
    # Example: Generate a random image using random_pixel module
    logger.info("\nExample: Generating random image...")
    result = connector.control_module('random_pixel', 'gen')
    if result.get('status') == 'success':
        logger.info("Random image generated successfully")
    else:
        logger.error(f"Failed to generate image: {result.get('message')}")
        
    # Example workflow
    logger.info("\nExample: Running a simple workflow...")
    workflow = [
        {'module': 'random_pixel', 'action': 'gen'},
        {'module': 'cv_module', 'action': 'batch', 'params': {'folder': 'data'}}
    ]
    
    results = connector.execute_workflow(workflow)
    for i, result in enumerate(results):
        logger.info(f"Step {i+1}: {result['step']} - {result['result'].get('status')}")
        
    logger.info("\nConnector demonstration completed")


if __name__ == "__main__":
    main()