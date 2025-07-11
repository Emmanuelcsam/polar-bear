
import os
import logging
import sys
import json
from pathlib import Path
from script_interface import ScriptManager

# --- Configuration ---
LOG_FILE = "connector.log"

# --- Setup Logging ---
# Ensure the logger is configured from scratch for each script
logger = logging.getLogger(os.path.abspath(__file__))
logger.setLevel(logging.INFO)

# Prevent logging from propagating to the root logger
logger.propagate = False

# Remove any existing handlers to avoid duplicate logs
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

# File Handler
try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except (IOError, OSError) as e:
    # Fallback to console if file logging fails
    print(f"Could not write to log file {LOG_FILE}: {e}", file=sys.stderr)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


class Connector:
    """Main connector class for managing scripts"""
    
    def __init__(self):
        self.script_manager = ScriptManager(os.getcwd())
        self.logger = logger
        
    def initialize(self):
        """Initialize the connector and discover scripts"""
        self.logger.info(f"--- Connector Script Initialized in {os.getcwd()} ---")
        self.logger.info("This script connects and controls all modules in this directory.")
        
        # Discover and load scripts
        scripts = self.script_manager.discover_scripts()
        self.logger.info(f"Discovered {len(scripts)} scripts: {scripts}")
        
        # Load all scripts
        load_results = self.script_manager.load_all_scripts()
        for script, success in load_results.items():
            status = "loaded" if success else "failed"
            self.logger.info(f"- {script}: {status}")
            
        return load_results
        
    def get_script_info(self, script_name=None):
        """Get information about scripts"""
        if script_name:
            return self.script_manager.get_script_info(script_name)
        return self.script_manager.get_all_scripts_info()
        
    def execute_action(self, script_name, action):
        """Execute an action on a script"""
        result = self.script_manager.execute_on_script(script_name, action)
        self.logger.info(f"Executed {action.get('type')} on {script_name}: {result}")
        return result
        
    def broadcast_action(self, action):
        """Broadcast an action to all scripts"""
        results = self.script_manager.broadcast_action(action)
        self.logger.info(f"Broadcast {action.get('type')} to all scripts")
        return results
        
    def orchestrate_workflow(self, workflow):
        """Orchestrate a workflow across multiple scripts"""
        self.logger.info(f"Orchestrating workflow with {len(workflow)} steps")
        results = self.script_manager.orchestrate_collaboration(workflow)
        return results
        
    def control_variable(self, script_name, variable_name, value=None):
        """Control a variable in a script"""
        if value is not None:
            action = {
                'type': 'set_variable',
                'variable_name': variable_name,
                'value': value
            }
        else:
            action = {
                'type': 'get_variable',
                'variable_name': variable_name
            }
        return self.execute_action(script_name, action)
        
    def call_function(self, script_name, function_name, *args, **kwargs):
        """Call a function in a script"""
        action = {
            'type': 'call_function',
            'function_name': function_name,
            'args': args,
            'kwargs': kwargs
        }
        return self.execute_action(script_name, action)
        
    def run_script_main(self, script_name):
        """Run a script's main function"""
        action = {'type': 'run_main'}
        return self.execute_action(script_name, action)
        
    def get_function_signatures(self, script_name, function_name=None):
        """Get function signatures from a script"""
        if function_name:
            action = {
                'type': 'get_function_signature',
                'function_name': function_name
            }
            return self.execute_action(script_name, action)
        else:
            # Get all function signatures
            info = self.get_script_info(script_name)
            signatures = {}
            for func in info.get('functions', []):
                action = {
                    'type': 'get_function_signature',
                    'function_name': func
                }
                signatures[func] = self.execute_action(script_name, action)
            return signatures


def main():
    """Main function for the connector script."""
    connector = Connector()
    connector.initialize()
    
    # Example usage
    all_info = connector.get_script_info()
    logger.info(f"Script information: {json.dumps(all_info, indent=2)}")
    
    logger.info("Connector ready for operations.")

if __name__ == "__main__":
    main()
