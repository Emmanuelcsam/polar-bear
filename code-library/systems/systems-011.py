
import os
import sys
import logging
import subprocess
import importlib
import inspect
from datetime import datetime
import importlib.util
import json

# --- Start of Connector Integration ---
try:
    import connector
    logger = connector.logger
except ImportError:
    print("Connector not found, using basic logging.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    logger = logging.getLogger(__name__)

CONFIG_FILE = "shared_config.json"
SCRIPT_NAME = "neural_connector"
# --- End of Connector Integration ---


class ModuleManager:
    """Loads and manages the tutorial modules."""
    def __init__(self, logger):
        self.logger = logger
        self.modules = {}
        self.registry = {}

    def discover_and_load_modules(self):
        """Discovers and loads Python scripts from the current directory."""
        self.logger.info("Discovering and loading modules...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(current_dir):
            if filename.endswith(".py") and filename not in ["neural_connector.py", "setup.py", "connector.py", "hivemind_connector.py"]:
                module_name = filename[:-3]
                try:
                    spec = importlib.util.spec_from_file_location(module_name, os.path.join(current_dir, filename))
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.modules[module_name] = module
                    self.logger.info(f"Successfully loaded module: {module_name}")
                    self._register_module_contents(module_name, module)
                except Exception as e:
                    self.logger.error(f"Failed to load module {module_name}: {e}")
        self.logger.info("Module discovery complete.")

    def _register_module_contents(self, module_name, module):
        """Inspects a module and registers its functions, classes, and variables."""
        self.registry[module_name] = {"functions": {}, "classes": {}, "variables": {}}
        for name, member in inspect.getmembers(module):
            if not name.startswith("_"):
                if inspect.isfunction(member):
                    self.registry[module_name]["functions"][name] = {
                        "callable": member,
                        "signature": str(inspect.signature(member))
                    }
                elif inspect.isclass(member):
                     self.registry[module_name]["classes"][name] = {
                        "class": member,
                        "methods": {m_name: m_member for m_name, m_member in inspect.getmembers(member, predicate=inspect.isfunction)}
                    }
                # Simple variable detection (avoids modules, etc.)
                elif isinstance(member, (int, float, str, list, dict, tuple, bool)):
                     self.registry[module_name]["variables"][name] = member

    def list_modules(self):
        """Returns a list of loaded module names."""
        return list(self.registry.keys())

    def get_module_details(self, module_name):
        """Returns the registered contents of a module."""
        return self.registry.get(module_name)

class NeuralConnector:
    def __init__(self):
        self.logger = logger
        self.module_manager = ModuleManager(self.logger)
        
    def initialize(self):
        self.logger.info("Initializing Neural Connector...")
        self.module_manager.discover_and_load_modules()
        
    def run(self):
        self.initialize()
        self.logger.info("Neural Connector is running. Type 'help' for commands.")
        self.interactive_cli()

    def interactive_cli(self):
        """The main interactive command-line interface loop."""
        while True:
            try:
                command = input("connector> ").strip().lower()
                if not command:
                    continue
                
                parts = command.split()
                base_cmd = parts[0]

                if base_cmd == "exit":
                    self.logger.info("Exiting Neural Connector.")
                    break
                elif base_cmd == "help":
                    self.print_help()
                elif base_cmd == "list":
                    self.list_modules()
                elif base_cmd == "show":
                    if len(parts) > 1:
                        self.show_module(parts[1])
                    else:
                        self.logger.warning("Usage: show <module_name>")
                elif base_cmd == "config":
                    self.show_config()
                else:
                    self.logger.warning(f"Unknown command: '{base_cmd}'. Type 'help' for a list of commands.")

            except KeyboardInterrupt:
                self.logger.info("\nExiting due to user request (Ctrl+C).")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the CLI: {e}")

    def print_help(self):
        """Prints the help message for the CLI."""
        print("\n--- Neural Connector CLI Help ---")
        print("Available Commands:")
        print("  list              - List all loaded modules.")
        print("  show <module_name>  - Show details of a specific module (functions, classes).")
        print("  config            - Show the content of shared_config.json.")
        print("  help              - Show this help message.")
        print("  exit              - Exit the connector.")
        print("---------------------------------\n")

    def list_modules(self):
        """Handles the 'list' command."""
        modules = self.module_manager.list_modules()
        if not modules:
            self.logger.info("No modules loaded.")
            return
        print("\n--- Loaded Modules ---")
        for module_name in modules:
            print(f"  - {module_name}")
        print("----------------------\n")

    def show_module(self, module_name):
        """Handles the 'show' command."""
        details = self.module_manager.get_module_details(module_name)
        if not details:
            self.logger.error(f"Module '{module_name}' not found.")
            return

        print(f"\n--- Details for Module: {module_name} ---")
        
        if details["classes"]:
            print("\n[Classes]")
            for class_name, class_info in details["classes"].items():
                print(f"  - {class_name}")
        
        if details["functions"]:
            print("\n[Functions]")
            for func_name, func_info in details["functions"].items():
                print(f"  - {func_name}{func_info['signature']}")

        if details["variables"]:
            print("\n[Tunable Parameters (Variables)]")
            for var_name, var_value in details["variables"].items():
                value_str = str(var_value)
                if len(value_str) > 70:
                    value_str = value_str[:67] + "..."
                print(f"  - {var_name}: {value_str}")

        print("---------------------------------------\n")

    def show_config(self):
        """Displays the content of the shared configuration file."""
        self.logger.info(f"Reading configuration from {CONFIG_FILE}...")
        try:
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
            
            print("\n--- Shared Configuration ---")
            print(json.dumps(config_data, indent=4))
            print("--------------------------\n")

        except FileNotFoundError:
            self.logger.error(f"{CONFIG_FILE} not found.")
        except json.JSONDecodeError:
            self.logger.error(f"Could not decode JSON from {CONFIG_FILE}.")


if __name__ == "__main__":
    connector_instance = NeuralConnector()
    connector_instance.run()
