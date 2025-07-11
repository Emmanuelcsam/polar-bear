
import os
import sys
import logging
import subprocess
import importlib
import inspect
from datetime import datetime
import importlib.util

class ColoredFormatter(logging.Formatter):
    """A custom log formatter with colors."""
    
    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    GREEN = "\x1b[32;20m"
    RESET = "\x1b[0m"
    
    def __init__(self, fmt):
        super().__init__(fmt)
        self.FORMATS = {
            logging.DEBUG: self.GREY + fmt + self.RESET,
            logging.INFO: self.GREEN + fmt + self.RESET,
            logging.WARNING: self.YELLOW + fmt + self.RESET,
            logging.ERROR: self.RED + fmt + self.RESET,
            logging.CRITICAL: self.BOLD_RED + fmt + self.RESET
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger():
    """Sets up the logger for the connector."""
    logger = logging.getLogger("NeuralConnector")
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    log_file = "neural_connector.log"
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter('%(asctime)s - [%(levelname)s] - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

class DependencyManager:
    """Manages detection and installation of dependencies."""
    
    def __init__(self, logger):
        self.logger = logger
        self.dependencies = {
            # From computer-vision-tutorial.py
            "json": ("json", None),
            "os": ("os", None),
            "cv2": ("opencv-python", "cv2"),
            "matplotlib": ("matplotlib", "matplotlib"),
            "numpy": ("numpy", "numpy"),
            "torch": ("torch", "torch"),
            "torchvision": ("torchvision", "torchvision"),
            "tqdm": ("tqdm", "tqdm"),
            "pandas": ("pandas", "pandas"),
            "sklearn": ("scikit-learn", "sklearn"),
            # From cv-training-guide.py
            "requests": ("requests", "requests"),
            "torchmetrics": ("torchmetrics", "torchmetrics"),
            "mlxtend": ("mlxtend", "mlxtend"),
            # From detection-tutorial.py
            "tensorboard": ("tensorboard", "tensorboard"),
        }

    def check_and_install(self):
        """Checks all dependencies and installs them if they are missing."""
        self.logger.info("Starting dependency check...")
        all_good = True
        for package, (package_name, import_name) in self.dependencies.items():
            if import_name: # Skip standard libraries like os, json
                if not self._check_single_package(package_name, import_name):
                    all_good = False
        
        if all_good:
            self.logger.info("All dependencies are satisfied.")
        else:
            self.logger.warning("Some dependencies were installed. It's recommended to restart the script.")
        return all_good
            
    def _check_single_package(self, package_name, import_name):
        """Checks and installs a single package."""
        try:
            importlib.import_module(import_name)
            self.logger.info(f"Dependency '{package_name}' is already installed.")
            return True
        except ImportError:
            self.logger.warning(f"Dependency '{package_name}' not found. Attempting to install...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
                self.logger.info(f"Successfully installed '{package_name}'.")
                # Verify installation
                importlib.import_module(import_name)
                return True
            except (subprocess.CalledProcessError, ImportError) as e:
                self.logger.error(f"Failed to install '{package_name}'. Error: {e}")
                return False

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
            if filename.endswith(".py") and filename not in ["neural_connector.py", "setup.py"]:
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
        self.logger = setup_logger()
        self.dependency_manager = DependencyManager(self.logger)
        self.module_manager = ModuleManager(self.logger)
        
    def initialize(self):
        self.logger.info("Initializing Neural Connector...")
        if not self.dependency_manager.check_and_install():
             self.logger.critical("Critical dependencies failed to install. Exiting.")
             sys.exit(1)
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
        print("  run <module.func> - Execute a function from a module (TODO).")
        print("  test <module>     - Run tests for a specific module (TODO).")
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


if __name__ == "__main__":
    connector = NeuralConnector()
    connector.run()
