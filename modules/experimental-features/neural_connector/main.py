import logging
import json
import sys
from pathlib import Path

from core.logging_config import setup_logging
from core.dependency_manager import check_and_install_dependencies
from core.config_manager import ConfigManager
from modules.loader import ModuleLoader

# Setup logging at the very beginning
setup_logging()
logger = logging.getLogger("neural_connector.main")

class NeuralConnector:
    """
    The main application class for the Neural Connector system.
    Orchestrates all components and provides the main user interface.
    """
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = {}
        self.module_loader = None
        self.loaded_modules = {}

    def startup(self):
        """
        Initializes the entire system.
        """
        logger.info("--- Starting Neural Connector System ---")

        logger.info("STEP 1: Verifying dependencies...")
        check_and_install_dependencies()
        logger.info("Dependency check complete.")

        logger.info("STEP 2: Loading configuration...")
        self.config = self.config_manager.load_config()
        setup_logging(self.config.get("general", {}).get("log_level", "INFO"))
        logger.info("Configuration loaded successfully.")

        logger.info("STEP 3: Discovering and loading experimental modules...")
        base_dir = Path(__file__).parent.parent
        ignore_list = self.config.get("module_loader", {}).get("scripts_to_ignore", [])
        ignore_list.append(Path(__file__).name)

        self.module_loader = ModuleLoader(base_dir=str(base_dir), ignore_list=ignore_list)
        self.loaded_modules = self.module_loader.discover_and_load_modules()
        logger.info("Module loading complete.")
        
        self.update_tunable_parameters()

    def update_tunable_parameters(self):
        """
        Updates the main configuration with parameters discovered by the loader.
        """
        logger.info("Updating configuration with tunable parameters from loaded modules.")
        all_params = self.module_loader.get_all_tunable_parameters()
        
        if "tunable_parameters" not in self.config:
            self.config["tunable_parameters"] = {}
            
        self.config["tunable_parameters"] = all_params
        self.config_manager.save_config()
        logger.info("Configuration updated and saved.")

    def run_cli(self):
        """
        Runs the main interactive command-line interface.
        """
        print("\n" + "="*50)
        print("    Welcome to the Neural Connector Interface")
        print("="*50)

        while True:
            print("\nAvailable Commands:")
            print("  1. List all loaded modules and their functions")
            print("  2. View all tunable parameters")
            print("  3. Run unit tests")
            print("  4. Exit")

            choice = input("\nEnter your choice: ")

            if choice == '1':
                self.cli_list_modules()
            elif choice == '2':
                self.cli_view_parameters()
            elif choice == '3':
                self.cli_run_tests()
            elif choice == '4':
                logger.info("Exiting Neural Connector.")
                break
            else:
                print("Invalid choice. Please try again.")

    def cli_list_modules(self):
        print("\n--- Loaded Modules and Functions ---")
        if not self.loaded_modules:
            print("No modules were loaded.")
            return
            
        for name, info in sorted(self.loaded_modules.items()):
            print(f"\n[+] Module: {name}.py")
            if info.functions:
                for func_name in sorted(info.functions.keys()):
                    param_str = ", ".join(info.tunable_parameters.get(func_name, {}).keys())
                    print(f"    - Function: {func_name}({param_str})")
            else:
                print("    - No functions found in this module.")
        print("-" * 34)

    def cli_view_parameters(self):
        print("\n--- All Tunable Parameters ---")
        params = self.config.get("tunable_parameters", {})
        if not params:
            print("No tunable parameters were found.")
            return
            
        print(json.dumps(params, indent=2, default=str))
        print("-" * 30)

    def cli_run_tests(self):
        """Runs the test suite from the CLI."""
        from run_tests import run_all_tests
        run_all_tests()

    def run_summary(self):
        """Prints a non-interactive summary and exits."""
        print("--- Neural Connector Non-Interactive Summary ---")
        print(f"Successfully loaded {len(self.loaded_modules)} modules.")
        self.cli_list_modules()
        print("\n--- System Initialized Successfully ---")


def main():
    """
    The main entry point of the application.
    """
    try:
        connector = NeuralConnector()
        connector.startup()
        
        if sys.stdout.isatty():
            connector.run_cli()
        else:
            connector.run_summary()

    except Exception as e:
        logger.critical("A critical error occurred in the main application.", exc_info=True)
        print(f"A critical error occurred: {e}")
        print("Please check the log file for more details.")

if __name__ == "__main__":
    main()