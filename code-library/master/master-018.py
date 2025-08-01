# Merged Main Orchestration File
# -------------------------------
# This script combines the functionality of main.py, main-2.py, and main-5.py.
#
# It aims to create a single, robust entry point for the Neural Framework,
# incorporating the best features from all three source files:
#
# - From main-5.py (used as the base):
#   - Dependency verification and installation.
#   - Configuration management (loading/saving config.json).
#   - Discovery of "tunable parameters" from modules.
#   - A robust CLI structure and non-interactive summary mode.
#   - Unit test execution integration.
#
# - From main.py:
#   - Detailed module discovery logic (walking subdirectories).
#   - In-depth module analysis for the CLI (listing classes, methods, docstrings).
#
# - From main-2.py:
#   - The core concept of dynamically executing a specific function from a loaded module.
#

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
import inspect

# --- Path Setup ---
# Ensure the core framework modules can be imported.
# This adds the parent directory of 'core' to the Python path.
# Assumes a structure like: /project_root/core/main/merged_main.py
# We want to add /project_root to the path.
try:
    core_dir = Path(__file__).parent
    project_root = core_dir.parent.parent
    sys.path.insert(0, str(project_root))
except NameError:
    # Handle case where __file__ is not defined (e.g., interactive interpreter)
    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))


# --- Import Core Components ---
# These imports are collected from all three files.
# We will use the components from the most feature-rich implementations.
from core.logging_config import setup_logging
from core.dependency_manager import check_and_install_dependencies
from core.config_manager import ConfigManager
from modules.loader import ModuleLoader # Using the loader from main-5
from core.logger import log # Using the logger setup from main and main-2

# Setup logging at the very beginning
setup_logging()
logger = logging.getLogger("neural_orchestrator")


class NeuralOrchestrator:
    """
    The main application class for the Neural Orchestrator system.
    Orchestrates all components, provides an interactive CLI, and can run
    specific module functions.
    (Class combines concepts from NeuralConnector and MainOrchestrator)
    """
    def __init__(self):
        self.project_root = project_root
        self.config_manager = ConfigManager()
        self.config = {}
        self.module_loader = None
        self.loaded_modules = {}
        self.sorted_modules = [] # For easy CLI selection

    def startup(self):
        """
        Initializes the entire system.
        (Logic combined from startup() methods in all files)
        """
        logger.info("=================================================")
        logger.info("--- Initializing Neural Orchestrator System ---")
        logger.info("=================================================")

        logger.info("STEP 1: Verifying dependencies...")
        # This function comes from main-5's dependency_manager
        check_and_install_dependencies()
        logger.info("Dependency check complete.")

        logger.info("\nSTEP 2: Loading configuration...")
        # Using the ConfigManager from main-5
        self.config = self.config_manager.load_config()
        # Re-init logging with level from config
        setup_logging(self.config.get("general", {}).get("log_level", "INFO"))
        logger.info("Configuration loaded successfully.")

        logger.info("\nSTEP 3: Discovering and loading modules...")
        # Using the ModuleLoader from main-5, which is more advanced
        # and includes analysis of functions and parameters.
        ignore_list = self.config.get("module_loader", {}).get("scripts_to_ignore", [])
        ignore_list.append(Path(__file__).name)

        self.module_loader = ModuleLoader(base_dir=str(self.project_root), ignore_list=ignore_list)
        self.loaded_modules = self.module_loader.discover_and_load_modules()
        logger.info(f"Module loading complete. Found {len(self.loaded_modules)} modules.")

        logger.info("\nSTEP 4: Updating tunable parameters in configuration...")
        # This feature is from main-5
        self.update_tunable_parameters()

        logger.info("\n--- Framework startup complete ---")


    def update_tunable_parameters(self):
        """
        Updates the main configuration with parameters discovered by the loader.
        (Logic from main-5.py)
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
        (CLI is a merge of the CLIs from main.py and main-5.py)
        """
        print("\n" + "="*50)
        print("    Welcome to the Neural Orchestrator Interface")
        print("="*50)

        while True:
            print("\nAvailable Commands:")
            print("  1. List all loaded modules")
            print("  2. View detailed module analysis")
            print("  3. Run a function from a module")
            print("  4. View all tunable parameters")
            print("  5. Run unit tests")
            print("  6. Exit")

            choice = input("\nEnter your choice: ")

            if choice == '1':
                self.cli_list_modules()
            elif choice == '2':
                self.cli_view_module_details()
            elif choice == '3':
                self.cli_run_function()
            elif choice == '4':
                self.cli_view_parameters()
            elif choice == '5':
                self.cli_run_tests()
            elif choice == '6':
                logger.info("Exiting Neural Orchestrator.")
                break
            else:
                print("Invalid choice. Please try again.")

    def cli_list_modules(self):
        """
        Lists all loaded modules.
        (Adapted from _list_modules in main.py)
        """
        print("\n--- Loaded Modules ---")
        if not self.loaded_modules:
            print("No modules were loaded.")
            return

        # Use a sorted list for consistent ordering
        self.sorted_modules = sorted(list(self.loaded_modules.keys()))
        for i, name in enumerate(self.sorted_modules, 1):
            print(f"{i}. {name}")
        print("-" * 22)

    def cli_view_module_details(self):
        """
        Shows a detailed analysis of a specific module, including its
        functions, classes, and methods.
        (This is a direct port of the feature from main.py)
        """
        self.cli_list_modules() # Show the list to choose from
        if not self.loaded_modules:
            return
        try:
            choice_str = input("\nSelect a module to view details (or press Enter to cancel): ")
            if not choice_str: return
            choice = int(choice_str) - 1
            if not (0 <= choice < len(self.sorted_modules)):
                raise IndexError

            module_name = self.sorted_modules[choice]
            info = self.loaded_modules[module_name] # ModuleInfo object from loader

            print(f"\n--- Details for Module: {module_name} ---")
            print(f"  File Path: {info.file_path}")

            print("\n  Functions:")
            if not info.functions:
                print("    No top-level functions found.")
            else:
                for func_name, func_obj in sorted(info.functions.items()):
                    doc = inspect.getdoc(func_obj) or "No docstring."
                    sig = inspect.signature(func_obj)
                    print(f"    - def {func_name}{sig}:")
                    print(f"      \"\"\"{doc}\"\"\"")


            print("\n  Classes:")
            if not info.classes:
                print("    No classes found.")
            else:
                for class_name, class_obj in sorted(info.classes.items()):
                    doc = inspect.getdoc(class_obj) or "No docstring."
                    print(f"    - class {class_name}:")
                    print(f"      \"\"\"{doc}\"\"\"")
                    for method_name, method_obj in sorted(info.methods.get(class_name, {}).items()):
                        if not method_name.startswith('_'):
                            doc = inspect.getdoc(method_obj) or "No docstring."
                            sig = inspect.signature(method_obj)
                            print(f"      - def {method_name}{sig}:")
                            print(f"        \"\"\"{doc}\"\"\"")

        except (ValueError, IndexError):
            print("Invalid selection.")
        except Exception as e:
            logger.error(f"An error occurred while viewing module details: {e}", exc_info=True)

    def cli_run_function(self):
        """
        Allows the user to select and run a function from a loaded module.
        (This feature is inspired by the demonstration in main-2.py)
        """
        print("\n--- Run a Function ---")
        self.cli_list_modules()
        if not self.loaded_modules:
            return

        try:
            # 1. Select Module
            choice_str = input("\nSelect a module to run a function from (or press Enter to cancel): ")
            if not choice_str: return
            module_idx = int(choice_str) - 1
            if not (0 <= module_idx < len(self.sorted_modules)):
                raise IndexError
            module_name = self.sorted_modules[module_idx]
            module_info = self.loaded_modules[module_name]

            # 2. Select Function
            if not module_info.functions:
                print(f"No runnable top-level functions found in {module_name}.")
                return

            print(f"\n--- Functions in {module_name} ---")
            runnable_functions = sorted(list(module_info.functions.keys()))
            for i, func_name in enumerate(runnable_functions, 1):
                print(f"  {i}. {func_name}")

            func_choice_str = input("\nSelect a function to run (or press Enter to cancel): ")
            if not func_choice_str: return
            func_idx = int(func_choice_str) - 1
            if not (0 <= func_idx < len(runnable_functions)):
                raise IndexError
            func_name = runnable_functions[func_idx]

            # 3. Execute
            logger.info(f"Attempting to run function '{func_name}' from module '{module_name}'.")
            print(f"\nRunning {module_name}.{func_name}()...")
            print("-" * 30)

            func_to_run = module_info.functions[func_name]
            
            # Simple execution for now. A more advanced version could
            # prompt for arguments using the tunable_parameters.
            func_to_run()

            print("-" * 30)
            logger.info(f"Successfully executed '{func_name}'.")
            print("Function execution finished.")

        except (ValueError, IndexError):
            print("Invalid selection.")
        except Exception as e:
            logger.error(f"Failed to execute function: {e}", exc_info=True)
            print(f"An error occurred during execution: {e}")


    def cli_view_parameters(self):
        """
        Displays all discovered tunable parameters in JSON format.
        (Logic from main-5.py)
        """
        print("\n--- All Tunable Parameters ---")
        params = self.config.get("tunable_parameters", {})
        if not params:
            print("No tunable parameters were found.")
            return

        print(json.dumps(params, indent=2, default=str))
        print("-" * 30)

    def cli_run_tests(self):
        """
        Runs the test suite from the CLI.
        (Logic from main-5.py)
        """
        print("\n--- Running Unit Test Suite ---")
        try:
            from run_tests import run_all_tests
            run_all_tests()
        except ImportError:
            logger.error("Could not find 'run_tests.py'. Place it in the project root.")
            print("Error: Could not find 'run_tests.py'.")
        except Exception as e:
            logger.error(f"An error occurred while running tests: {e}", exc_info=True)

    def run_summary(self):
        """
        Prints a non-interactive summary and exits.
        (Logic from main-5.py)
        """
        print("--- Neural Orchestrator Non-Interactive Summary ---")
        print(f"Successfully loaded {len(self.loaded_modules)} modules.")
        # Create a temporary sorted list for the summary
        self.sorted_modules = sorted(list(self.loaded_modules.keys()))
        for name in self.sorted_modules:
            info = self.loaded_modules[name]
            print(f"\n[+] Module: {name}.py")
            if info.functions:
                for func_name in sorted(info.functions.keys()):
                    param_str = ", ".join(info.tunable_parameters.get(func_name, {}).keys())
                    print(f"    - Function: {func_name}({param_str})")
            else:
                print("    - No functions found in this module.")
        print("\n--- System Initialized Successfully ---")


def main():
    """
    The main entry point of the application.
    (Logic combined from all three files)
    """
    try:
        orchestrator = NeuralOrchestrator()
        orchestrator.startup()

        # Check if running in an interactive terminal (like in main-5)
        if sys.stdout.isatty():
            orchestrator.run_cli()
        else:
            # If not (e.g., piped to a file), run the summary
            orchestrator.run_summary()

    except Exception as e:
        logger.critical("A critical error occurred in the main application.", exc_info=True)
        print(f"\nA critical error occurred: {e}")
        print("Please check the log file for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
