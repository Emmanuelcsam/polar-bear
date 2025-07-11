"""
A sophisticated, interactive connector to analyze, manage, and execute 
functionality from various visualization modules within this directory.

This script serves as a node in a larger neural network, providing a 
standardized interface to the tools available in this directory.
"""

import os
import sys
import importlib
import inspect
import subprocess
import re
from pathlib import Path

# Add the script's directory to the Python path for local imports
# This allows the script to find and import other modules in the same directory.
file_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(file_dir))

# Import common logging utility
from common_data_and_utils import log_message as log

# --- Configuration ---
# LOG_FILE_NAME is now handled by common_data_and_utils if it sets up file logging
# LOG_LEVEL is now handled by common_data_and_utils if it sets up logging

log("Neural Connector Initializing...", level="INFO")

class NeuralConnector:
    """
    A robust system to discover, analyze, and interact with Python modules
    in the current directory.
    """
    def __init__(self):
        self.current_dir = Path(__file__).parent.resolve()
        self.modules = {}
        self.script_name = Path(__file__).name

    def find_importable_modules(self, file_path):
        """
        Parses a Python file to find all unique import statements.
        This is a simplified approach and might not catch all edge cases.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Regex to find module names from import statements
            # Handles: import module, from module import something
            patterns = [
                re.compile(r"^\s*import\s+([a-zA-Z0-9_]+)"),
                re.compile(r"^\s*from\s+([a-zA-Z0-9_]+)\s+import")
            ]
            
            found_modules = set()
            for line in content.splitlines():
                for pattern in patterns:
                    match = pattern.match(line)
                    if match:
                        found_modules.add(match.group(1))
                        break
            return found_modules
        except Exception as e:
            log.error(f"Could not read or parse {file_path}: {e}")
            return set()

    def check_and_install_dependencies(self):
        """
        Discovers all python scripts, checks their imports, and offers to 
        install missing dependencies.
        """
        log.info("Phase 1: Starting dependency analysis.")
        py_files = [p for p in self.current_dir.glob("*.py") if p.name != self.script_name]
        local_module_names = {p.stem for p in py_files}
        all_dependencies = set()

        for file_path in py_files:
            log.debug(f"Analyzing dependencies in: {file_path.name}")
            deps = self.find_importable_modules(file_path)
            all_dependencies.update(deps)

        # Common packages that are part of Python standard library
        standard_libs = set(sys.stdlib_module_names)
        
        missing_deps = set()
        for dep in all_dependencies:
            if dep in standard_libs or dep in local_module_names:
                continue
            try:
                importlib.import_module(dep)
                log.debug(f"Dependency '{dep}' is already installed.")
            except ImportError:
                log.warning(f"Potential missing dependency found: '{dep}'")
                missing_deps.add(dep)
        
        if not missing_deps:
            log.info("All dependencies appear to be satisfied.")
            return

        log.info("The following potential missing dependencies were found:")
        for dep in sorted(list(missing_deps)):
            print(f"  - {dep}")
        
        try:
            answer = input("Would you like to attempt to install them using pip? (y/n): ").lower()
            if answer == 'y':
                for dep in sorted(list(missing_deps)):
                    self.install_package(dep)
            else:
                log.info("Skipping installation of missing dependencies.")
        except KeyboardInterrupt:
            log.warning("\nOperation cancelled by user.")
            sys.exit(0)

    def install_package(self, package_name):
        """Installs a single package using pip."""
        log.info(f"Attempting to install '{package_name}'...")
        try:
            # Using sys.executable to ensure pip from the correct env is used
            command = [sys.executable, "-m", "pip", "install", package_name]
            log.debug(f"Executing command: {' '.join(command)}")
            
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Stream output in real-time
            for stdout_line in iter(process.stdout.readline, ""):
                log.info(f"[pip] {stdout_line.strip()}")
            process.stdout.close()
            
            return_code = process.wait()
            
            if return_code == 0:
                log.info(f"Successfully installed '{package_name}'.")
            else:
                stderr_output = process.stderr.read()
                log.error(f"Failed to install '{package_name}'. Pip exit code: {return_code}")
                log.error(f"[pip stderr] {stderr_output.strip()}")

        except FileNotFoundError:
            log.error("'pip' command not found. Please ensure Python and pip are installed and in your PATH.")
        except Exception as e:
            log.error(f"An unexpected error occurred during installation of {package_name}: {e}")

    def discover_modules(self):
        """
        Discovers and dynamically imports all valid Python scripts in the directory.
        """
        log.info("Phase 2: Discovering and loading local modules.")
        # Exclude the connector itself and any setup scripts.
        excluded_files = {self.script_name, "setup.py"}
        py_files = [p for p in self.current_dir.glob("*.py") if p.name not in excluded_files]

        for file_path in py_files:
            module_name = file_path.stem
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.modules[module_name] = module
                    log.debug(f"Successfully loaded module: '{module_name}'")
                else:
                    log.warning(f"Could not create module spec for {file_path.name}")
            except Exception as e:
                log.error(f"Failed to import module '{module_name}' from {file_path.name}: {e}", exc_info=True)
        
        if not self.modules:
            log.warning("No local modules were successfully loaded.")
        else:
            log.info(f"Successfully loaded {len(self.modules)} modules.")

    def get_callable_functions(self, module):
        """
        Inspects a module and returns a list of its public, callable functions.
        """
        functions = {}
        for name, func in inspect.getmembers(module, inspect.isfunction):
            # Exclude private functions and functions imported from other modules
            if not name.startswith("_") and func.__module__ == module.__name__:
                functions[name] = func
        return functions

    def run_interactive_session(self):
        """
        Runs the main interactive loop for the user to select and run functions.
        """
        log.info("Phase 3: Starting interactive session.")
        if not self.modules:
            log.error("No modules available to run. Exiting.")
            return

        while True:
            try:
                print("\n--- Neural Connector: Module Selection ---")
                module_list = sorted(self.modules.keys())
                for i, name in enumerate(module_list):
                    print(f"  {i+1}: {name}")
                print("  0: Exit")

                choice = input("Select a module to inspect: ")
                if not choice.isdigit() or not (0 <= int(choice) <= len(module_list)):
                    print("Invalid choice, please try again.")
                    continue
                
                choice_idx = int(choice) - 1
                if choice_idx == -1:
                    break
                
                module_name = module_list[choice_idx]
                self.inspect_module(module_name)

            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"An error occurred in the interactive session: {e}", exc_info=True)

        log.info("Interactive session ended.")

    def inspect_module(self, module_name):
        """Handles the interaction for a single selected module."""
        module = self.modules[module_name]
        functions = self.get_callable_functions(module)

        if not functions:
            print(f"\nModule '{module_name}' has no public, callable functions.")
            input("Press Enter to return to module selection...")
            return

        while True:
            try:
                print(f"\n--- Module: {module_name} | Function Selection ---")
                func_list = sorted(functions.keys())
                for i, name in enumerate(func_list):
                    print(f"  {i+1}: {name}")
                print("  0: Back to module selection")

                choice = input("Select a function to execute: ")
                if not choice.isdigit() or not (0 <= int(choice) <= len(func_list)):
                    print("Invalid choice, please try again.")
                    continue
                
                choice_idx = int(choice) - 1
                if choice_idx == -1:
                    break
                
                func_name = func_list[choice_idx]
                self.execute_function(module, func_name, functions[func_name])

            except KeyboardInterrupt:
                log("\nReturning to module selection.", level="WARNING")
                break
            except Exception as e:
                log.error(f"An error occurred while inspecting module {module_name}: {e}", exc_info=True)


    def execute_function(self, module, func_name, func):
        """Gathers arguments for and executes a selected function."""
        log.info(f"Preparing to execute {module.__name__}.{func_name}")
        print(f"\n--- Executing: {func_name} ---")
        
        try:
            sig = inspect.signature(func)
            args = {}
            
            if sig.parameters:
                log.debug(f"Signature for {func_name}: {sig}")
                print("Please provide values for the following parameters:")
                for name, param in sig.parameters.items():
                    prompt = f"  - {name}"
                    if param.annotation != inspect.Parameter.empty:
                        prompt += f" (type: {param.annotation.__name__})"
                    if param.default != inspect.Parameter.empty:
                        prompt += f" [default: {param.default!r}]"
                    prompt += ": "
                    
                    while True:
                        value_str = input(prompt)
                        if not value_str and param.default != inspect.Parameter.empty:
                            args[name] = param.default
                            log.debug(f"Using default value for '{name}': {param.default!r}")
                            break
                        
                        # Attempt to cast to the annotated type if possible
                        if param.annotation != inspect.Parameter.empty:
                            try:
                                args[name] = param.annotation(value_str)
                                log.debug(f"Casted '{name}' to {param.annotation.__name__}")
                                break
                            except (ValueError, TypeError):
                                print(f"  Error: Could not cast '{value_str}' to {param.annotation.__name__}. Please try again.")
                        else:
                            # No type hint, so just pass as string
                            args[name] = value_str
                            break
            else:
                print("This function takes no parameters.")

            log.info(f"Executing '{func_name}' with arguments: {args}")
            print("\nRunning... (Press Ctrl+C to interrupt)")
            
            result = func(**args)
            
            log.info(f"Function '{func_name}' executed successfully.")
            if result is not None:
                log.info(f"Return value: {result}")
                print(f"\nExecution Result:\n{result}")

        except KeyboardInterrupt:
            log.warning(f"Execution of '{func_name}' interrupted by user.")
            print("\nExecution cancelled.")
        except Exception as e:
            log.error(f"An error occurred while executing {func_name}: {e}", exc_info=True)
            print(f"\nAn error occurred: {e}")
        
        input("\nPress Enter to return to function selection...")


def main():
    """Main entry point for the Neural Connector script."""
    log("="*50)
    log("Neural Connector Initializing...")
    log("="*50)
    
    connector = NeuralConnector()
    connector.check_and_install_dependencies()
    connector.discover_modules()
    connector.run_interactive_session()
    
    log("Neural Connector shutting down.")

if __name__ == "__main__":
    main()
