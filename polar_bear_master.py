
import os
import sys
import subprocess
import logging
import json
import time
import argparse
from pathlib import Path
import importlib.util

class LoggingManager:
    """Handles logging to both console and file."""
    def __init__(self, log_file="polar_bear_master.log", log_level=logging.INFO):
        self.logger = logging.getLogger("PolarBearMaster")
        self.logger.setLevel(log_level)

        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger

class DependencyManager:
    """Handles checking and installing Python dependencies."""
    def __init__(self, logger):
        self.logger = logger

    def check_and_install(self, requirements_paths):
        """
        Checks for and installs dependencies from a list of requirements files.
        """
        self.install_package('setuptools')
        for req_path in requirements_paths:
            if not req_path.exists():
                self.logger.warning(f"Requirements file not found: {req_path}")
                continue

            self.logger.info(f"Checking requirements from {req_path}...")
            with open(req_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.install_package(line)

    def install_package(self, package_name):
        """Installs a single package if it's not already installed."""
        try:
            import pkg_resources
            try:
                pkg_resources.get_distribution(package_name.split('==')[0].split('>=')[0])
                self.logger.info(f"Requirement already satisfied: {package_name}")
                return
            except pkg_resources.DistributionNotFound:
                pass  # Package not found, proceed with installation.
        except ImportError:
            self.logger.debug("pkg_resources not found. Proceeding with installation.")

        self.logger.info(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--break-system-packages"])
            self.logger.info(f"Successfully installed {package_name}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install {package_name}: {e}")


class ConfigurationManager:
    """Manages user configuration via interactive prompts."""
    def __init__(self, logger, config_file="polar_bear_config.json"):
        self.logger = logger
        self.config_file = Path(config_file)
        self.config = {}

    def load_config(self, interactive=True):
        """Loads configuration from file or starts interactive setup."""
        if self.config_file.exists():
            self.logger.info(f"Loading configuration from {self.config_file}")
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        elif interactive:
            self.logger.info("No configuration file found. Starting interactive setup.")
            self.interactive_setup()
            self.save_config()
        else:
            self.logger.warning("No config file found and non-interactive mode. Using default settings.")


    def save_config(self):
        """Saves the current configuration to a file."""
        self.logger.info(f"Saving configuration to {self.config_file}")
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get(self, key, default=None):
        """Gets a configuration value."""
        return self.config.get(key, default)

    def _ask_question(self, question, default=None):
        """Helper to ask a question and get an answer."""
        prompt = f"{question}"
        if default is not None:
            prompt += f" [{default}]"
        
        answer = input(prompt + ": ").strip()
        return answer if answer else default

    def interactive_setup(self):
        """Runs the interactive configuration setup."""
        self.logger.info("Starting interactive configuration...")
        # Add questions here based on script analysis
        self.config['project_name'] = self._ask_question("Enter project name", "Polar Bear")
        self.config['log_level'] = self._ask_question("Enter log level (INFO, DEBUG)", "INFO")
        # ... more questions to come
        self.logger.info("Interactive configuration complete.")

class ConnectorManager:
    """Discovers and manages connector scripts."""
    def __init__(self, logger):
        self.logger = logger
        self.connectors = []

    def find_connectors(self, root_path):
        """Finds all connector scripts in the project."""
        self.logger.info(f"Searching for connectors in {root_path}...")
        self.connectors = list(Path(root_path).rglob("*connector.py"))
        self.logger.info(f"Found {len(self.connectors)} connector scripts.")
        return self.connectors

import ast
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor

class ScriptAnalyzer:
    """Analyzes Python scripts to extract metadata."""
    def __init__(self, logger):
        self.logger = logger

    def analyze_script(self, script_path):
        """Analyzes a single Python script."""
        self.logger.debug(f"Analyzing script: {script_path}")
        metadata = {
            'path': str(script_path),
            'functions': [],
            'classes': [],
            'imports': [],
            'docstring': None,
            'hash': None,
            'error': None
        }
        try:
            with open(script_path, 'rb') as f:
                content_bytes = f.read()
                metadata['hash'] = hashlib.md5(content_bytes).hexdigest()
                content = content_bytes.decode('utf-8', errors='ignore')

            tree = ast.parse(content)
            metadata['docstring'] = ast.get_docstring(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        metadata['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        metadata['imports'].append(node.module)
                elif isinstance(node, ast.FunctionDef):
                    metadata['functions'].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    metadata['classes'].append(node.name)
        except Exception as e:
            metadata['error'] = str(e)
            self.logger.error(f"Error analyzing {script_path}: {e}")
        return metadata

class TaskManager:
    """Manages and executes high-level tasks."""
    def __init__(self, logger, config, analyzer, connectors):
        self.logger = logger
        self.config = config
        self.analyzer = analyzer
        self.connectors = connectors
        self.tasks = {
            "1": ("Analyze Project", "analyze_project"),
            "2": ("Run Diagnostics", "run_diagnostics"),
            "3": ("Manage Connectors", "manage_connectors"),
            "4": ("Exit", "exit_program")
        }

    def display_menu(self):
        """Displays the main menu of tasks."""
        self.logger.info("\nAvailable Tasks:")
        for key, (name, _) in self.tasks.items():
            self.logger.info(f"  {key}. {name}")

    def get_task(self):
        """Gets a task from the user."""
        while True:
            choice = input("Select a task: ")
            task_info = self.tasks.get(choice)
            if task_info:
                task_name = task_info[1]
                return getattr(self, task_name, self.invalid_task)
            else:
                self.logger.warning("Invalid choice. Please try again.")

    def analyze_project(self):
        """Analyzes all Python scripts in the project and displays a summary."""
        self.logger.info("Starting project analysis...")
        
        py_files = list(Path.cwd().rglob("*.py"))
        self.logger.info(f"Found {len(py_files)} Python files to analyze.")
        
        all_metadata = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.analyzer.analyze_script, p) for p in py_files]
            for future in futures:
                try:
                    result = future.result()
                    if not result['error']:
                        all_metadata.append(result)
                except Exception as e:
                    self.logger.error(f"Error during script analysis execution: {e}")

        if not all_metadata:
            self.logger.warning("No Python files were successfully analyzed.")
            return

        # Aggregate results
        total_files = len(all_metadata)
        total_functions = sum(len(m['functions']) for m in all_metadata)
        total_classes = sum(len(m['classes']) for m in all_metadata)
        
        all_imports = set()
        for m in all_metadata:
            all_imports.update(m['imports'])

        self.logger.info("--- Project Analysis Summary ---")
        self.logger.info(f"Total Python Files Analyzed: {total_files}")
        self.logger.info(f"Total Functions Found: {total_functions}")
        self.logger.info(f"Total Classes Found: {total_classes}")
        self.logger.info(f"Total Unique Imports: {len(all_imports)}")
        self.logger.info("---------------------------------")
        
        # Optionally, save the detailed analysis to a file
        save_report = input("Save detailed analysis report to a file? (y/n): ").lower()
        if save_report == 'y':
            report_path = "project_analysis_report.json"
            with open(report_path, 'w') as f:
                json.dump(all_metadata, f, indent=4)
            self.logger.info(f"Detailed report saved to {report_path}")
        
        self.logger.info("Project analysis complete.")

    def run_diagnostics(self):
        """Runs basic diagnostic checks on the project environment."""
        self.logger.info("Running diagnostics...")
        checks_passed = True

        # Check 1: Configuration file existence
        if self.config.config_file.exists():
            self.logger.info("[PASS] Configuration file found.")
        else:
            self.logger.warning("[FAIL] Configuration file not found.")
            checks_passed = False

        # Check 2: Log file was created
        log_file_found = False
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file = Path(handler.baseFilename)
                if log_file.exists():
                    self.logger.info(f"[PASS] Log file found at {log_file}.")
                    log_file_found = True
                break
        if not log_file_found:
            self.logger.error(f"[FAIL] Log file could not be verified.")
            checks_passed = False
            
        # Check 3: Required directories
        required_dirs = ["modules", "used", "docs", "meta-tools"]
        for req_dir in required_dirs:
            if Path(req_dir).is_dir():
                self.logger.info(f"[PASS] Directory '{req_dir}' found.")
            else:
                self.logger.warning(f"[FAIL] Required directory '{req_dir}' not found.")
                checks_passed = False

        self.logger.info("--- Diagnostics Summary ---")
        if checks_passed:
            self.logger.info("All basic diagnostic checks passed.")
        else:
            self.logger.warning("Some diagnostic checks failed. Please review the logs.")
        self.logger.info("---------------------------")

    def manage_connectors(self):
        """Finds and provides options for managing connector scripts."""
        self.logger.info("--- Connector Management ---")
        connectors = self.connectors.find_connectors(os.getcwd())
        
        if not connectors:
            self.logger.warning("No connector scripts found in the project.")
            return

        self.logger.info(f"Found {len(connectors)} connector scripts.")
        
        while True:
            self.logger.info("\nConnector Options:")
            self.logger.info("  1. List all connector paths")
            self.logger.info("  2. Return to main menu")
            choice = input("Select an option: ").strip()

            if choice == '1':
                self.logger.info("--- Listing All Connector Paths ---")
                for i, conn_path in enumerate(connectors):
                    self.logger.info(f"  {i+1:03d}: {conn_path}")
                self.logger.info("------------------------------------")
            elif choice == '2':
                break
            else:
                self.logger.warning("Invalid choice. Please try again.")
        self.logger.info("Exiting Connector Management.")

    def exit_program(self):
        """Exits the program."""
        self.logger.info("Exiting Polar Bear Master System.")
        sys.exit()

    def invalid_task(self):
        """Handles an invalid task selection."""
        self.logger.error("Selected task does not exist.")


class PolarBearMaster:
    """
    The master script to rule them all.
    """
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        self.logging_manager = LoggingManager()
        self.logger = self.logging_manager.get_logger()
        self.config = ConfigurationManager(self.logger)
        self.deps = DependencyManager(self.logger)
        self.connectors = ConnectorManager(self.logger)
        self.analyzer = ScriptAnalyzer(self.logger)
        self.task_manager = TaskManager(self.logger, self.config, self.analyzer, self.connectors)

    def run(self):
        """Main execution method."""
        self.logger.info("Initializing Polar Bear Master System...")
        
        # Load configuration
        self.config.load_config(interactive=not self.test_mode)

        # Check and install dependencies
        requirements_files = [Path('polar_bear_requirements.txt'), Path('requirements_web.txt')]
        self.deps.check_and_install(requirements_files)

        if self.test_mode:
            self.run_tests()
        else:
            self.run_interactive_mode()

    def run_tests(self):
        """Runs a non-interactive suite of tests and diagnostics."""
        self.logger.info("--- Running Non-Interactive Test Suite ---")
        self.task_manager.run_diagnostics()
        self.task_manager.analyze_project()
        self.task_manager.manage_connectors()
        self.logger.info("--- Non-Interactive Test Suite Complete ---")

    def run_interactive_mode(self):
        """Runs the main interactive functionality of the script."""
        self.logger.info("Entering interactive mode...")
        self.logger.info("Welcome to the Polar Bear Master Control System!")
        while True:
            self.task_manager.display_menu()
            try:
                task = self.task_manager.get_task()
                if task:
                    task()
                else:
                    self.logger.warning("Invalid task selected. Please try again.")
            except SystemExit:
                self.logger.info("Exiting Polar Bear Master System.")
                raise # Re-raise the exception to exit the program
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        self.logger.info("Interactive mode finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polar Bear Master Control Script")
    parser.add_argument(
        "-test",
        "--test",
        action="store_true",
        help="Run in non-interactive test mode.",
    )
    args = parser.parse_args()

    master_system = PolarBearMaster(test_mode=args.test)
    master_system.run()
