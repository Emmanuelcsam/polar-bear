
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
    def __init__(self, logger, analyzer, connectors):
        self.logger = logger
        self.analyzer = analyzer
        self.connectors = connectors
        self.tasks = {
            "1": ("Analyze Project", self.analyze_project),
            "2": ("Run Diagnostics", self.run_diagnostics),
            "3": ("Manage Connectors", self.manage_connectors),
            "4": ("Exit", sys.exit)
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
            if choice in self.tasks:
                return self.tasks[choice][1]
            else:
                self.logger.warning("Invalid choice. Please try again.")

    def analyze_project(self):
        """Analyzes all Python scripts in the project."""
        self.logger.info("Analyzing project...")
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.analyzer.analyze_script, p) for p in Path.cwd().rglob("*.py")]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error during script analysis: {e}")
        self.logger.info("Project analysis complete.")

    def run_diagnostics(self):
        """Runs diagnostic tests."""
        self.logger.info("Running diagnostics...")
        # Placeholder for diagnostic logic
        self.logger.info("Diagnostics complete.")

    def manage_connectors(self):
        """Manages connector scripts."""
        self.logger.info("Managing connectors...")
        self.connectors.find_connectors(os.getcwd())
        self.logger.info(f"Found {len(self.connectors.connectors)} connectors.")
        # Placeholder for connector management logic
        self.logger.info("Connector management complete.")


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
        self.task_manager = TaskManager(self.logger, self.analyzer, self.connectors)

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
        """Runs all unit tests and other automated checks."""
        self.logger.info("Running in test mode...")
        # Here we will call the unit tests
        self.logger.info("Test mode finished.")

    def run_interactive_mode(self):
        """Runs the main interactive functionality of the script."""
        self.logger.info("Running in interactive mode...")
        while True:
            self.task_manager.display_menu()
            task = self.task_manager.get_task()
            task()


    def run_tests(self):
        """Runs all unit tests and other automated checks."""
        self.logger.info("Running in test mode...")
        # Here we will call the unit tests
        self.logger.info("Test mode finished.")

    def run_interactive_mode(self):
        """Runs the main interactive functionality of the script."""
        self.logger.info("Running in interactive mode...")
        # Main application logic will go here
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
