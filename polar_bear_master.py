
import os
import sys
import subprocess
import logging
import json
import time
import argparse
from pathlib import Path
import importlib.util
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

class UnifiedConnectorInterface:
    """Unified interface for all connector communications."""
    
    def __init__(self, logger):
        self.logger = logger
        self.connectors: Dict[str, Dict[str, Any]] = {}
        self.active_connections: Dict[str, socket.socket] = {}
        self.lock = threading.Lock()
        
    def register_connector(self, connector_id: str, path: str, port: Optional[int] = None):
        """Register a connector in the system."""
        with self.lock:
            self.connectors[connector_id] = {
                'path': path,
                'port': port,
                'status': 'registered',
                'last_heartbeat': time.time(),
                'directory': str(Path(path).parent)
            }
    
    def send_command(self, connector_id: str, command: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Send a command to a specific connector."""
        if connector_id not in self.connectors:
            return {'error': 'Connector not found'}
        
        connector = self.connectors[connector_id]
        if not connector.get('port'):
            return {'error': 'Connector port not configured'}
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(('localhost', connector['port']))
            
            message = {
                'command': command,
                **kwargs
            }
            sock.send(json.dumps(message).encode())
            
            response = sock.recv(4096).decode()
            sock.close()
            
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error communicating with connector {connector_id}: {e}")
            return {'error': str(e)}
    
    def broadcast_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """Broadcast a command to all registered connectors."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_connector = {
                executor.submit(self.send_command, conn_id, command, **kwargs): conn_id 
                for conn_id in self.connectors.keys()
            }
            
            for future in as_completed(future_to_connector):
                conn_id = future_to_connector[future]
                try:
                    result = future.result()
                    results[conn_id] = result
                except Exception as e:
                    results[conn_id] = {'error': str(e)}
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the overall system status from all connectors."""
        status_results = self.broadcast_command('status')
        
        active_count = sum(1 for r in status_results.values() 
                          if r.get('status') == 'active')
        
        return {
            'total_connectors': len(self.connectors),
            'active_connectors': active_count,
            'inactive_connectors': len(self.connectors) - active_count,
            'connector_details': status_results
        }

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
        self.active_connections = {}
        self.connector_status = {}
        self.lock = threading.Lock()

    def find_connectors(self, root_path):
        """Finds all connector scripts in the project."""
        self.logger.info(f"Searching for connectors in {root_path}...")
        
        # Find all connector.py files, excluding certain directories
        connectors = []
        skip_dirs = {'venv', 'env', '.env', '__pycache__', '.git', 
                     'node_modules', '.venv', 'virtualenv', '.tox',
                     'build', 'dist', '.pytest_cache', '.mypy_cache',
                     'test_env', '.egg-info'}
        
        for path in Path(root_path).rglob("connector.py"):
            # Skip if any parent directory is in skip_dirs
            if not any(part in skip_dirs for part in path.parts):
                connectors.append(path)
        
        self.connectors = connectors
        self.logger.info(f"Found {len(self.connectors)} connector scripts.")
        return self.connectors
    
    def test_connector(self, connector_path):
        """Test a single connector by trying to connect to it."""
        try:
            # Try to import and check if it has the expected interface
            spec = importlib.util.spec_from_file_location("connector", connector_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check if it has expected functions/classes
                has_interface = False
                if hasattr(module, 'ConnectorInterface'):
                    has_interface = True
                elif hasattr(module, 'main'):
                    has_interface = True
                elif hasattr(module, 'HivemindConnector'):
                    has_interface = True
                
                return {
                    'path': str(connector_path),
                    'status': 'available',
                    'has_interface': has_interface,
                    'directory': str(connector_path.parent)
                }
        except Exception as e:
            return {
                'path': str(connector_path),
                'status': 'error',
                'error': str(e),
                'directory': str(connector_path.parent)
            }
    
    def connect_to_all(self):
        """Attempt to connect to all discovered connectors."""
        self.logger.info("Testing connectivity to all connectors...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_connector = {
                executor.submit(self.test_connector, conn): conn 
                for conn in self.connectors
            }
            
            for future in as_completed(future_to_connector):
                connector_path = future_to_connector[future]
                try:
                    result = future.result()
                    with self.lock:
                        self.connector_status[str(connector_path)] = result
                    
                    if result['status'] == 'available':
                        self.logger.debug(f"✓ Connector available: {connector_path.parent.name}/{connector_path.name}")
                    else:
                        self.logger.debug(f"✗ Connector error: {connector_path}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    self.logger.error(f"Failed to test connector {connector_path}: {e}")
        
        # Summary
        available = sum(1 for s in self.connector_status.values() if s['status'] == 'available')
        self.logger.info(f"Connector connectivity test complete: {available}/{len(self.connectors)} available")
        
        return self.connector_status
    
    def execute_connector_command(self, connector_path, command, **kwargs):
        """Execute a command on a specific connector."""
        try:
            # Import the connector module
            spec = importlib.util.spec_from_file_location("connector", connector_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Try different interfaces
                if hasattr(module, 'main') and callable(module.main):
                    # Change to connector's directory before execution
                    original_dir = os.getcwd()
                    os.chdir(Path(connector_path).parent)
                    try:
                        result = module.main()
                        return {'status': 'success', 'result': result}
                    finally:
                        os.chdir(original_dir)
                
                return {'status': 'error', 'error': 'No executable interface found'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def get_connector_info(self):
        """Get detailed information about all connectors."""
        info = {
            'total': len(self.connectors),
            'available': sum(1 for s in self.connector_status.values() if s['status'] == 'available'),
            'by_directory': {}
        }
        
        # Group by parent directory
        for conn_path, status in self.connector_status.items():
            parent = Path(conn_path).parent.name
            if parent not in info['by_directory']:
                info['by_directory'][parent] = []
            info['by_directory'][parent].append({
                'path': conn_path,
                'status': status['status'],
                'has_interface': status.get('has_interface', False)
            })
        
        return info

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
            "4": ("Run Full System Scan", "run_full_system_scan"),
            "5": ("Exit", "exit_program")
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
        
        # Find all connectors
        connectors = self.connectors.find_connectors(os.getcwd())
        
        if not connectors:
            self.logger.warning("No connector scripts found in the project.")
            return

        self.logger.info(f"Found {len(connectors)} connector scripts.")
        
        # Test connectivity to all connectors
        self.connectors.connect_to_all()
        
        while True:
            self.logger.info("\nConnector Options:")
            self.logger.info("  1. List all connector paths")
            self.logger.info("  2. Show connector status summary")
            self.logger.info("  3. Execute a connector")
            self.logger.info("  4. Refresh connector status")
            self.logger.info("  5. Return to main menu")
            choice = input("Select an option: ").strip()

            if choice == '1':
                self.logger.info("--- Listing All Connector Paths ---")
                for i, conn_path in enumerate(connectors):
                    status = self.connectors.connector_status.get(str(conn_path), {})
                    status_icon = "✓" if status.get('status') == 'available' else "✗"
                    self.logger.info(f"  {status_icon} {i+1:03d}: {conn_path}")
                self.logger.info("------------------------------------")
                
            elif choice == '2':
                info = self.connectors.get_connector_info()
                self.logger.info("\n--- Connector Status Summary ---")
                self.logger.info(f"Total Connectors: {info['total']}")
                self.logger.info(f"Available: {info['available']}")
                self.logger.info(f"Errors: {info['total'] - info['available']}")
                
                self.logger.info("\nBy Directory:")
                for dir_name, conns in sorted(info['by_directory'].items()):
                    available = sum(1 for c in conns if c['status'] == 'available')
                    self.logger.info(f"  {dir_name}: {available}/{len(conns)} available")
                self.logger.info("--------------------------------")
                
            elif choice == '3':
                # List connectors with numbers
                available_connectors = [(i, conn) for i, conn in enumerate(connectors) 
                                       if self.connectors.connector_status.get(str(conn), {}).get('status') == 'available']
                
                if not available_connectors:
                    self.logger.warning("No available connectors to execute.")
                    continue
                
                self.logger.info("\nAvailable Connectors:")
                for idx, (i, conn) in enumerate(available_connectors):
                    self.logger.info(f"  {idx+1}. {conn.parent.name}/{conn.name}")
                
                try:
                    selection = int(input("Select connector to execute (number): ")) - 1
                    if 0 <= selection < len(available_connectors):
                        _, selected_conn = available_connectors[selection]
                        self.logger.info(f"Executing {selected_conn}...")
                        result = self.connectors.execute_connector_command(selected_conn, 'execute')
                        if result['status'] == 'success':
                            self.logger.info("Execution completed successfully.")
                        else:
                            self.logger.error(f"Execution failed: {result.get('error', 'Unknown error')}")
                    else:
                        self.logger.warning("Invalid selection.")
                except ValueError:
                    self.logger.warning("Please enter a valid number.")
                    
            elif choice == '4':
                self.logger.info("Refreshing connector status...")
                self.connectors.connect_to_all()
                
            elif choice == '5':
                break
            else:
                self.logger.warning("Invalid choice. Please try again.")
                
        self.logger.info("Exiting Connector Management.")
    
    def run_full_system_scan(self):
        """Runs a comprehensive scan of all connectors and displays system status."""
        self.logger.info("--- Running Full System Scan ---")
        
        # Phase 1: Discover all connectors
        self.logger.info("Phase 1: Discovering connectors...")
        connectors = self.connectors.find_connectors(os.getcwd())
        
        if not connectors:
            self.logger.warning("No connectors found in the system.")
            return
        
        # Phase 2: Test connectivity
        self.logger.info("Phase 2: Testing connectivity to all connectors...")
        self.connectors.connect_to_all()
        
        # Phase 3: Gather system information
        self.logger.info("Phase 3: Gathering system information...")
        info = self.connectors.get_connector_info()
        
        # Phase 4: Display comprehensive report
        self.logger.info("\n=== POLAR BEAR SYSTEM REPORT ===")
        self.logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total Connectors Found: {info['total']}")
        self.logger.info(f"Active Connectors: {info['available']}")
        self.logger.info(f"Failed Connectors: {info['total'] - info['available']}")
        self.logger.info(f"Success Rate: {(info['available'] / info['total'] * 100) if info['total'] > 0 else 0:.1f}%")
        
        # Directory breakdown
        self.logger.info("\n--- Connector Distribution by Directory ---")
        dir_stats = []
        for dir_name, conns in sorted(info['by_directory'].items()):
            available = sum(1 for c in conns if c['status'] == 'available')
            total = len(conns)
            percentage = (available / total * 100) if total > 0 else 0
            dir_stats.append((dir_name, available, total, percentage))
        
        # Sort by percentage descending
        dir_stats.sort(key=lambda x: x[3], reverse=True)
        
        for dir_name, available, total, percentage in dir_stats[:20]:  # Top 20 directories
            bar_length = int(percentage / 5)  # 20 character bar max
            bar = '█' * bar_length + '░' * (20 - bar_length)
            self.logger.info(f"  {dir_name:30s} [{bar}] {available:3d}/{total:3d} ({percentage:5.1f}%)")
        
        if len(dir_stats) > 20:
            self.logger.info(f"  ... and {len(dir_stats) - 20} more directories")
        
        # Failed connectors
        failed_connectors = [(path, status) for path, status in self.connectors.connector_status.items() 
                            if status['status'] != 'available']
        
        if failed_connectors:
            self.logger.info("\n--- Failed Connectors ---")
            for path, status in failed_connectors[:10]:  # Show first 10
                error = status.get('error', 'Unknown error')
                self.logger.info(f"  ✗ {Path(path).parent.name}/{Path(path).name}")
                self.logger.info(f"    Error: {error[:100]}{'...' if len(error) > 100 else ''}")
            
            if len(failed_connectors) > 10:
                self.logger.info(f"  ... and {len(failed_connectors) - 10} more failed connectors")
        
        # Summary
        self.logger.info("\n--- System Health Summary ---")
        if info['available'] == info['total']:
            self.logger.info("✓ All connectors are operational!")
        elif info['available'] > info['total'] * 0.8:
            self.logger.info("✓ System is mostly operational (>80% connectors active)")
        elif info['available'] > info['total'] * 0.5:
            self.logger.info("⚠ System is partially operational (>50% connectors active)")
        else:
            self.logger.info("✗ System has significant issues (<50% connectors active)")
        
        self.logger.info("\n=== END OF SYSTEM REPORT ===")
        
        # Ask if user wants to save the report
        save_report = input("\nSave detailed system report to file? (y/n): ").lower()
        if save_report == 'y':
            report_path = f"polar_bear_system_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
            report_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'summary': info,
                'connector_status': self.connectors.connector_status,
                'failed_connectors': failed_connectors
            }
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=4)
            self.logger.info(f"Detailed report saved to {report_path}")

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
        self.unified_interface = UnifiedConnectorInterface(self.logger)
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
        
        # Initialize connector system
        self.logger.info("Initializing connector system...")
        self.initialize_connectors()

        if self.test_mode:
            self.run_tests()
        else:
            self.run_interactive_mode()
    
    def initialize_connectors(self):
        """Initialize the connector system by discovering all connectors."""
        try:
            # Find all connectors
            connectors = self.connectors.find_connectors(os.getcwd())
            
            if connectors:
                self.logger.info(f"Discovered {len(connectors)} connectors in the system")
                
                # Optionally test connectivity in background
                if self.config.get('auto_test_connectors', True):
                    self.logger.info("Testing connector connectivity in background...")
                    thread = threading.Thread(target=self.connectors.connect_to_all)
                    thread.daemon = True
                    thread.start()
            else:
                self.logger.warning("No connectors found during initialization")
                
        except Exception as e:
            self.logger.error(f"Error initializing connector system: {e}")

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
