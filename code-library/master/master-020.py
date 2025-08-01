#!/usr/bin/env python3
"""
Neural Framework Core - Merged Module
This module combines all functionality from the neural_framework core components.
"""

import ast
import json
import os
import sys
import socket
import threading
import time
import subprocess
import importlib
import importlib.util
import logging
from datetime import datetime
from pathlib import Path
from setuptools import find_packages

# Try to import stdlib_list, provide fallback if not available
try:
    from stdlib_list import stdlib_list
except ImportError:
    stdlib_list = None


# ===== Logger Component =====
LOG_DIR = "neural_framework/logs"

def setup_logger(name="NeuralFramework", log_level=logging.DEBUG):
    """
    Sets up a logger that logs to both the console and a file.
    """
    abs_log_dir = os.path.abspath(LOG_DIR)

    if not os.path.exists(abs_log_dir):
        os.makedirs(abs_log_dir)

    log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".log"
    log_filepath = os.path.join(abs_log_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# Global logger instance
log = setup_logger()


# ===== Config Manager Component =====
class ConfigManager:
    def __init__(self, config_name="config.json"):
        self.config_path = os.path.join("neural_framework/config", config_name)
        self.config = {}
        self.load_config()

    def load_config(self):
        """Loads the configuration from a JSON file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                log.info(f"Configuration loaded from {self.config_path}")
            else:
                log.info("No configuration file found. A new one will be created.")
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        except Exception as e:
            log.error(f"Could not load configuration: {e}")

    def get(self, key, default=None, prompt=""):
        """
        Gets a value from the configuration.
        If the key is not found, it prompts the user for the value.
        """
        value = self.config.get(key)
        if value is not None:
            return value
        
        if prompt:
            try:
                value = input(prompt)
                self.config[key] = value
                self.save_config()
                return value
            except EOFError:
                log.warning("EOFError received, cannot prompt for input. Using default.")
                return default

        return default

    def set(self, key, value):
        """Sets a configuration value and saves it."""
        self.config[key] = value
        self.save_config()

    def save_config(self):
        """Saves the configuration to a JSON file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4, sort_keys=True)
            log.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            log.error(f"Could not save configuration: {e}")


# ===== Module Analyzer Component =====
class ModuleAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.functions = []
        self.classes = []

    def analyze(self):
        """
        Analyzes the Python file to extract information about its functions and classes.
        """
        log.info(f"Analyzing module: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=self.file_path)
                
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.FunctionDef):
                        self.functions.append(self._extract_function_info(node))
                    elif isinstance(node, ast.ClassDef):
                        self.classes.append(self._extract_class_info(node))
        except Exception as e:
            log.error(f"Could not analyze module {self.file_path}: {e}")
        
        return {
            "file_path": self.file_path,
            "module_name": os.path.splitext(os.path.basename(self.file_path))[0],
            "functions": self.functions,
            "classes": self.classes
        }

    def _extract_function_info(self, node):
        """Extracts information from a FunctionDef node."""
        return {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "defaults": len(node.args.defaults),
            "docstring": ast.get_docstring(node) or "No docstring."
        }

    def _extract_class_info(self, node):
        """Extracts information from a ClassDef node."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._extract_function_info(item))
        
        return {
            "name": node.name,
            "methods": methods,
            "docstring": ast.get_docstring(node) or "No docstring."
        }


# ===== Dependency Manager Component =====
class DependencyManager:
    def __init__(self, project_path):
        self.project_path = os.path.abspath(project_path)
        self.dependencies = set()
        
        if stdlib_list:
            try:
                self.std_libs = set(stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}"))
            except Exception:
                self.std_libs = set(stdlib_list("3.11"))
        else:
            # Fallback list of common standard library modules
            self.std_libs = {
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 
                'datetime', 'email', 'functools', 'hashlib', 'http', 'importlib',
                'io', 'itertools', 'json', 'logging', 'math', 'os', 'pathlib',
                're', 'shutil', 'socket', 'subprocess', 'sys', 'threading', 
                'time', 'typing', 'unittest', 'urllib', 'uuid', 'warnings'
            }
            
        self.local_modules = self._find_local_modules()
        log.info(f"Initialized DependencyManager for {self.project_path}")
        log.debug(f"Found local modules: {self.local_modules}")

    def _find_local_modules(self):
        """Finds all .py files and directories with __init__.py in the project path."""
        local_modules = set()
        for root, dirs, files in os.walk(self.project_path):
            if 'neural_framework' in dirs:
                dirs.remove('neural_framework')
            for file in files:
                if file.endswith(".py"):
                    local_modules.add(os.path.splitext(file)[0])
            for d in dirs:
                if os.path.exists(os.path.join(root, d, "__init__.py")):
                    local_modules.add(d)
        return local_modules

    def analyze(self):
        """
        Analyzes a project's dependencies, prioritizing requirements.txt.
        """
        log.info(f"Analyzing dependencies for project: {os.path.basename(self.project_path)}")
        if not self._analyze_from_requirements():
            self._analyze_from_source()
        return self

    def _analyze_from_requirements(self):
        """Parses a requirements.txt file if it exists."""
        requirements_file = os.path.join(self.project_path, "requirements.txt")
        if os.path.exists(requirements_file):
            log.info(f"Found requirements.txt for {self.project_path}")
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].strip()
                        self.dependencies.add(dep_name)
            return True
        return False

    def _analyze_from_source(self):
        """Analyzes python source files to find dependencies."""
        log.info(f"No requirements.txt found. Analyzing source files in {self.project_path}")
        for root, dirs, files in os.walk(self.project_path):
            if 'neural_framework' in dirs:
                dirs.remove('neural_framework')
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            tree = ast.parse(content, filename=file_path)
                            for node in ast.walk(tree):
                                if isinstance(node, ast.Import):
                                    for alias in node.names:
                                        self._add_dependency(alias.name)
                                elif isinstance(node, ast.ImportFrom):
                                    if node.module and node.level == 0:
                                        self._add_dependency(node.module)
                    except Exception as e:
                        log.error(f"Could not analyze file {file_path}: {e}")

    def _add_dependency(self, full_import):
        """Adds a dependency if it's not a standard or local module."""
        top_level_module = full_import.split('.')[0]
        if (top_level_module and 
            top_level_module not in self.std_libs and
            top_level_module not in self.local_modules and
            not top_level_module.startswith('_')):
            self.dependencies.add(top_level_module)

    def get_found_dependencies(self):
        return list(self.dependencies)


# ===== Module Loader Component =====
class ModuleLoader:
    def __init__(self, modules_to_load):
        self.modules_to_load = modules_to_load
        self.module_registry = {}

    def load_modules(self):
        log.info(f"Attempting to import {len(self.modules_to_load)} discovered modules...")
        for module_name, file_path in self.modules_to_load.items():
            module = None
            try:
                module = importlib.import_module(module_name)
                log.info(f"Successfully imported installed module: {module_name}")
            except ImportError:
                log.warning(f"Module '{module_name}' not found via standard import. Attempting to load from file.")
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec:
                        module = importlib.util.module_from_spec(spec)
                        sys.path.insert(0, os.path.dirname(file_path))
                        spec.loader.exec_module(module)
                        sys.path.pop(0)
                        log.info(f"Successfully loaded module '{module_name}' directly from file.")
                    else:
                        log.error(f"Could not create a module spec for {file_path}")
                        continue
                except Exception as e:
                    log.error(f"Failed to load module '{module_name}' from file: {e}")
                    continue
            except Exception as e:
                log.error(f"An unexpected error occurred while importing module {module_name}: {e}")
                continue

            if module:
                try:
                    analyzer = ModuleAnalyzer(file_path)
                    analysis_result = analyzer.analyze()
                    self.module_registry[module_name] = {
                        "analysis": analysis_result,
                        "module": module,
                        "file_path": file_path
                    }
                    log.debug(f"Successfully registered module: {module_name}")
                except Exception as e:
                    log.error(f"Failed to analyze module {module_name} after loading: {e}")

        log.info(f"Module loading complete. Successfully loaded {len(self.module_registry)} modules.")
        return self.module_registry


# ===== Hivemind Connector Component =====
class HivemindConnector:
    def __init__(self):
        self.connector_id = "579489f1"
        self.directory = Path("/home/jarvis/Documents/GitHub/polar-bear/modules/computer-vision/neural_framework/core")
        self.port = 10106
        self.parent_port = 10101
        self.depth = 4
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {}
        self.scripts_in_directory = {}
        
        self.logger = logging.getLogger(f"Connector_{self.connector_id}")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def start(self):
        """Start the connector"""
        self.logger.info(f"Starting connector {self.connector_id} on port {self.port}")
        
        self.listen_thread = threading.Thread(target=self.listen_for_connections)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        self.connect_to_parent()
        self.scan_directory()
        self.heartbeat_loop()
        
    def listen_for_connections(self):
        """Listen for incoming connections"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('localhost', self.port))
        self.socket.listen(5)
        
        while self.running:
            try:
                conn, addr = self.socket.accept()
                threading.Thread(target=self.handle_connection, args=(conn,)).start()
            except:
                pass
                
    def handle_connection(self, conn):
        """Handle incoming connection"""
        try:
            data = conn.recv(4096).decode()
            message = json.loads(data)
            
            response = self.process_message(message)
            conn.send(json.dumps(response).encode())
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
        finally:
            conn.close()
            
    def process_message(self, message):
        """Process incoming message"""
        cmd = message.get('command')
        
        if cmd == 'status':
            return {
                'status': 'active',
                'connector_id': self.connector_id,
                'directory': str(self.directory),
                'depth': self.depth,
                'scripts': len(self.scripts_in_directory),
                'children': len(self.child_connectors)
            }
        elif cmd == 'scan':
            self.scan_directory()
            return {'status': 'scan_complete', 'scripts': list(self.scripts_in_directory.keys())}
        elif cmd == 'execute':
            script = message.get('script')
            if script in self.scripts_in_directory:
                return self.execute_script(script)
            return {'error': 'Script not found'}
        elif cmd == 'troubleshoot':
            return self.troubleshoot_connections()
        else:
            return {'error': 'Unknown command'}
            
    def connect_to_parent(self):
        """Connect to parent connector"""
        try:
            self.parent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.parent_socket.connect(('localhost', self.parent_port))
            
            register_msg = {
                'command': 'register',
                'connector_id': self.connector_id,
                'port': self.port,
                'directory': str(self.directory)
            }
            self.parent_socket.send(json.dumps(register_msg).encode())
            self.logger.info(f"Connected to parent on port {self.parent_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to parent: {e}")
            
    def scan_directory(self):
        """Scan directory for Python scripts and subdirectories"""
        self.scripts_in_directory.clear()
        
        try:
            for item in self.directory.iterdir():
                if item.is_file() and item.suffix == '.py' and item.name != 'hivemind_connector.py':
                    self.scripts_in_directory[item.name] = str(item)
                elif item.is_dir() and not self.should_skip_directory(item):
                    child_connector = item / 'hivemind_connector.py'
                    if child_connector.exists():
                        self.child_connectors[item.name] = str(child_connector)
                        
            self.logger.info(f"Found {len(self.scripts_in_directory)} scripts and {len(self.child_connectors)} child connectors")
        except Exception as e:
            self.logger.error(f"Error scanning directory: {e}")
            
    def should_skip_directory(self, path):
        """Check if directory should be skipped"""
        skip_dirs = {
            'venv', 'env', '.env', '__pycache__', '.git', 
            'node_modules', '.venv', 'virtualenv', '.tox',
            'build', 'dist', '.pytest_cache', '.mypy_cache'
        }
        return path.name.startswith('.') or path.name.lower() in skip_dirs
        
    def execute_script(self, script_name):
        """Execute a script in the directory"""
        if script_name not in self.scripts_in_directory:
            return {'error': 'Script not found'}
            
        try:
            result = subprocess.run(
                [sys.executable, self.scripts_in_directory[script_name]],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                'status': 'executed',
                'script': script_name,
                'returncode': result.returncode,
                'stdout': result.stdout[-1000:],
                'stderr': result.stderr[-1000:]
            }
        except Exception as e:
            return {'error': f'Execution failed: {str(e)}'}
            
    def troubleshoot_connections(self):
        """Troubleshoot connector connections"""
        issues = []
        
        if not self.parent_socket:
            issues.append("Not connected to parent")
        
        if not self.socket:
            issues.append("Not listening for connections")
            
        for name, path in self.child_connectors.items():
            if not Path(path).exists():
                issues.append(f"Child connector missing: {name}")
                
        return {
            'connector_id': self.connector_id,
            'issues': issues,
            'healthy': len(issues) == 0
        }
        
    def heartbeat_loop(self):
        """Send heartbeat to parent"""
        while self.running:
            try:
                if self.parent_socket:
                    heartbeat = {
                        'command': 'heartbeat',
                        'connector_id': self.connector_id,
                        'timestamp': time.time()
                    }
                    self.parent_socket.send(json.dumps(heartbeat).encode())
            except:
                self.connect_to_parent()
                
            time.sleep(30)


# ===== Simple Connector Component =====
def run_simple_connector():
    """Main function for the connector script."""
    LOG_FILE = "connector.log"
    
    logger = logging.getLogger(os.path.abspath(__file__))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    
    try:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except (IOError, OSError) as e:
        print(f"Could not write to log file {LOG_FILE}: {e}", file=sys.stderr)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"--- Connector Script Initialized in {os.getcwd()} ---")
    logger.info(f"This script is responsible for connecting the modules in this directory to the main control script.")
    
    try:
        files = os.listdir()
        if files:
            logger.info("Files in this directory:")
            for file in files:
                logger.info(f"- {file}")
        else:
            logger.info("No files found in this directory.")
    except OSError as e:
        logger.error(f"Error listing files: {e}")
    
    logger.info("Connector script finished.")


# ===== Setup Generator Component =====
def generate_setup_py(project_path, project_name):
    """
    Generates a basic setup.py file for a given project.
    """
    setup_content = f"""
from setuptools import setup, find_packages
import os

# Basic information
NAME = "{project_name}"
VERSION = "0.1.0"
DESCRIPTION = "A project dynamically packaged by the Neural Framework."

# Find requirements
requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
install_requires = []
if os.path.exists(requirements_file):
    with open(requirements_file, 'r') as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6',
)
"""
    setup_file_path = os.path.join(project_path, "setup.py")
    try:
        if not os.path.exists(setup_file_path):
            with open(setup_file_path, 'w') as f:
                f.write(setup_content)
            return True
        return False
    except Exception as e:
        print(f"Error generating setup.py for {project_name}: {e}")
        return False


# ===== Main Entry Point =====
if __name__ == '__main__':
    print("Neural Framework Core - Merged Module")
    print("=" * 50)
    print("This module combines all neural framework core functionality.")
    print("\nAvailable components:")
    print("1. Logger (log)")
    print("2. ConfigManager")
    print("3. ModuleAnalyzer")
    print("4. DependencyManager")
    print("5. ModuleLoader")
    print("6. HivemindConnector")
    print("7. Simple Connector (run_simple_connector)")
    print("8. Setup Generator (generate_setup_py)")
    print("\nRunning basic tests...")
    
    # Test logger
    log.info("Logger is working correctly")
    
    # Test config manager
    config = ConfigManager("test_merged.json")
    config.set("test_key", "test_value")
    log.info(f"ConfigManager test: {config.get('test_key')}")
    
    # Clean up test config
    try:
        os.remove(os.path.join("neural_framework/config", "test_merged.json"))
    except:
        pass
    
    print("\nAll components loaded successfully!")