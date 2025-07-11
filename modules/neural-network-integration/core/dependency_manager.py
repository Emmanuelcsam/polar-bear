#!/usr/bin/env python3
"""
Ultra-Intelligent Dependency Management System
Automatically detects, installs, and manages all dependencies across the neural network
"""

import os
import sys
import ast
import json
import subprocess
import importlib.util
import importlib.metadata
import re
import logging
import time
import threading
import pathlib
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
import pkg_resources
import traceback


class DependencyManager:
    """Advanced dependency detection and installation system"""
    
    def __init__(self, log_file: str = "dependency_manager.log"):
        self.log_file = log_file
        self.detected_imports = set()
        self.installed_packages = set()
        self.failed_packages = set()
        self.version_constraints = {}
        self.system_packages = {
            'os', 'sys', 'time', 'datetime', 'json', 'math', 'random',
            'collections', 'itertools', 'functools', 'pathlib', 'typing',
            'subprocess', 'threading', 'multiprocessing', 'queue', 'io',
            're', 'string', 'copy', 'pickle', 'sqlite3', 'csv', 'xml',
            'logging', 'warnings', 'traceback', 'inspect', 'ast', 'imp',
            'importlib', 'pkgutil', 'pkg_resources', 'platform', 'locale',
            'glob', 'fnmatch', 'tempfile', 'shutil', 'stat', 'filecmp',
            'configparser', 'argparse', 'getopt', 'getpass', 'cmd', 'shlex'
        }
        self.package_mappings = {
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn',
            'skimage': 'scikit-image',
            'PIL': 'Pillow',
            'Image': 'Pillow',
            'yaml': 'PyYAML',
            'mpl_toolkits': 'matplotlib',
            'scipy.ndimage': 'scipy',
            'scipy.signal': 'scipy',
            'torchvision': 'torchvision',
            'tensorflow': 'tensorflow',
            'keras': 'tensorflow',
            'h5py': 'h5py',
            'pandas': 'pandas',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'bokeh': 'bokeh',
            'dash': 'dash',
            'flask': 'flask',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'requests': 'requests',
            'aiohttp': 'aiohttp',
            'websocket': 'websocket-client',
            'serial': 'pyserial',
            'redis': 'redis',
            'pymongo': 'pymongo',
            'sqlalchemy': 'sqlalchemy',
            'psycopg2': 'psycopg2-binary',
            'mysql': 'mysql-connector-python',
            'paho': 'paho-mqtt',
            'confluent_kafka': 'confluent-kafka',
            'celery': 'celery',
            'rq': 'rq',
            'schedule': 'schedule',
            'apscheduler': 'apscheduler',
            'click': 'click',
            'tqdm': 'tqdm',
            'colorama': 'colorama',
            'termcolor': 'termcolor',
            'rich': 'rich',
            'pytest': 'pytest',
            'unittest2': 'unittest2',
            'mock': 'mock',
            'faker': 'faker',
            'hypothesis': 'hypothesis',
            'coverage': 'coverage',
            'pylint': 'pylint',
            'flake8': 'flake8',
            'black': 'black',
            'mypy': 'mypy',
            'pre_commit': 'pre-commit'
        }
        
        self._setup_logging()
        self.log("Dependency Manager initialized")
    
    def _setup_logging(self):
        """Setup dual logging to file and console"""
        self.logger = logging.getLogger('DependencyManager')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler with color
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log(self, message: str, level: str = "INFO"):
        """Log message to both file and console"""
        getattr(self.logger, level.lower())(message)
        
    def scan_directory(self, directory: str) -> Set[str]:
        """Recursively scan directory for Python files and extract imports"""
        self.log(f"Scanning directory: {directory}")
        imports = set()
        
        for root, dirs, files in os.walk(directory):
            # Skip virtual environments and common ignored directories
            dirs[:] = [d for d in dirs if d not in {
                'venv', '__pycache__', '.git', 'node_modules', 
                '.pytest_cache', '.mypy_cache', 'build', 'dist'
            }]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        file_imports = self._extract_imports_from_file(filepath)
                        imports.update(file_imports)
                    except Exception as e:
                        self.log(f"Error scanning {filepath}: {str(e)}", "WARNING")
        
        self.detected_imports = imports
        self.log(f"Found {len(imports)} unique imports")
        return imports
    
    def _extract_imports_from_file(self, filepath: str) -> Set[str]:
        """Extract all imports from a Python file using AST"""
        imports = set()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filepath)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
            
            # Also check for dynamic imports in code
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find __import__ calls
            import_pattern = r'__import__\s*\(\s*[\'"]([^\'"\)]+)[\'"]'
            dynamic_imports = re.findall(import_pattern, content)
            imports.update(i.split('.')[0] for i in dynamic_imports)
            
            # Find importlib.import_module calls
            importlib_pattern = r'import_module\s*\(\s*[\'"]([^\'"\)]+)[\'"]'
            importlib_imports = re.findall(importlib_pattern, content)
            imports.update(i.split('.')[0] for i in importlib_imports)
            
        except Exception as e:
            self.log(f"Failed to parse {filepath}: {str(e)}", "ERROR")
        
        return imports
    
    def check_installed_packages(self) -> Dict[str, str]:
        """Get all currently installed packages and their versions"""
        installed = {}
        try:
            for dist in pkg_resources.working_set:
                installed[dist.key] = dist.version
        except Exception as e:
            self.log(f"Error checking installed packages: {str(e)}", "ERROR")
        
        return installed
    
    def is_package_installed(self, package: str) -> bool:
        """Check if a package is installed"""
        try:
            importlib.import_module(package)
            return True
        except ImportError:
            # Try with package mapping
            if package in self.package_mappings:
                pip_name = self.package_mappings[package]
                try:
                    pkg_resources.get_distribution(pip_name)
                    return True
                except:
                    pass
            return False
    
    def get_package_version(self, package: str) -> Optional[str]:
        """Get the version of an installed package"""
        try:
            if package in self.package_mappings:
                package = self.package_mappings[package]
            return pkg_resources.get_distribution(package).version
        except:
            return None
    
    def install_package(self, package: str, upgrade: bool = True) -> bool:
        """Install a package using pip"""
        pip_name = self.package_mappings.get(package, package)
        
        self.log(f"Installing package: {pip_name}")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.append(pip_name)
            
            # Run pip with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log(f"  {line.strip()}", "DEBUG")
            
            process.wait()
            
            if process.returncode == 0:
                self.installed_packages.add(package)
                self.log(f"Successfully installed {pip_name}", "INFO")
                return True
            else:
                self.failed_packages.add(package)
                self.log(f"Failed to install {pip_name}", "ERROR")
                return False
                
        except Exception as e:
            self.failed_packages.add(package)
            self.log(f"Exception installing {pip_name}: {str(e)}", "ERROR")
            return False
    
    def analyze_requirements_files(self, directory: str) -> Dict[str, str]:
        """Analyze requirements.txt, setup.py, pyproject.toml files"""
        requirements = {}
        
        # Check requirements.txt
        req_files = ['requirements.txt', 'requirements-dev.txt', 'requirements-test.txt']
        for req_file in req_files:
            filepath = os.path.join(directory, req_file)
            if os.path.exists(filepath):
                self.log(f"Analyzing {req_file}")
                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Parse requirement specifier
                            match = re.match(r'^([a-zA-Z0-9\-_]+)(.*)$', line)
                            if match:
                                pkg_name = match.group(1)
                                version_spec = match.group(2).strip()
                                requirements[pkg_name] = version_spec
        
        # Check setup.py
        setup_file = os.path.join(directory, 'setup.py')
        if os.path.exists(setup_file):
            self.log("Analyzing setup.py")
            try:
                with open(setup_file, 'r') as f:
                    content = f.read()
                # Extract install_requires
                install_req_match = re.search(
                    r'install_requires\s*=\s*\[(.*?)\]', 
                    content, 
                    re.DOTALL
                )
                if install_req_match:
                    req_str = install_req_match.group(1)
                    req_list = re.findall(r'[\'"]([^\'"]+)[\'"]', req_str)
                    for req in req_list:
                        match = re.match(r'^([a-zA-Z0-9\-_]+)(.*)$', req)
                        if match:
                            requirements[match.group(1)] = match.group(2).strip()
            except Exception as e:
                self.log(f"Error parsing setup.py: {str(e)}", "WARNING")
        
        # Check pyproject.toml
        pyproject_file = os.path.join(directory, 'pyproject.toml')
        if os.path.exists(pyproject_file):
            self.log("Analyzing pyproject.toml")
            try:
                import toml
                with open(pyproject_file, 'r') as f:
                    data = toml.load(f)
                deps = data.get('tool', {}).get('poetry', {}).get('dependencies', {})
                for pkg, version in deps.items():
                    if pkg != 'python':
                        requirements[pkg] = version if isinstance(version, str) else ''
            except ImportError:
                self.log("toml package not available, skipping pyproject.toml", "WARNING")
            except Exception as e:
                self.log(f"Error parsing pyproject.toml: {str(e)}", "WARNING")
        
        return requirements
    
    def auto_install_missing(self, directory: str) -> Dict[str, bool]:
        """Automatically detect and install all missing dependencies"""
        self.log("=== Starting Automatic Dependency Installation ===")
        
        # Step 1: Scan for imports
        imports = self.scan_directory(directory)
        
        # Step 2: Analyze requirement files
        requirements = self.analyze_requirements_files(directory)
        
        # Step 3: Combine detected imports with explicit requirements
        all_packages = imports.union(set(requirements.keys()))
        
        # Step 4: Filter out system packages
        external_packages = all_packages - self.system_packages
        
        self.log(f"Found {len(external_packages)} external packages to check")
        
        # Step 5: Check and install missing packages
        results = {}
        for package in sorted(external_packages):
            if not self.is_package_installed(package):
                self.log(f"Package '{package}' is not installed")
                success = self.install_package(package)
                results[package] = success
            else:
                version = self.get_package_version(package)
                self.log(f"Package '{package}' already installed (version: {version})")
                results[package] = True
        
        # Summary
        self.log("=== Installation Summary ===")
        self.log(f"Total packages checked: {len(external_packages)}")
        self.log(f"Successfully installed: {len(self.installed_packages)}")
        self.log(f"Failed installations: {len(self.failed_packages)}")
        
        if self.failed_packages:
            self.log("Failed packages: " + ", ".join(self.failed_packages), "WARNING")
        
        return results
    
    def create_requirements_file(self, output_file: str = "requirements_auto.txt"):
        """Generate a requirements.txt file with all detected dependencies"""
        self.log(f"Creating requirements file: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("# Auto-generated requirements file\n")
            f.write(f"# Generated on: {datetime.now().isoformat()}\n\n")
            
            installed = self.check_installed_packages()
            
            for package in sorted(self.detected_imports - self.system_packages):
                pip_name = self.package_mappings.get(package, package)
                if pip_name.lower() in installed:
                    version = installed[pip_name.lower()]
                    f.write(f"{pip_name}=={version}\n")
                else:
                    f.write(f"{pip_name}\n")
        
        self.log(f"Requirements file created: {output_file}")


def interactive_setup():
    """Interactive configuration wizard for dependency management"""
    print("\n🔧 Neural Network Dependency Manager Setup 🔧")
    print("=" * 50)
    
    # Ask for project directory
    while True:
        project_dir = input("\nEnter the project directory path (or '.' for current): ").strip()
        if not project_dir:
            project_dir = "."
        
        project_dir = os.path.abspath(project_dir)
        if os.path.exists(project_dir):
            break
        else:
            print(f"❌ Directory '{project_dir}' does not exist. Please try again.")
    
    # Ask for upgrade preference
    upgrade_input = input("\nAlways upgrade to latest versions? (y/n) [y]: ").strip().lower()
    upgrade = upgrade_input != 'n'
    
    # Ask for virtual environment
    venv_input = input("\nCreate/use virtual environment? (y/n) [y]: ").strip().lower()
    use_venv = venv_input != 'n'
    
    if use_venv:
        venv_path = os.path.join(project_dir, 'venv')
        if not os.path.exists(venv_path):
            print(f"\n📦 Creating virtual environment at: {venv_path}")
            subprocess.run([sys.executable, "-m", "venv", venv_path])
            
        # Activate virtual environment instructions
        if sys.platform == "win32":
            activate_cmd = f"{venv_path}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_path}/bin/activate"
        
        print(f"\n✅ Virtual environment ready!")
        print(f"   To activate: {activate_cmd}")
        input("\nPress Enter after activating the virtual environment...")
    
    # Initialize dependency manager
    print("\n🔍 Initializing Dependency Manager...")
    dm = DependencyManager()
    
    # Run auto-installation
    print(f"\n🚀 Scanning {project_dir} for dependencies...")
    results = dm.auto_install_missing(project_dir)
    
    # Create requirements file
    create_req = input("\n📄 Create requirements.txt file? (y/n) [y]: ").strip().lower()
    if create_req != 'n':
        dm.create_requirements_file(os.path.join(project_dir, "requirements_auto.txt"))
    
    print("\n✨ Dependency setup complete!")
    
    return dm


if __name__ == "__main__":
    # Run interactive setup when executed directly
    interactive_setup()