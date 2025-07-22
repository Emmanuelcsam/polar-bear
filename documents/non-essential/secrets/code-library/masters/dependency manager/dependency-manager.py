#!/usr/bin/env python3
"""
Ultra-Intelligent Dependency Management System - Merged Version
Combines functionality from all dependency manager implementations
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
from pathlib import Path

try:
    from stdlib_list import stdlib_list
    HAS_STDLIB_LIST = True
except ImportError:
    HAS_STDLIB_LIST = False


class DependencyManager:
    """Advanced dependency detection and installation system with all features combined"""
    
    def __init__(self, project_path: str = ".", log_file: str = "dependency_manager.log", 
                 use_venv: bool = False, venv_path: str = ".venv-neural-connector"):
        self.project_path = os.path.abspath(project_path)
        self.log_file = log_file
        self.use_venv = use_venv
        self.venv_path = Path(venv_path)
        
        # Initialize sets for tracking
        self.detected_imports = set()
        self.installed_packages = set()
        self.failed_packages = set()
        self.dependencies = set()
        self.local_modules = set()
        
        # Version constraints and mappings
        self.version_constraints = {}
        self.special_constraints = {
            "mediapipe": {"numpy": "numpy<2.0"}  # Mediapipe requires numpy < 2.0
        }
        
        # Initialize standard library detection
        self._init_stdlib_detection()
        
        # Comprehensive package mappings combining all sources
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
            'torchaudio': 'torchaudio',
            'tensorflow': 'tensorflow',
            'keras': 'tensorflow',
            'h5py': 'h5py',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib',
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
            'colorlog': 'colorlog',
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
            'pre_commit': 'pre-commit',
            'mediapipe': 'mediapipe',
            'aiofiles': 'aiofiles',
            'openpyxl': 'openpyxl',
            'autopy': 'autopy',
            'pycaw': 'pycaw',
            'networkx': 'networkx',
            'pynmea2': 'pynmea2',
        }
        
        # Optional packages that may not be critical
        self.optional_packages = {
            'accelerator',  # Optional C++ accelerator
        }
        
        # Internal/framework modules to ignore
        self.internal_modules = {
            'neural_framework',
            'neural_connector',
        }
        
        # Directories to skip during scanning
        self.skip_dirs = {
            'venv', '__pycache__', '.git', 'node_modules',
            '.pytest_cache', '.mypy_cache', 'build', 'dist',
            'neural_framework', '.venv-neural-connector'
        }
        
        self._setup_logging()
        self._find_local_modules()
        
        if self.use_venv:
            self.python_executable = self._setup_virtual_environment()
        else:
            self.python_executable = sys.executable
            
        self.log("Dependency Manager initialized")
    
    def _init_stdlib_detection(self):
        """Initialize standard library detection with fallback"""
        if HAS_STDLIB_LIST:
            try:
                self.system_packages = set(stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}"))
            except Exception:
                # Fallback for unsupported Python versions
                self.system_packages = set(stdlib_list("3.11"))
        else:
            # Manual fallback list combining all sources
            self.system_packages = {
                'os', 'sys', 'time', 'datetime', 'json', 'math', 'random',
                'collections', 'itertools', 'functools', 'pathlib', 'typing',
                'subprocess', 'threading', 'multiprocessing', 'queue', 'io',
                're', 'string', 'copy', 'pickle', 'sqlite3', 'csv', 'xml',
                'logging', 'warnings', 'traceback', 'inspect', 'ast', 'imp',
                'importlib', 'pkgutil', 'pkg_resources', 'platform', 'locale',
                'glob', 'fnmatch', 'tempfile', 'shutil', 'stat', 'filecmp',
                'configparser', 'argparse', 'getopt', 'getpass', 'cmd', 'shlex',
                'hashlib', 'unittest', 'abc'
            }
    
    def _setup_logging(self):
        """Setup dual logging to file and console"""
        self.logger = logging.getLogger('DependencyManager')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
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
    
    def _setup_virtual_environment(self) -> str:
        """Setup or use existing virtual environment"""
        venv_python = self.venv_path / "Scripts" / "python.exe" if sys.platform == "win32" else self.venv_path / "bin" / "python"
        
        if self.venv_path.exists() and venv_python.exists():
            self.log(f"Virtual environment found at {self.venv_path}")
            return str(venv_python)
        
        self.log(f"Creating virtual environment at {self.venv_path}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_path)],
                check=True, capture_output=True, text=True
            )
            self.log("Successfully created virtual environment")
            return str(venv_python)
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to create virtual environment: {e}\n{e.stderr}", "ERROR")
            return sys.executable
    
    def _find_local_modules(self):
        """Find all local module names in the project"""
        local_modules = set()
        
        for root, dirs, files in os.walk(self.project_path):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            
            # Add directory names that have __init__.py
            for d in dirs:
                if os.path.exists(os.path.join(root, d, "__init__.py")):
                    local_modules.add(d.replace('-', '_'))
            
            # Add Python file names
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    module_name = os.path.splitext(file)[0].replace('-', '_')
                    local_modules.add(module_name)
        
        self.local_modules = local_modules
        self.log(f"Found {len(local_modules)} local modules", "DEBUG")
    
    def scan_directory(self, directory: str = None) -> Set[str]:
        """Recursively scan directory for Python files and extract imports"""
        if directory is None:
            directory = self.project_path
            
        self.log(f"Scanning directory: {directory}")
        imports = set()
        
        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            
            for file in files:
                if file.endswith('.py') and "mock" not in file.lower():
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
                content = f.read()
                tree = ast.parse(content, filepath)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.level == 0:  # Only absolute imports
                        imports.add(node.module.split('.')[0])
            
            # Also check for dynamic imports
            # Find __import__ calls
            import_pattern = r'__import__\s*\(\s*[\'"]([^\'"\)]+)[\'"]'
            dynamic_imports = re.findall(import_pattern, content)
            imports.update(i.split('.')[0] for i in dynamic_imports)
            
            # Find importlib.import_module calls
            importlib_pattern = r'import_module\s*\(\s*[\'"]([^\'"\)]+)[\'"]'
            importlib_imports = re.findall(importlib_pattern, content)
            imports.update(i.split('.')[0] for i in importlib_imports)
            
        except (SyntaxError, UnicodeDecodeError) as e:
            self.log(f"Failed to parse {filepath}: {str(e)}", "ERROR")
        except Exception as e:
            self.log(f"Unexpected error parsing {filepath}: {str(e)}", "ERROR")
        
        return imports
    
    def analyze_requirements_files(self, directory: str = None) -> Dict[str, str]:
        """Analyze requirements.txt, setup.py, pyproject.toml files"""
        if directory is None:
            directory = self.project_path
            
        requirements = {}
        
        # Check various requirements files
        req_files = [
            'requirements.txt', 'requirements-dev.txt', 'requirements-test.txt',
            'requirements_auto.txt', 'consolidated_requirements.txt'
        ]
        
        for req_file in req_files:
            filepath = os.path.join(directory, req_file)
            if os.path.exists(filepath):
                self.log(f"Analyzing {req_file}")
                try:
                    with open(filepath, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and not line.startswith('-'):
                                # Parse requirement specifier
                                match = re.match(r'^([a-zA-Z0-9\-_\[\]]+)(.*)$', line)
                                if match:
                                    pkg_name = match.group(1).split('[')[0]  # Remove extras
                                    version_spec = match.group(2).strip()
                                    requirements[pkg_name] = version_spec
                except Exception as e:
                    self.log(f"Error reading {req_file}: {str(e)}", "WARNING")
        
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
                
                # Check poetry dependencies
                deps = data.get('tool', {}).get('poetry', {}).get('dependencies', {})
                for pkg, version in deps.items():
                    if pkg != 'python':
                        requirements[pkg] = version if isinstance(version, str) else ''
                
                # Check PEP 621 dependencies
                pep621_deps = data.get('project', {}).get('dependencies', [])
                for dep in pep621_deps:
                    match = re.match(r'^([a-zA-Z0-9\-_]+)(.*)$', dep)
                    if match:
                        requirements[match.group(1)] = match.group(2).strip()
                        
            except ImportError:
                self.log("toml package not available, skipping pyproject.toml", "WARNING")
            except Exception as e:
                self.log(f"Error parsing pyproject.toml: {str(e)}", "WARNING")
        
        return requirements
    
    def check_installed_packages(self) -> Dict[str, str]:
        """Get all currently installed packages and their versions"""
        installed = {}
        
        # Try multiple methods to get installed packages
        try:
            # Method 1: pkg_resources
            for dist in pkg_resources.working_set:
                installed[dist.key.lower()] = dist.version
        except Exception as e:
            self.log(f"pkg_resources method failed: {str(e)}", "WARNING")
        
        # Method 2: pip list
        try:
            result = subprocess.run(
                [self.python_executable, "-m", "pip", "list", "--format=json"],
                capture_output=True, text=True, check=True
            )
            packages = json.loads(result.stdout)
            for pkg in packages:
                installed[pkg['name'].lower()] = pkg['version']
        except Exception as e:
            self.log(f"pip list method failed: {str(e)}", "WARNING")
        
        return installed
    
    def is_package_installed(self, package: str) -> bool:
        """Check if a package is installed"""
        # First try direct import
        try:
            importlib.import_module(package)
            return True
        except ImportError:
            pass
        
        # Try with package mapping
        pip_name = self.package_mappings.get(package, package)
        
        # Check installed packages list
        installed = self.check_installed_packages()
        return pip_name.lower() in installed
    
    def get_package_version(self, package: str) -> Optional[str]:
        """Get the version of an installed package"""
        pip_name = self.package_mappings.get(package, package)
        installed = self.check_installed_packages()
        return installed.get(pip_name.lower())
    
    def _apply_special_constraints(self, package: str) -> str:
        """Apply special version constraints for specific packages"""
        # Check if any installed package requires special constraints
        for trigger_pkg, constraints in self.special_constraints.items():
            if self.is_package_installed(trigger_pkg) and package in constraints:
                return constraints[package]
        
        # Check if this package requires constraints on others
        if package in self.special_constraints:
            # We'll handle this after installing the package
            pass
            
        return package
    
    def install_package(self, package: str, upgrade: bool = True, 
                       version_spec: str = "") -> bool:
        """Install a package using pip with advanced features"""
        pip_name = self.package_mappings.get(package, package)
        
        # Apply special constraints
        constrained_name = self._apply_special_constraints(pip_name)
        if constrained_name != pip_name:
            self.log(f"Applying constraint: {pip_name} -> {constrained_name}")
            pip_name = constrained_name
        
        # Add version spec if provided
        if version_spec and not any(op in pip_name for op in ['<', '>', '=', '!']):
            pip_name += version_spec
        
        self.log(f"Installing package: {pip_name}")
        
        try:
            cmd = [self.python_executable, "-m", "pip", "install"]
            if upgrade and '=' not in pip_name:  # Don't upgrade if specific version requested
                cmd.append("--upgrade")
            cmd.extend(["--no-cache-dir", pip_name])
            
            # Run pip with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_lines.append(line.strip())
                    self.log(f"  {line.strip()}", "DEBUG")
            
            process.wait()
            
            if process.returncode == 0:
                self.installed_packages.add(package)
                self.log(f"Successfully installed {pip_name}", "INFO")
                
                # Check if we need to install/update other packages due to constraints
                if package in self.special_constraints:
                    for dep_pkg, dep_constraint in self.special_constraints[package].items():
                        self.log(f"Installing constrained dependency: {dep_constraint}")
                        self.install_package(dep_pkg, upgrade=False, version_spec=dep_constraint)
                
                return True
            else:
                self.failed_packages.add(package)
                self.log(f"Failed to install {pip_name}", "ERROR")
                # Log last few lines of output for debugging
                for line in output_lines[-10:]:
                    self.log(f"  {line}", "ERROR")
                return False
                
        except Exception as e:
            self.failed_packages.add(package)
            self.log(f"Exception installing {pip_name}: {str(e)}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
            return False
    
    def discover_dependencies(self) -> Set[str]:
        """Discover all dependencies using multiple methods"""
        self.log("Starting comprehensive dependency discovery...")
        
        # Method 1: Scan source files
        source_imports = self.scan_directory()
        
        # Method 2: Analyze requirement files
        requirements = self.analyze_requirements_files()
        
        # Combine all discovered packages
        all_packages = source_imports.union(set(requirements.keys()))
        
        # Filter out standard library, optional, internal, and local modules
        discovered_packages = {
            pkg for pkg in all_packages
            if pkg not in self.system_packages
            and pkg not in self.optional_packages
            and pkg not in self.internal_modules
            and pkg not in self.local_modules
            and not pkg.startswith('_')
        }
        
        # Update version constraints from requirements
        for pkg, version in requirements.items():
            if version and pkg in discovered_packages:
                self.version_constraints[pkg] = version
        
        self.dependencies = discovered_packages
        self.log(f"Discovered {len(discovered_packages)} third-party packages")
        self.log(f"Packages: {sorted(list(discovered_packages))}", "DEBUG")
        
        return discovered_packages
    
    def check_missing_packages(self) -> List[str]:
        """Check which packages are missing"""
        missing = []
        for package in self.dependencies:
            if not self.is_package_installed(package):
                missing.append(package)
        return missing
    
    def auto_install_missing(self, interactive: bool = True, 
                           upgrade: bool = True) -> Dict[str, bool]:
        """Automatically detect and install all missing dependencies"""
        self.log("=== Starting Automatic Dependency Installation ===")
        
        # Discover all dependencies
        dependencies = self.discover_dependencies()
        
        if not dependencies:
            self.log("No external dependencies found")
            return {}
        
        # Check what's missing
        missing_packages = self.check_missing_packages()
        
        if not missing_packages:
            self.log("All required packages are already installed")
            # Still log what's installed
            for package in sorted(dependencies):
                version = self.get_package_version(package)
                self.log(f"Package '{package}' already installed (version: {version})")
            return {pkg: True for pkg in dependencies}
        
        self.log(f"Missing packages: {', '.join(missing_packages)}")
        
        # Ask for confirmation if interactive
        if interactive and missing_packages:
            try:
                response = input(f"\nInstall {len(missing_packages)} missing packages? (y/n) [y]: ").strip().lower()
                if response == 'n':
                    self.log("Installation cancelled by user")
                    return {pkg: False for pkg in missing_packages}
            except (EOFError, KeyboardInterrupt):
                self.log("\nInstallation cancelled by user")
                return {pkg: False for pkg in missing_packages}
        
        # Sort packages to handle dependencies (numpy before packages that depend on it)
        sorted_packages = sorted(missing_packages, key=lambda p: (
            0 if p == 'numpy' else 1,  # Install numpy first
            p
        ))
        
        # Install missing packages
        results = {}
        for package in sorted_packages:
            version_spec = self.version_constraints.get(package, "")
            success = self.install_package(package, upgrade=upgrade, version_spec=version_spec)
            results[package] = success
        
        # Check already installed packages
        for package in dependencies - set(missing_packages):
            version = self.get_package_version(package)
            self.log(f"Package '{package}' already installed (version: {version})")
            results[package] = True
        
        # Summary
        self.log("=== Installation Summary ===")
        self.log(f"Total packages checked: {len(dependencies)}")
        self.log(f"Successfully installed: {len(self.installed_packages)}")
        self.log(f"Failed installations: {len(self.failed_packages)}")
        
        if self.failed_packages:
            self.log("Failed packages: " + ", ".join(self.failed_packages), "WARNING")
            self.log("You may need to install these packages manually", "WARNING")
        
        return results
    
    def create_requirements_file(self, output_file: str = "requirements_merged.txt",
                               include_versions: bool = True):
        """Generate a requirements.txt file with all detected dependencies"""
        self.log(f"Creating requirements file: {output_file}")
        
        # Ensure we have discovered dependencies
        if not self.dependencies:
            self.discover_dependencies()
        
        installed = self.check_installed_packages()
        
        with open(output_file, 'w') as f:
            f.write("# Auto-generated requirements file by merged dependency manager\n")
            f.write(f"# Generated on: {datetime.now().isoformat()}\n")
            f.write(f"# Project: {os.path.basename(self.project_path)}\n\n")
            
            # Group packages
            installed_deps = []
            missing_deps = []
            
            for package in sorted(self.dependencies):
                pip_name = self.package_mappings.get(package, package)
                if self.is_package_installed(package) and include_versions:
                    version = self.get_package_version(package)
                    if version:
                        installed_deps.append(f"{pip_name}=={version}")
                    else:
                        installed_deps.append(pip_name)
                else:
                    # Use version constraint if available
                    version_spec = self.version_constraints.get(package, "")
                    if version_spec:
                        missing_deps.append(f"{pip_name}{version_spec}")
                    else:
                        missing_deps.append(pip_name)
            
            # Write installed packages
            if installed_deps:
                f.write("# Installed packages\n")
                for dep in installed_deps:
                    f.write(f"{dep}\n")
                f.write("\n")
            
            # Write missing packages
            if missing_deps:
                f.write("# Missing packages\n")
                for dep in missing_deps:
                    f.write(f"{dep}\n")
        
        self.log(f"Requirements file created: {output_file}")
        return output_file
    
    def analyze_project(self, project_path: str = None) -> 'DependencyManager':
        """Analyze a project's dependencies (compatibility method)"""
        if project_path:
            self.project_path = os.path.abspath(project_path)
            self._find_local_modules()
        
        self.discover_dependencies()
        return self
    
    def get_found_dependencies(self) -> List[str]:
        """Get list of found dependencies (compatibility method)"""
        return list(self.dependencies)


def interactive_setup():
    """Interactive configuration wizard for dependency management"""
    print("\n🔧 Neural Network Dependency Manager Setup (Merged Version) 🔧")
    print("=" * 60)
    
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
    
    # Ask for virtual environment preference
    venv_input = input("\nUse virtual environment? (y/n) [n]: ").strip().lower()
    use_venv = venv_input == 'y'
    
    venv_path = ".venv-neural-connector"
    if use_venv:
        custom_path = input(f"Virtual environment path [{venv_path}]: ").strip()
        if custom_path:
            venv_path = custom_path
    
    # Ask for upgrade preference
    upgrade_input = input("\nUpgrade packages to latest versions? (y/n) [y]: ").strip().lower()
    upgrade = upgrade_input != 'n'
    
    # Initialize dependency manager
    print("\n🔍 Initializing Dependency Manager...")
    dm = DependencyManager(
        project_path=project_dir,
        use_venv=use_venv,
        venv_path=venv_path
    )
    
    # Run auto-installation
    print(f"\n🚀 Scanning {project_dir} for dependencies...")
    results = dm.auto_install_missing(interactive=True, upgrade=upgrade)
    
    # Create requirements file
    create_req = input("\n📄 Create requirements.txt file? (y/n) [y]: ").strip().lower()
    if create_req != 'n':
        req_filename = input("Requirements filename [requirements_merged.txt]: ").strip()
        if not req_filename:
            req_filename = "requirements_merged.txt"
        
        include_versions = input("Include version numbers? (y/n) [y]: ").strip().lower() != 'n'
        
        output_path = os.path.join(project_dir, req_filename)
        dm.create_requirements_file(output_path, include_versions=include_versions)
    
    print("\n✨ Dependency setup complete!")
    
    # Show summary
    if dm.failed_packages:
        print(f"\n⚠️  Failed to install: {', '.join(dm.failed_packages)}")
        print("   Please install these packages manually.")
    
    if dm.installed_packages:
        print(f"\n✅ Successfully installed: {', '.join(dm.installed_packages)}")
    
    return dm


def main():
    """Main entry point with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Neural Network Dependency Manager - Merged Version"
    )
    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Path to the project directory (default: current directory)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run in non-interactive mode"
    )
    parser.add_argument(
        "--no-upgrade",
        action="store_true",
        help="Don't upgrade existing packages"
    )
    parser.add_argument(
        "--venv",
        action="store_true",
        help="Use virtual environment"
    )
    parser.add_argument(
        "--venv-path",
        default=".venv-neural-connector",
        help="Virtual environment path"
    )
    parser.add_argument(
        "--requirements",
        help="Generate requirements file with specified name"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for missing dependencies, don't install"
    )
    
    args = parser.parse_args()
    
    if not args.auto and not args.check_only:
        # Run interactive setup
        interactive_setup()
    else:
        # Run automated
        dm = DependencyManager(
            project_path=args.project_path,
            use_venv=args.venv,
            venv_path=args.venv_path
        )
        
        if args.check_only:
            # Just check and report
            dm.discover_dependencies()
            missing = dm.check_missing_packages()
            
            if missing:
                print(f"\nMissing packages: {', '.join(missing)}")
                print(f"\nTo install: pip install {' '.join(missing)}")
                sys.exit(1)
            else:
                print("\nAll dependencies are satisfied.")
                sys.exit(0)
        else:
            # Auto-install
            results = dm.auto_install_missing(
                interactive=False,
                upgrade=not args.no_upgrade
            )
            
            if args.requirements:
                dm.create_requirements_file(args.requirements)
            
            # Exit with error if any installations failed
            if dm.failed_packages:
                sys.exit(1)


if __name__ == "__main__":
    main()