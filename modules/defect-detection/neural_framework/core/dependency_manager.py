

import os
import ast
import subprocess
import sys
import importlib.util
from typing import Set, List

from .logger import log

# A mapping from common import names to their corresponding package names for pip.
# This is not exhaustive but covers many common cases in scientific computing.
IMPORT_TO_PACKAGE_MAP = {
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "tensorflow": "tensorflow",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "colorlog": "colorlog",
    "pytest": "pytest",
}

# Standard library modules to ignore during dependency scanning.
# This list is not exhaustive but covers the most common ones.
STANDARD_LIB_MODULES = {
    "os", "sys", "math", "datetime", "json", "csv", "logging", "collections",
    "itertools", "functools", "re", "subprocess", "multiprocessing", "threading",
    "argparse", "pathlib", "glob", "ast", "typing", "abc", "time", "random",
    "unittest", "pickle", "copy", "inspect", "warnings", "io", "struct",
}

# Packages that are handled by a `try...except ImportError` block and are not critical.
# This is a pragmatic whitelist to prevent the dependency check from failing on optional packages.
OPTIONAL_PACKAGES = {
    "accelerator", # Optional C++ accelerator for DO2MR
}

# Modules that are part of the framework itself.
INTERNAL_MODULES = {
    "neural_framework",
}




class DependencyManager:
    """
    Analyzes project files to discover and manage dependencies.
    """
    def __init__(self, project_root: str):
        """
        Initializes the DependencyManager.
        :param project_root: The absolute path to the root of the project to scan.
        """
        self.project_root = project_root
        self.logger = log
        self.local_module_names = self._discover_local_modules()

    def _discover_local_modules(self) -> Set[str]:
        """
        Scans the project to find all potential local module names from file and directory names.
        """
        local_modules = set()
        framework_dir = os.path.join(self.project_root, 'neural_framework')
        for root, _, files in os.walk(self.project_root):
            if root.startswith(framework_dir):
                continue
            for file in files:
                if file.endswith('.py'):
                    module_name = os.path.splitext(file)[0].replace('-', '_')
                    if module_name == '__init__':
                        pkg_name = os.path.basename(root).replace('-', '_')
                        local_modules.add(pkg_name)
                    else:
                        local_modules.add(module_name)
        self.logger.debug(f"Discovered {len(local_modules)} local module names: {sorted(list(local_modules))}")
        return local_modules

    def _get_script_paths(self) -> List[str]:
        """
        Finds all Python script paths in the project directory, excluding the framework itself.
        """
        script_paths = []
        framework_dir = os.path.join(self.project_root, 'neural_framework')
        for root, _, files in os.walk(self.project_root):
            if root.startswith(framework_dir):
                continue
            for file in files:
                if file.endswith(".py"):
                    script_paths.append(os.path.join(root, file))
        self.logger.debug(f"Found {len(script_paths)} Python scripts to analyze.")
        return script_paths

    def _extract_imports(self, file_path: str) -> Set[str]:
        """
        Parses a single Python file and extracts all top-level import names.
        """
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # e.g., import numpy -> 'numpy'
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    # e.g., from collections import deque -> 'collections'
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
            self.logger.warning(f"Could not parse {os.path.basename(file_path)}: {e}")
        return imports

    def discover_dependencies(self) -> Set[str]:
        """
        Scans all Python files in the project to find all unique dependencies.
        """
        self.logger.info("Starting dependency discovery...")
        all_imports = set()
        script_paths = self._get_script_paths()

        for path in script_paths:
            imports = self._extract_imports(path)
            all_imports.update(imports)

        # Filter out standard library modules, optional packages, internal modules and local project imports
        discovered_packages = {
            imp for imp in all_imports 
            if imp not in STANDARD_LIB_MODULES 
            and imp not in OPTIONAL_PACKAGES
            and imp not in INTERNAL_MODULES
            and not self._is_local_import(imp)
        }
        
        self.logger.info(f"Discovered {len(discovered_packages)} potential third-party packages.")
        self.logger.debug(f"Discovered packages: {sorted(list(discovered_packages))}")
        return discovered_packages

    def _is_local_import(self, import_name: str) -> bool:
        """
        Checks if an import name corresponds to a discovered local module.
        """
        return import_name in self.local_module_names

    def check_and_install(self, auto_install: bool = False):
        """
        Checks for missing dependencies and installs them.
        :param auto_install: If True, installs without asking. (Not implemented via CLI yet)
        """
        self.logger.info("Checking for required packages...")
        dependencies = self.discover_dependencies()
        missing_packages = []

        for import_name in dependencies:
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                package_name = IMPORT_TO_PACKAGE_MAP.get(import_name, import_name)
                missing_packages.append(package_name)

        if not missing_packages:
            self.logger.info("All required packages are already installed.")
            return True

        self.logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        
        # This part will be handled by the main script to allow for user interaction
        # before running the installation command.
        return missing_packages

if __name__ == '__main__':
    # Example usage:
    project_directory = os.path.abspath('.')
    manager = DependencyManager(project_directory)
    missing = manager.check_and_install()
    if isinstance(missing, list) and missing:
        print("\nTo install missing packages, you could run:")
        print(f"pip install --upgrade {' '.join(missing)}")


