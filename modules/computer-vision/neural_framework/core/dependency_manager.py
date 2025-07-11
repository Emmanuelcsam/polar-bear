import ast
import os
import subprocess
import sys
from stdlib_list import stdlib_list

from .logger import log

class DependencyManager:
    def __init__(self, project_path):
        self.project_path = os.path.abspath(project_path)
        self.dependencies = set()
        try:
            self.std_libs = set(stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}"))
        except Exception:
            # Fallback for if the python version is not in stdlib_list
            self.std_libs = set(stdlib_list("3.11"))
            
        self.local_modules = self._find_local_modules()
        log.info(f"Initialized DependencyManager for {self.project_path}")
        log.debug(f"Found local modules: {self.local_modules}")

    def _find_local_modules(self):
        """Finds all .py files and directories with __init__.py in the project path."""
        local_modules = set()
        for root, dirs, files in os.walk(self.project_path):
            # Make sure we don't go into the neural_framework directory itself
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
                        # Simple parsing, ignores version specifiers, comments, etc.
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
                                    if node.module and node.level == 0: # Only check absolute imports
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