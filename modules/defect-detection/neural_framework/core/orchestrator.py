

import os
import ast
import importlib.util
import inspect
from typing import Dict, List, Any, Set
from dataclasses import dataclass, field

from .logger import log

@dataclass
class FunctionInfo:
    """
    A container for metadata about a discovered function.
    """
    name: str
    module_path: str
    module_name: str
    docstring: str
    params: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"Function(name='{self.name}', module='{self.module_name}', params={self.params})"

class Orchestrator:
    """
    Discovers, catalogs, and executes functions from the project scripts.
    """
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.functions: Dict[str, List[FunctionInfo]] = {}
        self.broken_modules: Set[str] = set()
        self.logger = log

    def discover_modules(self):
        """
        Scans all Python files in the project, parsing them with AST to
        discover all function definitions.
        """
        self.logger.info("Starting module discovery and function analysis...")
        py_files = self._get_py_files()

        for file_path in py_files:
            module_name = os.path.splitext(os.path.basename(file_path))[0].replace('-', '_')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=file_path)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name.startswith('_'):
                            continue
                        
                        params = [arg.arg for arg in node.args.args]
                        docstring = ast.get_docstring(node) or "No docstring."
                        
                        func_info = FunctionInfo(
                            name=node.name,
                            module_path=file_path,
                            module_name=module_name,
                            docstring=docstring,
                            params=params
                        )
                        
                        if node.name not in self.functions:
                            self.functions[node.name] = []
                        self.functions[node.name].append(func_info)

            except Exception as e:
                self.logger.error(f"Could not parse {file_path}: {e}")
                self.broken_modules.add(module_name)
        
        self.logger.info(f"Discovery complete. Found {sum(len(v) for v in self.functions.values())} total function instances.")
        if self.broken_modules:
            self.logger.warning(f"Could not parse {len(self.broken_modules)} modules: {self.broken_modules}")

    def _get_py_files(self) -> List[str]:
        """Gets all python files in a directory, excluding the framework itself."""
        py_files = []
        framework_dir = os.path.join(self.project_root, 'neural_framework')
        for root, _, files in os.walk(self.project_root):
            if root.startswith(framework_dir):
                continue
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.join(root, file))
        return py_files

    def list_functions(self):
        """Prints a formatted list of all discovered functions."""
        if not self.functions:
            self.logger.warning("No functions discovered. Run discover_modules() first.")
            return
            
        self.logger.info("--- Available Functions ---")
        for func_name in sorted(self.functions.keys()):
            func_infos = self.functions[func_name]
            if len(func_infos) == 1:
                info = func_infos[0]
                param_str = ", ".join(info.params)
                print(f"  - {info.name}({param_str})  (from {info.module_name})")
            else:
                print(f"  - {func_name} (defined in {len(func_infos)} modules):")
                for info in sorted(func_infos, key=lambda f: f.module_name):
                    param_str = ", ".join(info.params)
                    print(f"    - from {info.module_name}: {info.name}({param_str})")

        self.logger.info("-------------------------")

    def run_function(self, function_name: str, module_name: str = None, **kwargs: Any) -> Any:
        """
        Dynamically imports and executes a function by its name.
        If module_name is provided, it will disambiguate between functions with the same name.
        """
        if function_name not in self.functions:
            self.logger.error(f"Function '{function_name}' not found.")
            return None

        possible_funcs = self.functions[function_name]
        func_info = None

        if module_name:
            for f in possible_funcs:
                if f.module_name == module_name:
                    func_info = f
                    break
            if not func_info:
                self.logger.error(f"Function '{function_name}' not found in module '{module_name}'.")
                return None
        elif len(possible_funcs) == 1:
            func_info = possible_funcs[0]
        else:
            self.logger.error(f"Function '{function_name}' is ambiguous. Please specify a module.")
            self.logger.error(f"Available modules for '{function_name}': {[f.module_name for f in possible_funcs]}")
            return None

        self.logger.info(f"Executing function '{function_name}' from module '{func_info.module_name}'...")
        
        try:
            spec = importlib.util.spec_from_file_location(func_info.module_name, func_info.module_path)
            if not spec or not spec.loader:
                raise ImportError(f"Could not create module spec for {func_info.module_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            func_to_run = getattr(module, function_name)
            
            result = func_to_run(**kwargs)
            self.logger.info(f"Function '{function_name}' executed successfully.")
            return result

        except ImportError as e:
            self.logger.error(f"Failed to import module '{func_info.module_name}' to run '{function_name}'.")
            self.logger.error(f"The module likely has a broken internal import: {e}")
            self.broken_modules.add(func_info.module_name)
            return None
        except Exception as e:
            self.logger.error(f"An error occurred while executing '{function_name}': {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
