

import importlib.util
import inspect
import ast
from pathlib import Path
import logging
from typing import List, Dict, Any, Callable

logger = logging.getLogger("neural_connector.loader")

class ModuleInfo:
    """
    A data class to hold information about a dynamically loaded module.
    """
    def __init__(self, name: str, path: Path, module: Any):
        self.name = name
        self.path = path
        self.module = module
        self.functions: Dict[str, Callable] = {}
        self.tunable_parameters: Dict[str, Dict[str, Any]] = {}
        self._analyze()

    def _analyze(self):
        """
        Analyzes the module to extract functions and their parameters.
        """
        logger.debug("Analyzing module: %s", self.name)
        for name, func in inspect.getmembers(self.module, inspect.isfunction):
            # Ignore private functions and functions from imported modules
            if not name.startswith("_") and func.__module__ == self.name:
                self.functions[name] = func
                params = self._get_function_parameters(func)
                if params:
                    self.tunable_parameters[name] = params
                    logger.debug("  - Found function '%s' with parameters: %s", name, list(params.keys()))

    def _get_function_parameters(self, func: Callable) -> Dict[str, Any]:
        """
        Extracts parameters and their default values from a function signature.
        """
        params = {}
        try:
            signature = inspect.signature(func)
            for param in signature.parameters.values():
                param_info = {"type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"}
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                params[param.name] = param_info
        except (ValueError, TypeError) as e:
            logger.warning("Could not inspect parameters for function '%s': %s", func.__name__, e)
        return params

    def __repr__(self):
        return f"<ModuleInfo name='{self.name}' functions={len(self.functions)}>"


class ModuleLoader:
    """
    Dynamically discovers, loads, and analyzes Python scripts as modules.
    """
    def __init__(self, base_dir: str, ignore_list: List[str] = None):
        self.base_dir = Path(base_dir)
        self.ignore_list = ignore_list or []
        self.loaded_modules: Dict[str, ModuleInfo] = {}

    def discover_and_load_modules(self):
        """
        Finds all Python scripts in the base directory and loads them.
        """
        logger.info("Starting module discovery in: %s", self.base_dir)
        
        for py_file in self.base_dir.glob("*.py"):
            if self._should_ignore(py_file):
                continue

            module_name = py_file.stem
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    module_info = ModuleInfo(module_name, py_file, module)
                    self.loaded_modules[module_name] = module_info
                    logger.info("Successfully loaded and analyzed module: '%s'", module_name)
                else:
                    logger.warning("Could not create module spec for: %s", py_file)
            except Exception as e:
                logger.error("Failed to load module '%s': %s", module_name, e, exc_info=True)
        
        logger.info("Module discovery complete. Loaded %d modules.", len(self.loaded_modules))
        return self.loaded_modules

    def _should_ignore(self, file_path: Path) -> bool:
        """
        Checks if a file should be ignored based on the ignore list.
        """
        # Ignore the file if its name is in the ignore list
        if file_path.name in self.ignore_list:
            logger.debug("Ignoring script by name: %s", file_path.name)
            return True
            
        # Ignore files that are part of the neural_connector package itself
        if "neural_connector" in file_path.parts:
            logger.debug("Ignoring script from neural_connector package: %s", file_path.name)
            return True
            
        return False

    def get_all_tunable_parameters(self) -> Dict[str, Any]:
        """
        Gathers all tunable parameters from all loaded modules.
        """
        all_params = {}
        for module_name, module_info in self.loaded_modules.items():
            if module_info.tunable_parameters:
                all_params[module_name] = module_info.tunable_parameters
        return all_params

if __name__ == '__main__':
    # This allows for standalone testing of the module loader
    from neural_connector.core.logging_config import setup_logging
    setup_logging()

    # The base directory is the parent of this file's location
    # experimental-features/neural_connector/modules/loader.py -> experimental-features/
    current_dir = Path(__file__).parent.parent.parent
    
    # Ignore the main script of the connector itself
    ignore = ["main.py"]

    loader = ModuleLoader(base_dir=str(current_dir), ignore_list=ignore)
    modules = loader.discover_and_load_modules()

    print("\n--- Discovered Modules ---")
    for name, info in modules.items():
        print(f"- {name} ({len(info.functions)} functions)")

    print("\n--- All Tunable Parameters ---")
    all_parameters = loader.get_all_tunable_parameters()
    import json
    print(json.dumps(all_parameters, indent=2, default=str))

    # Example of how to call a function dynamically
    if "gradient_peak_detector" in modules:
        print("\n--- Dynamic Function Call Example ---")
        gpd_module = modules["gradient_peak_detector"]
        if "GradientPeakFiberAnalyzer" in gpd_module.functions:
             # This is a class, so we can instantiate it
            try:
                # Note: This is just an example. The class requires an image path.
                # We are just showing that the class is accessible.
                print("Successfully accessed class: GradientPeakFiberAnalyzer")
            except Exception as e:
                print(f"Could not instantiate GradientPeakFiberAnalyzer: {e}")


