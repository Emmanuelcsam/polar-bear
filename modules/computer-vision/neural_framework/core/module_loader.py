import os
import sys
import importlib
import importlib.util
from .logger import log
from .module_analyzer import ModuleAnalyzer

class ModuleLoader:
    def __init__(self, modules_to_load):
        self.modules_to_load = modules_to_load
        self.module_registry = {}

    def load_modules(self):
        log.info(f"Attempting to import {len(self.modules_to_load)} discovered modules...")
        for module_name, file_path in self.modules_to_load.items():
            module = None
            try:
                # First, try to import it as a standard module (for installed packages)
                module = importlib.import_module(module_name)
                log.info(f"Successfully imported installed module: {module_name}")
            except ImportError:
                log.warning(f"Module '{module_name}' not found via standard import. Attempting to load from file.")
                try:
                    # Fallback for standalone scripts or modules not in packages
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec:
                        module = importlib.util.module_from_spec(spec)
                        # Add the script's directory to the path for its local imports
                        sys.path.insert(0, os.path.dirname(file_path))
                        spec.loader.exec_module(module)
                        sys.path.pop(0) # Clean up path
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
