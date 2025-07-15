import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.logger import log
from core.module_loader import ModuleLoader

class MainOrchestrator:
    def __init__(self):
        self.modules_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.module_registry = {}
        self.sorted_modules = []

    def startup(self):
        """
        Loads all installed modules and starts the interactive CLI.
        """
        log.info("--- Starting Neural Framework ---")
        log.info("Assuming projects were installed via 'setup_and_install.py'.")

        # --- Module Loading Phase ---
        log.info("\n\n--- Loading all discovered modules ---")
        modules_to_load = self._discover_modules()
        loader = ModuleLoader(modules_to_load)
        self.module_registry = loader.load_modules()
        
        if not self.module_registry:
            log.warning("No modules were loaded successfully. Exiting.")
            return

        log.info("\n\n--- Framework startup complete ---")
        self.run_interactive_cli()

    def _discover_modules(self):
        """
        Discovers all python files in the project directories and maps them to module names.
        Excludes common test and utility directories, and files with 'test' in their name.
        """
        log.info("Discovering all potential modules...")
        excluded_dirs = {'__pycache__', '.venv-neural-connector', '.git', 'tests', 'test'}
        
        modules_to_load = {}
        
        for root_dir_name in os.listdir(self.modules_root):
            project_path = os.path.join(self.modules_root, root_dir_name)
            if not os.path.isdir(project_path) or root_dir_name in excluded_dirs:
                continue

            for root, dirs, files in os.walk(project_path):
                # Modify dirs in-place to exclude specified directories from traversal
                dirs[:] = [d for d in dirs if d not in excluded_dirs]
                
                for file in files:
                    if file.endswith(".py") and not file.startswith("__") and "test" not in file and file != "setup.py":
                        file_path = os.path.join(root, file)
                        
                        # Create a module name from the file path relative to the project
                        relative_path = os.path.relpath(file_path, project_path)
                        module_name = os.path.splitext(relative_path)[0].replace(os.sep, '.')
                        
                        # Special handling for top-level modules in the project root
                        if os.path.dirname(relative_path) == '':
                           module_name = os.path.splitext(file)[0]

                        modules_to_load[module_name] = file_path
        
        log.info(f"Discovered {len(modules_to_load)} potential modules.")
        return modules_to_load

    def run_interactive_cli(self):
        """Runs the main interactive command-line interface."""
        while True:
            print("\n" + "="*50)
            print("--- Neural Framework Interactive CLI ---")
            print("="*50)
            print("1. List Loaded Modules")
            print("2. View Module Details")
            print("3. Exit")
            print("-"*50)
            
            try:
                choice = input("Enter your choice: ")
            except EOFError:
                log.warning("EOFError received, exiting.")
                break

            if choice == '1':
                self._list_modules()
            elif choice == '2':
                self._view_module_details()
            elif choice == '3':
                log.info("Exiting Neural Framework.")
                break
            else:
                print("Invalid choice. Please try again.")

    def _list_modules(self):
        if not self.module_registry:
            print("No modules were loaded.")
            return
        print("\n--- Loaded Modules ---")
        self.sorted_modules = sorted(list(self.module_registry.keys()))
        for i, name in enumerate(self.sorted_modules, 1):
            print(f"{i}. {name}")

    def _view_module_details(self):
        self._list_modules()
        if not self.module_registry:
            return
        try:
            choice_str = input("Select a module to view details (or press Enter to cancel): ")
            if not choice_str: return
            choice = int(choice_str)
            module_name = self.sorted_modules[choice - 1]
            module_data = self.module_registry[module_name]
            analysis = module_data['analysis']
            
            print(f"\n--- Details for Module: {module_name} ---")
            print(f"File Path: {analysis['file_path']}")
            
            print("\nFunctions:")
            if not analysis['functions']: print("  No top-level functions found.")
            for func in analysis['functions']:
                print(f"  - {func['name']}({', '.join(func['args'])})")
                print(f"    \"{func['docstring']}\"")

            print("\nClasses:")
            if not analysis['classes']: print("  No classes found.")
            for cls in analysis['classes']:
                print(f"  - class {cls['name']}:")
                print(f"    \"{cls['docstring']}\"")
                for method in cls['methods']:
                    print(f"    - {method['name']}({', '.join(method['args'])})")
                    print(f"      \"{method['docstring']}\"")
        except (ValueError, IndexError):
            print("Invalid selection.")
        except (EOFError):
            log.warning("Input stream closed.")

def main():
    orchestrator = MainOrchestrator()
    orchestrator.startup()

if __name__ == "__main__":
    main()