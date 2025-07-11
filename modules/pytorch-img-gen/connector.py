import os
import sys
import logging
import subprocess
import importlib
import ast
import inspect
from pathlib import Path

# --- Constants ---
LOG_FILE_NAME = "connector.log"
WHITELISTED_FILES = [
    "feature-extractor.py",
    "image-saver.py",
    "live-display.py",
    "neural-generator.py",
    "noise-generator.py",
    "pixel-guide.py",
    "ref-processor.py",
    "style-optimizer.py",
    "texture-synth.py",
]
CURRENT_DIR = Path(__file__).parent.resolve()

# This will hold the analyzed structure of all modules
module_registry = {}
# This will cache imported modules to allow state to persist
loaded_modules = {}

# --- Setup Logging ---
def setup_logging():
    """Sets up dual logging to console and a file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE_NAME, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logger initialized.")

# --- AST Analysis ---
class ImportVisitor(ast.NodeVisitor):
    """AST visitor to find all imported modules."""
    def __init__(self):
        self.imports = set()

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module.split('.')[0])
        self.generic_visit(node)

class ModuleAnalyzer(ast.NodeVisitor):
    """AST visitor to find functions, classes, and top-level variables."""
    def __init__(self):
        self.structure = {'functions': [], 'classes': [], 'variables': []}

    def visit_FunctionDef(self, node):
        if not node.name.startswith('_'):
            args = [arg.arg for arg in node.args.args]
            self.structure['functions'].append({'name': node.name, 'args': args})
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if not node.name.startswith('_'):
            self.structure['classes'].append({'name': node.name, 'methods': []})
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Capture top-level variable assignments
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Check if it's a module-level constant (convention: UPPERCASE)
                if target.id.isupper():
                    self.structure['variables'].append(target.id)
        self.generic_visit(node)

def _find_imports_ast(file_path):
    """Parses a python file and returns a set of all imported modules."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content, filename=file_path.name)
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports
    except Exception as e:
        logging.error(f"Could not parse '{file_path.name}' for imports: {e}")
        return set()

def analyze_modules():
    """
    Analyzes all whitelisted python files and populates the module_registry.
    """
    logging.info("Starting module analysis...")
    for file_name in WHITELISTED_FILES:
        module_name = file_name.replace('.py', '')
        file_path = CURRENT_DIR / file_name
        if not file_path.exists():
            logging.warning(f"File '{file_name}' not found. Skipping analysis.")
            continue

        logging.info(f"Analyzing structure of '{file_name}'...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content, filename=file_name)
            analyzer = ModuleAnalyzer()
            analyzer.visit(tree)
            module_registry[module_name] = analyzer.structure
            logging.info(f"Found: {len(analyzer.structure['functions'])} functions, "
                         f"{len(analyzer.structure['classes'])} classes, "
                         f"{len(analyzer.structure['variables'])} variables in {file_name}")
        except Exception as e:
            logging.error(f"Could not analyze module '{file_name}': {e}")
            module_registry[module_name] = {'functions': [], 'classes': [], 'variables': [], 'error': str(e)}
    logging.info("Module analysis complete.")


# --- Dependency Management ---
def get_std_lib_modules():
    """Gets a list of standard library modules for the current Python version."""
    if sys.version_info >= (3, 10):
        return sys.stdlib_module_names
    else:
        logging.warning("Using a partial list of stdlib modules for Python < 3.10.")
        # This is a snapshot and not exhaustive.
        return {'os', 'sys', 'logging', 'subprocess', 'importlib', 'pathlib', 'ast', 're', 'collections', 'itertools', 'functools', 'datetime', 'math', 'random', 'json', 'string', 'typing', 'inspect'}


def check_and_install_dependencies():
    """
    Scans whitelisted files for imports using AST, checks if they are installed,
    and prompts the user to install them if they are not.
    """
    logging.info("Starting dependency analysis with AST...")
    std_lib_modules = get_std_lib_modules()
    local_modules = {f.replace('.py', '') for f in WHITELISTED_FILES}
    required_packages = set()

    for file_name in WHITELISTED_FILES:
        file_path = CURRENT_DIR / file_name
        if not file_path.exists():
            continue
        
        imports = _find_imports_ast(file_path)
        for package in imports:
            if package and package not in std_lib_modules and package not in local_modules:
                required_packages.add(package)

    logging.info(f"Discovered potential third-party packages: {sorted(list(required_packages))}")

    missing_packages = []
    for package in required_packages:
        try:
            # Map common import names to pip package names
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(package)
            logging.info(f"Package '{package}' is already installed.")
        except ImportError:
            logging.warning(f"Package '{package}' appears to be missing.")
            missing_packages.append(package)

    if not missing_packages:
        logging.info("All dependencies appear to be satisfied.")
        return

    print("\n--- Missing Dependencies Detected ---")
    print("The following packages are required but seem to be missing:")
    for pkg in missing_packages:
        print(f" - {pkg}")

    try:
        answer = input("Would you like to attempt to install them now? (yes/no): ").lower()
        if answer in ['yes', 'y']:
            logging.info(f"User approved installation of {len(missing_packages)} packages.")
            for pkg in missing_packages:
                # Map common packages to their correct pip names
                if pkg == 'cv2':
                    pip_pkg_name = 'opencv-python'
                elif pkg == 'PIL':
                    pip_pkg_name = 'Pillow'
                else:
                    pip_pkg_name = pkg
                
                logging.info(f"Installing '{pip_pkg_name}'...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pip_pkg_name],
                    capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    logging.info(f"Successfully installed '{pip_pkg_name}'.")
                    print(f"Successfully installed '{pip_pkg_name}'.")
                else:
                    logging.error(f"Failed to install '{pip_pkg_name}'.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
                    print(f"ERROR: Failed to install '{pip_pkg_name}'. Check '{LOG_FILE_NAME}' for details.")
        else:
            logging.warning("User declined automatic installation. The application may not run correctly.")
            print("Installation skipped. Please install the dependencies manually.")
    except KeyboardInterrupt:
        logging.warning("User cancelled installation process.")
        print("\nInstallation cancelled.")

# --- Interactive Menus ---

def _get_module(module_name):
    """Dynamically imports and returns a module, caching it."""
    if module_name in loaded_modules:
        return loaded_modules[module_name]
    
    try:
        logging.info(f"Loading module '{module_name}' for the first time.")
        module = importlib.import_module(module_name)
        loaded_modules[module_name] = module
        return module
    except ImportError as e:
        logging.error(f"Failed to import module {module_name}: {e}")
        print(f"Error: Could not import module '{module_name}'. Is it in the right directory?")
        return None

def _get_typed_input(prompt):
    """Gets user input and safely tries to convert it to a Python type."""
    raw_input = input(prompt)
    try:
        # Use literal_eval for safe evaluation of Python literals
        return ast.literal_eval(raw_input)
    except (ValueError, SyntaxError, MemoryError, TypeError):
        # Fallback to string if it's not a simple literal
        logging.warning(f"Could not evaluate input '{raw_input}' as a literal. Treating as string.")
        return raw_input

def modify_variable_menu():
    """Menu for modifying a tunable variable in a module."""
    print("\n--- Modify Tunable Variable ---")
    module_name = input("Enter module name (e.g., 'feature-extractor'): ")
    if module_name not in module_registry:
        print("Module not found.")
        return

    var_name = input("Enter variable name to modify: ")
    if var_name not in module_registry[module_name].get('variables', []):
        print(f"Variable '{var_name}' not found or not listed as tunable in '{module_name}'.")
        return

    module = _get_module(module_name)
    if not module:
        return

    try:
        current_value = getattr(module, var_name)
        print(f"Current value of '{var_name}': {current_value} (Type: {type(current_value).__name__})")
        
        new_value = _get_typed_input("Enter new value: ")
        
        setattr(module, var_name, new_value)
        logging.info(f"Set {module_name}.{var_name} from '{current_value}' to '{new_value}'.")
        print(f"Successfully updated '{var_name}' to: {getattr(module, var_name)}")

    except Exception as e:
        logging.error(f"Failed to modify variable {var_name} in {module_name}: {e}")
        print(f"An error occurred: {e}")

def execute_function_menu():
    """Menu for executing a function from a module."""
    print("\n--- Execute Function ---")
    module_name = input("Enter module name (e.g., 'neural-generator'): ")
    if module_name not in module_registry:
        print("Module not found.")
        return

    func_name = input("Enter function name to execute: ")
    
    module = _get_module(module_name)
    if not module or not hasattr(module, func_name):
        print(f"Function '{func_name}' not found in module '{module_name}'.")
        return

    func = getattr(module, func_name)
    sig = inspect.signature(func)
    args = {}

    print(f"Enter arguments for {func_name}{sig}:")
    for param in sig.parameters.values():
        prompt = f"  - {param.name}"
        if param.default != inspect.Parameter.empty:
            prompt += f" (default: {param.default}): "
        else:
            prompt += ": "
        
        user_input = _get_typed_input(prompt)
        args[param.name] = user_input

    try:
        logging.info(f"Executing {module_name}.{func_name} with args: {args}")
        print("Executing function...")
        result = func(**args)
        logging.info(f"Function returned: {result}")
        print(f"\n--- Result from {func_name} ---")
        print(result)
        print("-" * (20 + len(func_name)))

    except Exception as e:
        logging.error(f"Exception during execution of {func_name}: {e}", exc_info=True)
        print(f"\nAn error occurred during execution: {e}")

def run_tests_menu():
    """Runs the test generation and execution script."""
    print("\n--- Running Test Suite ---")
    logging.info("Handing off to test_runner.py")
    try:
        # Use -u for unbuffered output, run test_runner.py non-interactively
        process = subprocess.run(
            [sys.executable, "-u", "test_runner.py"],
            input="3\n", text=True, check=True
        )
    except FileNotFoundError:
        logging.error("test_runner.py not found.")
        print("Error: 'test_runner.py' not found in the current directory.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Test runner exited with an error: {e}")
        print(f"The test runner failed. See '{LOG_FILE_NAME}' for details.")


def main_menu():
    """Displays the main interactive menu and handles user input."""
    while True:
        print("\n--- Neural Connector Main Menu ---")
        print("1. List all analyzed modules")
        print("2. Inspect a specific module")
        print("3. Modify a tunable variable")
        print("4. Execute a function")
        print("5. Run tests (generate stubs and execute)")
        print("6. Exit")

        choice = input("Enter your choice: ")
        logging.info(f"User selected menu option: {choice}")

        if choice == '1':
            print("\n--- Analyzed Modules ---")
            for name, data in sorted(module_registry.items()):
                if 'error' in data:
                    print(f"- {name} (Error: {data['error']})")
                else:
                    print(f"- {name} ({len(data['functions'])} funcs, {len(data['classes'])} classes, {len(data['variables'])} vars)")
        elif choice == '2':
            module_name = input("Enter the name of the module to inspect (e.g., 'feature-extractor'): ")
            if module_name in module_registry:
                data = module_registry[module_name]
                print(f"\n--- Details for {module_name} ---")
                print("Functions:")
                for func in data.get('functions', []):
                    print(f"  - {func['name']}({', '.join(func['args'])})")
                print("\nClasses:")
                for cls in data.get('classes', []):
                    print(f"  - {cls['name']}")
                print("\nTunable Variables (UPPERCASE):")
                for var in data.get('variables', []):
                    print(f"  - {var}")
            else:
                print("Module not found.")
                logging.warning(f"User tried to inspect non-existent module: {module_name}")
        elif choice == '3':
            modify_variable_menu()
        elif choice == '4':
            execute_function_menu()
        elif choice == '5':
            run_tests_menu()
        elif choice == '6':
            logging.info("User chose to exit.")
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    """Main entry point for the connector script."""
    setup_logging()
    logging.info("--- Starting Neural Network Connector ---")
    
    # Add current dir to path to allow importing local modules
    sys.path.insert(0, str(CURRENT_DIR))

    check_and_install_dependencies()
    analyze_modules()

    print("\n--- Connector Initialized ---")
    print(f"All actions are being logged to '{LOG_FILE_NAME}'.")
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nCaught interrupt, shutting down.")
        logging.warning("User interrupted the program. Shutting down.")
    finally:
        # Clean up sys.path
        sys.path.pop(0)

    logging.info("--- Connector Shutting Down ---")
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
