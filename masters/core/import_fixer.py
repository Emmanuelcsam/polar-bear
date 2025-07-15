import os
import ast
import sys
from typing import Dict, List, Set

# Add astor as a dependency for rewriting the AST
try:
    import astor
except ImportError:
    print("ERROR: This script requires the 'astor' package. Please install it first:")
    print("pip install astor")
    sys.exit(1)

# Assuming this script is in core, the project root is two levels up.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

class ImportFixer(ast.NodeTransformer):
    """
    An AST NodeTransformer that corrects relative imports based on a map
    of function definitions to their actual modules.
    """
    def __init__(self, func_to_module_map: Dict[str, str], current_file: str):
        self.func_to_module_map = func_to_module_map
        self.current_file_module = os.path.splitext(os.path.basename(current_file))[0].replace('-', '_')
        self.file_changed = False

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        # We only care about relative imports within the project
        if node.level == 0: # not a relative import
            return node

        # If the module name is already correct, do nothing
        if node.module and node.module in self.func_to_module_map.values():
            return node

        for alias in node.names:
            imported_name = alias.name
            if imported_name in self.func_to_module_map:
                correct_module = self.func_to_module_map[imported_name]
                
                # Don't try to import a function from its own file
                if correct_module == self.current_file_module:
                    continue

                if node.module != correct_module:
                    print(f"  [FIX] In '{self.current_file_module}.py':")
                    print(f"    - Changing 'from .{node.module} import {imported_name}'")
                    print(f"    + to 'from .{correct_module} import {imported_name}'")
                    node.module = correct_module
                    self.file_changed = True
        return node

def get_py_files(root_dir: str) -> List[str]:
    """Gets all python files in a directory, excluding the framework itself."""
    py_files = []
    framework_dir = os.path.join(root_dir, 'neural_framework')
    for root, _, files in os.walk(root_dir):
        if root.startswith(framework_dir):
            continue
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def map_functions_to_modules(py_files: List[str]) -> Dict[str, str]:
    """
    Parses all python files to create a mapping of {function_name: module_name}.
    """
    func_map = {}
    print("Mapping functions to modules...")
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=file_path)
            
            module_name = os.path.splitext(os.path.basename(file_path))[0].replace('-', '_')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name in func_map:
                        print(f"  [WARN] Function '{node.name}' is defined in multiple files. "
                              f"Using '{module_name}' over '{func_map[node.name]}'.")
                    func_map[node.name] = module_name
        except Exception as e:
            print(f"Could not parse {file_path}: {e}")
    print(f"Mapped {len(func_map)} unique function definitions.")
    return func_map

def run_fixer(project_root: str):
    """
    Main function to run the import fixing process.
    """
    all_files = get_py_files(project_root)
    func_map = map_functions_to_modules(all_files)

    print("\nScanning files for incorrect imports...")
    total_files_changed = 0
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                tree = ast.parse(source_code, filename=file_path)
            
            fixer = ImportFixer(func_map, file_path)
            new_tree = fixer.visit(tree)

            if fixer.file_changed:
                # Use astor to write the modified AST back to source code
                new_source_code = astor.to_source(new_tree)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_source_code)
                total_files_changed += 1
                print(f"  Successfully fixed and saved {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Could not process {file_path}: {e}")
    
    if total_files_changed > 0:
        print(f"\nFixer complete. Changed {total_files_changed} files.")
    else:
        print("\nFixer complete. No incorrect relative imports found.")


if __name__ == '__main__':
    run_fixer(PROJECT_ROOT)
