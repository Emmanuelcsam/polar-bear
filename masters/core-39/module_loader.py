
import sqlite3
import json
import logging
from pathlib import Path
import importlib.util
import sys

logger = logging.getLogger(__name__)

class ModuleLoader:
    """
    Dynamically loads modules and their components using the SQLite registry.
    """
    def __init__(self, db_path="module_registry.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Module registry database not found at {self.db_path}. Please run the system_analyzer.py first.")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def find_module_by_path(self, file_path: str) -> dict:
        """Finds a module's metadata by its file path."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM modules WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        if row:
            return self._format_row(row)
        return None

    def find_modules_by_function(self, function_name: str) -> list:
        """Finds all modules that contain a function with the given name."""
        cursor = self.conn.cursor()
        # Use JSON functions to search within the 'functions' JSON array
        cursor.execute("""
            SELECT * FROM modules, json_each(modules.functions)
            WHERE json_extract(json_each.value, '$.name') = ?
        """, (function_name,))
        rows = cursor.fetchall()
        return [self._format_row(row) for row in rows]

    def find_modules_by_class(self, class_name: str) -> list:
        """Finds all modules that contain a class with the given name."""
        cursor = self.conn.cursor()
        # Use JSON functions to search within the 'classes' JSON object
        cursor.execute("""
            SELECT * FROM modules, json_each(modules.classes)
            WHERE json_each.key = ?
        """, (class_name,))
        rows = cursor.fetchall()
        return [self._format_row(row) for row in rows]

    def get_all_modules(self) -> list:
        """Retrieves all modules from the registry."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM modules")
        rows = cursor.fetchall()
        return [self._format_row(row) for row in rows]

    def load_callable(self, file_path: str, class_name: str = None, function_name: str = None):
        """
        Dynamically imports a module and returns a callable (class or function).
        """
        if not (class_name or function_name):
            raise ValueError("Must provide either a class_name or a function_name.")

        p = Path(file_path)
        module_name = p.stem
        
        try:
            # Add the file's parent directory to the python path to handle imports
            parent_dir = str(p.parent.resolve())
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                raise ImportError(f"Could not create module spec for {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            callable_name = class_name if class_name else function_name
            if hasattr(module, callable_name):
                return getattr(module, callable_name)
            else:
                logger.error(f"'{callable_name}' not found in module {file_path}")
                return None

        except Exception as e:
            logger.error(f"Failed to dynamically load from {file_path}: {e}")
            # Clean up path modification on failure
            if parent_dir in sys.path and sys.path[0] == parent_dir:
                sys.path.pop(0)
            return None


    def _format_row(self, row: sqlite3.Row) -> dict:
        """Converts a sqlite3.Row object to a dictionary."""
        data = dict(row)
        # Deserialize JSON strings back into Python objects
        for key in ['imports', 'functions', 'classes']:
            if data[key]:
                data[key] = json.loads(data[key])
        return data

    def __del__(self):
        self.conn.close()

if __name__ == '__main__':
    # Example usage of the ModuleLoader
    from core.logging_manager import setup_logging
    setup_logging()
    
    loader = ModuleLoader()
    
    print("--- Module Loader Demo ---")
    
    print("\n1. Finding module by path:")
    # Adjust the path to be relative to the project root for a more realistic example
    example_path = 'modules/artificial-intelligence/system_analyzer.py'
    module_info = loader.find_module_by_path(example_path)
    if module_info:
        print(f"Found: {module_info['file_path']}")
        print(f"  Imports: {len(module_info['imports'])}")
        print(f"  Functions: {[f['name'] for f in module_info['functions']]}")
        print(f"  Classes: {list(module_info['classes'].keys())}")
    else:
        print(f"Module with path '{example_path}' not found.")

    print("\n2. Finding modules with function 'main':")
    modules_with_main = loader.find_modules_by_function('main')
    print(f"Found {len(modules_with_main)} modules with a 'main' function.")
    for m in modules_with_main[:5]: # Print first 5
        print(f"  - {m['file_path']}")

    print("\n3. Dynamically loading the 'CodeAnalyzer' class:")
    CodeAnalyzerClass = loader.load_callable(example_path, class_name='CodeAnalyzer')
    if CodeAnalyzerClass:
        print(f"Successfully loaded class: {CodeAnalyzerClass}")
        # We could now instantiate it: analyzer = CodeAnalyzerClass(Path('.'))
    else:
        print("Failed to load class.")
        
    print("\n--- Demo Complete ---")
