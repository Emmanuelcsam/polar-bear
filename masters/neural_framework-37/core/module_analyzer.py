import ast
import os
from .logger import log

class ModuleAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.functions = []
        self.classes = []

    def analyze(self):
        """
        Analyzes the Python file to extract information about its functions and classes.
        """
        log.info(f"Analyzing module: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=self.file_path)
                
                for node in ast.iter_child_nodes(tree): # Top-level nodes only
                    if isinstance(node, ast.FunctionDef):
                        self.functions.append(self._extract_function_info(node))
                    elif isinstance(node, ast.ClassDef):
                        self.classes.append(self._extract_class_info(node))
        except Exception as e:
            log.error(f"Could not analyze module {self.file_path}: {e}")
        
        return {
            "file_path": self.file_path,
            "module_name": os.path.splitext(os.path.basename(self.file_path))[0],
            "functions": self.functions,
            "classes": self.classes
        }

    def _extract_function_info(self, node):
        """Extracts information from a FunctionDef node."""
        return {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "defaults": len(node.args.defaults),
            "docstring": ast.get_docstring(node) or "No docstring."
        }

    def _extract_class_info(self, node):
        """Extracts information from a ClassDef node."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._extract_function_info(item))
        
        return {
            "name": node.name,
            "methods": methods,
            "docstring": ast.get_docstring(node) or "No docstring."
        }

if __name__ == '__main__':
    # Example Usage:
    # Create a dummy file for testing
    dummy_file = "test_module.py"
    with open(dummy_file, "w") as f:
        f.write('"""This is a test module."""\n\n')
        f.write('class MyClass:\n')
        f.write('    """A test class."""\n')
        f.write('    def my_method(self, arg1, arg2="default"):\n')
        f.write('        """A test method."""\n')
        f.write('        pass\n\n')
        f.write('def my_function(param1):\n')
        f.write('    """A test function."""\n')
        f.write('    return param1\n')

    analyzer = ModuleAnalyzer(dummy_file)
    analysis_result = analyzer.analyze()
    
    import json
    log.info(json.dumps(analysis_result, indent=4))

    # Clean up the dummy file
    os.remove(dummy_file)
