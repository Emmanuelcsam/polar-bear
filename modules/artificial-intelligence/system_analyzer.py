
"""
system_analyzer.py
------------------

This script is the heart of Phase 1: Exhaustive File Analysis.
It performs a deep introspection of the entire codebase to build a
comprehensive module registry.

Functionality:
1.  Scans all directories for Python files ('*.py').
2.  For each file, it uses Abstract Syntax Trees (AST) to:
    -   Extract all function definitions, including their arguments.
    -   Extract all class definitions, including their methods.
    -   Extract all import statements to map dependencies.
3.  It calculates basic complexity metrics for each function/method.
4.  The final output is a detailed 'module_registry.json' file, which
    serves as the foundational knowledge base for the Synapse orchestrator.
"""

import ast
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, List

# Initialize logger from the core logging manager
from core.logging_manager import setup_logging, get_logger

setup_logging()
logger = get_logger("SystemAnalyzer")

class CodeAnalyzer(ast.NodeVisitor):
    """
    An AST node visitor that extracts detailed information from a Python file.
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.metadata: Dict[str, Any] = {
            "file_path": str(self.file_path),
            "imports": [],
            "functions": [],
            "classes": {}
        }

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.metadata["imports"].append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.metadata["imports"].append(node.module)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        is_method = any(isinstance(p, ast.ClassDef) for p in getattr(node, '_parents', []))
        if not is_method:
            self.metadata["functions"].append(self._extract_function_details(node))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        class_info = {
            "name": node.name,
            "methods": [],
            "docstring": ast.get_docstring(node)
        }
        for body_item in node.body:
            if isinstance(body_item, ast.FunctionDef):
                body_item._parents = getattr(node, '_parents', []) + [node]
                class_info["methods"].append(self._extract_function_details(body_item))
        self.metadata["classes"][node.name] = class_info
        self.generic_visit(node)

    def _extract_function_details(self, node: ast.FunctionDef) -> Dict[str, Any]:
        return {
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "docstring": ast.get_docstring(node),
            "complexity": len(node.body)
        }
        
    def analyze(self):
        logger.debug(f"Analyzing {self.file_path}...")
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    for child in ast.iter_child_nodes(node):
                        child._parents = getattr(node, '_parents', []) + [node]
                self.visit(tree)
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
            logger.error(f"Could not analyze {self.file_path}: {e}")
            return None
        return self.metadata

def setup_database(db_path: Path):
    """Sets up the SQLite database and creates the necessary tables."""
    if db_path.exists():
        db_path.unlink() # Start fresh each time
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE modules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            imports TEXT,
            functions TEXT,
            classes TEXT
        )
    ''')
    conn.commit()
    return conn

def scan_and_save_to_db(root_path: Path, conn: sqlite3.Connection):
    """
    Scans the project and saves the analysis results directly to the database.
    """
    logger.info(f"Starting project scan from root: {root_path}")
    py_files = list(root_path.rglob("*.py"))
    logger.info(f"Found {len(py_files)} Python files to analyze.")
    
    cursor = conn.cursor()
    analyzed_count = 0

    for py_file in py_files:
        if any(part in str(py_file) for part in ['venv', '.env', 'build', 'dist', '__pycache__']):
            continue
            
        analyzer = CodeAnalyzer(py_file)
        file_metadata = analyzer.analyze()
        
        if file_metadata:
            try:
                cursor.execute(
                    "INSERT INTO modules (file_path, imports, functions, classes) VALUES (?, ?, ?, ?)",
                    (
                        file_metadata['file_path'],
                        json.dumps(file_metadata['imports']),
                        json.dumps(file_metadata['functions']),
                        json.dumps(file_metadata['classes'])
                    )
                )
                analyzed_count += 1
            except sqlite3.IntegrityError:
                logger.warning(f"File path {file_metadata['file_path']} already exists in database. Skipping.")

    conn.commit()
    logger.info(f"Project scan and analysis complete. Saved {analyzed_count} modules to the database.")
    return analyzed_count

def main():
    """
    Main execution function.
    """
    project_root = Path(__file__).parent.parent.parent
    db_path = Path("module_registry.db")
    
    conn = setup_database(db_path)
    analyzed_count = scan_and_save_to_db(project_root, conn)
    conn.close()
        
    print(f"Analysis complete. Module registry saved to {db_path}")
    print(f"Analyzed and stored {analyzed_count} Python files.")

if __name__ == "__main__":
    main()
