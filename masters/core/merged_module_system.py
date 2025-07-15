#!/usr/bin/env python3
"""
Merged Module System
This file combines all module-related functionality from:
- base_module.py
- module_analyzer-3.py
- module_analyzer.py
- module_loader-27.py
- module_loader.py
"""

import os
import sys
import ast
import json
import sqlite3
import inspect
import logging
import importlib
import importlib.util
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
import re

# =============================================================================
# BASE MODULE FUNCTIONALITY (from base_module.py)
# =============================================================================

class BaseModule(ABC):
    """
    Abstract Base Class for all modules in the system.
    It defines a standard interface for initialization, execution,
    and parameter management.
    """
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self._tunable_parameters = {}
        self.logger.info(f"Module '{self.__class__.__name__}' initialized.")

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        The main execution method for the module.
        This method must be implemented by all subclasses.
        """
        pass

    def get_tunable_parameters(self) -> dict:
        """
        Returns a dictionary of parameters that can be tuned by the
        external neural network.
        """
        self.logger.debug(f"Fetching tunable parameters: {self._tunable_parameters}")
        return self._tunable_parameters

    def set_tunable_parameters(self, params: dict):
        """
        Sets the tunable parameters for the module.
        """
        self.logger.info(f"Updating tunable parameters with: {params}")
        for key, value in params.items():
            if key in self._tunable_parameters:
                self._tunable_parameters[key] = value
                self.logger.debug(f"Set parameter '{key}' to '{value}'")
            else:
                self.logger.warning(f"Attempted to set unknown parameter '{key}'")

    def _register_tunable_parameter(self, name: str, default_value):
        """
        Registers a parameter as tunable.
        """
        self._tunable_parameters[name] = default_value
        self.logger.debug(f"Registered tunable parameter '{name}' with default value '{default_value}'")


# =============================================================================
# SIMPLE MODULE ANALYZER (from module_analyzer-3.py)
# =============================================================================

class SimpleModuleAnalyzer:
    """Simple module analyzer that extracts basic information from Python files"""
    def __init__(self, file_path):
        self.file_path = file_path
        self.functions = []
        self.classes = []
        # Note: Assuming logger is available globally or through a different import
        self.log = logging.getLogger("SimpleModuleAnalyzer")

    def analyze(self):
        """
        Analyzes the Python file to extract information about its functions and classes.
        """
        self.log.info(f"Analyzing module: {self.file_path}")
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
            self.log.error(f"Could not analyze module {self.file_path}: {e}")
        
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


# =============================================================================
# ADVANCED MODULE ANALYZER (from module_analyzer.py)
# =============================================================================

# Note: The following imports are assumed to be available or need to be defined
# from .logger import logger, LogChannel
# from .node_base import BaseNode, AtomicNode, CompositeNode, NodeMetadata, NodeInput, NodeOutput, NodeType

# For now, we'll create placeholder classes for missing dependencies
class LogChannel:
    MODULE = "MODULE"

class NodeType:
    ATOMIC = "ATOMIC"
    COMPOSITE = "COMPOSITE"

@dataclass
class NodeMetadata:
    name: str
    type: str
    description: str
    version: str
    tags: List[str]
    capabilities: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

class BaseNode:
    def __init__(self, metadata: NodeMetadata):
        self.metadata = metadata
    
    def connect(self, target, connection_type):
        pass

class AtomicNode(BaseNode):
    def __init__(self, func: Callable, metadata: NodeMetadata):
        super().__init__(metadata)
        self.func = func

class CompositeNode(BaseNode):
    def __init__(self, metadata: NodeMetadata):
        super().__init__(metadata)
        self.nodes = {}
    
    def add_node(self, node: BaseNode, name: str):
        self.nodes[name] = node

# Create a logger instance if not available
logger = logging.getLogger("AdvancedModuleAnalyzer")
logger.info = lambda channel, msg: logger.info(f"[{channel}] {msg}")
logger.debug = lambda channel, msg: logger.debug(f"[{channel}] {msg}")
logger.error = lambda channel, msg: logger.error(f"[{channel}] {msg}")
logger.warning = lambda channel, msg: logger.warning(f"[{channel}] {msg}")
logger.success = lambda channel, msg: logger.info(f"[{channel}] ✓ {msg}")

@dataclass
class FunctionInfo:
    """Information about a function"""
    name: str
    module: str
    file_path: str
    signature: str
    docstring: str
    parameters: List[str]
    return_annotation: Optional[str]
    dependencies: Set[str] = field(default_factory=set)
    calls: Set[str] = field(default_factory=set)
    complexity: int = 0
    lines_of_code: int = 0
    is_class_method: bool = False
    class_name: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'module': self.module,
            'file_path': self.file_path,
            'signature': self.signature,
            'docstring': self.docstring,
            'parameters': self.parameters,
            'return_annotation': self.return_annotation,
            'dependencies': list(self.dependencies),
            'calls': list(self.calls),
            'complexity': self.complexity,
            'lines_of_code': self.lines_of_code,
            'is_class_method': self.is_class_method,
            'class_name': self.class_name,
            'decorators': self.decorators
        }

@dataclass
class ClassInfo:
    """Information about a class"""
    name: str
    module: str
    file_path: str
    docstring: str
    base_classes: List[str]
    methods: List[FunctionInfo]
    attributes: List[str]
    is_abstract: bool = False
    decorators: List[str] = field(default_factory=list)

@dataclass
class ModuleInfo:
    """Information about a module"""
    name: str
    file_path: str
    docstring: str
    imports: Set[str] = field(default_factory=set)
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    global_vars: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)

class AdvancedModuleAnalyzer:
    """Analyzes Python modules and extracts metadata"""
    
    def __init__(self):
        """Initialize analyzer"""
        self.modules: Dict[str, ModuleInfo] = {}
        self.functions: Dict[str, FunctionInfo] = {}
        self.classes: Dict[str, ClassInfo] = {}
        self.import_graph: Dict[str, Set[str]] = {}
        self.call_graph: Dict[str, Set[str]] = {}
        
        logger.info(LogChannel.MODULE, "Module analyzer initialized")
    
    def analyze_file(self, file_path: str) -> Optional[ModuleInfo]:
        """Analyze a single Python file"""
        try:
            path = Path(file_path)
            if not path.exists() or not path.suffix == '.py':
                return None
            
            logger.debug(LogChannel.MODULE, f"Analyzing file: {file_path}")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=file_path)
            
            # Extract module info
            module_name = path.stem
            module_info = ModuleInfo(
                name=module_name,
                file_path=str(path.absolute()),
                docstring=ast.get_docstring(tree) or ""
            )
            
            # Analyze AST
            self._analyze_ast(tree, module_info)
            
            # Store module info
            self.modules[module_name] = module_info
            
            return module_info
            
        except Exception as e:
            logger.error(LogChannel.MODULE, f"Failed to analyze file {file_path}: {str(e)}")
            return None
    
    def analyze_directory(self, directory: str, recursive: bool = True) -> Dict[str, ModuleInfo]:
        """Analyze all Python files in a directory"""
        logger.info(LogChannel.MODULE, f"Analyzing directory: {directory}")
        
        analyzed = {}
        
        for root, dirs, files in os.walk(directory):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', '.git', 'venv', '.venv', 'env',
                'node_modules', '.pytest_cache', 'build', 'dist'
            }]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    module_info = self.analyze_file(file_path)
                    if module_info:
                        analyzed[module_info.name] = module_info
            
            if not recursive:
                break
        
        logger.info(LogChannel.MODULE, f"Analyzed {len(analyzed)} modules")
        return analyzed
    
    def _analyze_ast(self, tree: ast.AST, module_info: ModuleInfo):
        """Analyze AST and extract information"""
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_info.imports.add(alias.name)
                    module_info.dependencies.add(alias.name.split('.')[0])
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_info.imports.add(node.module)
                    module_info.dependencies.add(node.module.split('.')[0])
        
        # Extract top-level elements
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_function(node, module_info.name, module_info.file_path)
                module_info.functions.append(func_info)
                self.functions[f"{module_info.name}.{func_info.name}"] = func_info
            
            elif isinstance(node, ast.ClassDef):
                class_info = self._analyze_class(node, module_info.name, module_info.file_path)
                module_info.classes.append(class_info)
                self.classes[f"{module_info.name}.{class_info.name}"] = class_info
            
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_info.global_vars.append(target.id)
    
    def _analyze_function(self, node: ast.FunctionDef, module_name: str, 
                         file_path: str, class_name: Optional[str] = None) -> FunctionInfo:
        """Analyze a function definition"""
        # Get signature
        sig_parts = []
        for i, arg in enumerate(node.args.args):
            param = arg.arg
            if i < len(node.args.defaults):
                default_idx = i - (len(node.args.args) - len(node.args.defaults))
                if default_idx >= 0:
                    param += "=..."
            sig_parts.append(param)
        
        signature = f"{node.name}({', '.join(sig_parts)})"
        
        # Get return annotation
        return_annotation = None
        if node.returns:
            return_annotation = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # Get decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(f"{decorator.value.id if isinstance(decorator.value, ast.Name) else '...'}.{decorator.attr}")
        
        # Create function info
        func_info = FunctionInfo(
            name=node.name,
            module=module_name,
            file_path=file_path,
            signature=signature,
            docstring=ast.get_docstring(node) or "",
            parameters=[arg.arg for arg in node.args.args],
            return_annotation=return_annotation,
            is_class_method=class_name is not None,
            class_name=class_name,
            decorators=decorators,
            lines_of_code=node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
        )
        
        # Analyze function body
        self._analyze_function_body(node, func_info)
        
        return func_info
    
    def _analyze_function_body(self, node: ast.FunctionDef, func_info: FunctionInfo):
        """Analyze function body for calls and complexity"""
        class CallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.calls = set()
                self.complexity = 1  # Base complexity
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    self.calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        self.calls.add(f"{node.func.value.id}.{node.func.attr}")
                self.generic_visit(node)
            
            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_ExceptHandler(self, node):
                self.complexity += 1
                self.generic_visit(node)
        
        visitor = CallVisitor()
        visitor.visit(node)
        
        func_info.calls = visitor.calls
        func_info.complexity = visitor.complexity
    
    def _analyze_class(self, node: ast.ClassDef, module_name: str, file_path: str) -> ClassInfo:
        """Analyze a class definition"""
        # Get base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(ast.unparse(base) if hasattr(ast, 'unparse') else str(base))
        
        # Get decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
        
        # Create class info
        class_info = ClassInfo(
            name=node.name,
            module=module_name,
            file_path=file_path,
            docstring=ast.get_docstring(node) or "",
            base_classes=base_classes,
            methods=[],
            attributes=[],
            is_abstract='ABC' in base_classes or 'abc.ABC' in base_classes,
            decorators=decorators
        )
        
        # Analyze class body
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item, module_name, file_path, node.name)
                class_info.methods.append(method_info)
                self.functions[f"{module_name}.{node.name}.{method_info.name}"] = method_info
            
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info.attributes.append(target.id)
        
        return class_info
    
    def get_function_dependencies(self, function_name: str) -> Set[str]:
        """Get all dependencies for a function"""
        if function_name not in self.functions:
            return set()
        
        func_info = self.functions[function_name]
        dependencies = func_info.dependencies.copy()
        
        # Add dependencies from called functions
        for call in func_info.calls:
            if call in self.functions:
                dependencies.update(self.get_function_dependencies(call))
        
        return dependencies
    
    def create_function_node(self, function_name: str) -> Optional[BaseNode]:
        """Create a neural network node from a function"""
        if function_name not in self.functions:
            logger.error(LogChannel.MODULE, f"Function not found: {function_name}")
            return None
        
        func_info = self.functions[function_name]
        
        try:
            # Load the module
            module_path = func_info.file_path
            spec = importlib.util.spec_from_file_location(func_info.module, module_path)
            if not spec or not spec.loader:
                raise ImportError(f"Cannot load module from {module_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[func_info.module] = module
            spec.loader.exec_module(module)
            
            # Get the function
            if func_info.is_class_method and func_info.class_name:
                # Get class first
                cls = getattr(module, func_info.class_name)
                # For now, we'll skip instance methods that require instantiation
                if func_info.name == '__init__':
                    return None
                # Try to get class method or static method
                func = getattr(cls, func_info.name, None)
                if func is None:
                    return None
            else:
                func = getattr(module, func_info.name)
            
            # Create node metadata
            metadata = NodeMetadata(
                name=function_name,
                type=NodeType.ATOMIC,
                description=func_info.docstring,
                version="1.0.0",
                tags=["auto-wrapped", func_info.module],
                capabilities=[f"complexity:{func_info.complexity}"],
                parameters={
                    'source_file': func_info.file_path,
                    'signature': func_info.signature,
                    'parameters': func_info.parameters
                }
            )
            
            # Create atomic node
            node = AtomicNode(func, metadata)
            
            logger.success(LogChannel.MODULE, f"Created node for function: {function_name}")
            return node
            
        except Exception as e:
            logger.error(LogChannel.MODULE, f"Failed to create node for {function_name}: {str(e)}")
            traceback.print_exc()
            return None
    
    def create_module_node(self, module_name: str) -> Optional[CompositeNode]:
        """Create a composite node from an entire module"""
        if module_name not in self.modules:
            logger.error(LogChannel.MODULE, f"Module not found: {module_name}")
            return None
        
        module_info = self.modules[module_name]
        
        # Create composite node
        metadata = NodeMetadata(
            name=f"module_{module_name}",
            type=NodeType.COMPOSITE,
            description=module_info.docstring or f"Auto-wrapped module: {module_name}",
            version="1.0.0",
            tags=["auto-wrapped", "module", module_name],
            parameters={
                'source_file': module_info.file_path,
                'function_count': len(module_info.functions),
                'class_count': len(module_info.classes)
            }
        )
        
        composite_node = CompositeNode(metadata)
        
        # Add function nodes
        added_count = 0
        for func_info in module_info.functions:
            func_name = f"{module_name}.{func_info.name}"
            func_node = self.create_function_node(func_name)
            if func_node:
                composite_node.add_node(func_node, func_info.name)
                added_count += 1
        
        if added_count == 0:
            logger.warning(LogChannel.MODULE, f"No functions could be wrapped from module: {module_name}")
            return None
        
        logger.success(LogChannel.MODULE, f"Created composite node for module: {module_name} ({added_count} functions)")
        return composite_node
    
    def generate_node_graph(self) -> Dict[str, Any]:
        """Generate a graph representation of all analyzed modules"""
        nodes = []
        edges = []
        
        # Add function nodes
        for func_name, func_info in self.functions.items():
            nodes.append({
                'id': func_name,
                'type': 'function',
                'module': func_info.module,
                'complexity': func_info.complexity,
                'loc': func_info.lines_of_code
            })
            
            # Add edges for function calls
            for call in func_info.calls:
                if call in self.functions:
                    edges.append({
                        'source': func_name,
                        'target': call,
                        'type': 'calls'
                    })
        
        # Add module nodes
        for module_name, module_info in self.modules.items():
            nodes.append({
                'id': module_name,
                'type': 'module',
                'functions': len(module_info.functions),
                'classes': len(module_info.classes)
            })
            
            # Add edges for imports
            for dep in module_info.dependencies:
                if dep in self.modules:
                    edges.append({
                        'source': module_name,
                        'target': dep,
                        'type': 'imports'
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'total_modules': len(self.modules),
                'total_functions': len(self.functions),
                'total_classes': len(self.classes)
            }
        }
    
    def export_analysis(self, output_file: str):
        """Export analysis results to JSON file"""
        data = {
            'modules': {name: {
                'name': info.name,
                'file_path': info.file_path,
                'docstring': info.docstring,
                'imports': list(info.imports),
                'dependencies': list(info.dependencies),
                'functions': [f.to_dict() for f in info.functions],
                'classes': [{
                    'name': c.name,
                    'docstring': c.docstring,
                    'base_classes': c.base_classes,
                    'methods': [m.to_dict() for m in c.methods],
                    'attributes': c.attributes
                } for c in info.classes]
            } for name, info in self.modules.items()},
            'graph': self.generate_node_graph()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(LogChannel.MODULE, f"Analysis exported to: {output_file}")


class NodeFactory:
    """Factory for creating neural network nodes from analyzed modules"""
    
    def __init__(self, analyzer: AdvancedModuleAnalyzer):
        """Initialize factory"""
        self.analyzer = analyzer
        self.created_nodes: Dict[str, BaseNode] = {}
        
        logger.info(LogChannel.MODULE, "Node factory initialized")
    
    def create_all_nodes(self) -> Dict[str, BaseNode]:
        """Create nodes for all analyzed functions"""
        created = {}
        
        for func_name in self.analyzer.functions:
            node = self.analyzer.create_function_node(func_name)
            if node:
                created[func_name] = node
                self.created_nodes[func_name] = node
        
        logger.info(LogChannel.MODULE, f"Created {len(created)} function nodes")
        return created
    
    def create_module_nodes(self) -> Dict[str, CompositeNode]:
        """Create composite nodes for all modules"""
        created = {}
        
        for module_name in self.analyzer.modules:
            node = self.analyzer.create_module_node(module_name)
            if node:
                created[module_name] = node
                self.created_nodes[f"module_{module_name}"] = node
        
        logger.info(LogChannel.MODULE, f"Created {len(created)} module nodes")
        return created
    
    def get_node(self, name: str) -> Optional[BaseNode]:
        """Get a created node by name"""
        return self.created_nodes.get(name)
    
    def connect_nodes_by_dependencies(self):
        """Automatically connect nodes based on dependencies"""
        connected = 0
        
        for func_name, func_info in self.analyzer.functions.items():
            if func_name not in self.created_nodes:
                continue
            
            source_node = self.created_nodes[func_name]
            
            # Connect to called functions
            for call in func_info.calls:
                if call in self.created_nodes:
                    target_node = self.created_nodes[call]
                    source_node.connect(target_node, "calls")
                    connected += 1
        
        logger.info(LogChannel.MODULE, f"Created {connected} automatic connections")


# =============================================================================
# DATABASE MODULE LOADER (from module_loader-27.py)
# =============================================================================

class DatabaseModuleLoader:
    """
    Dynamically loads modules and their components using the SQLite registry.
    """
    def __init__(self, db_path="module_registry.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger("DatabaseModuleLoader")
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
                self.logger.error(f"'{callable_name}' not found in module {file_path}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to dynamically load from {file_path}: {e}")
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


# =============================================================================
# SIMPLE MODULE LOADER (from module_loader.py)
# =============================================================================

class SimpleModuleLoader:
    """Simple module loader that loads modules from file paths"""
    def __init__(self, modules_to_load):
        self.modules_to_load = modules_to_load
        self.module_registry = {}
        self.log = logging.getLogger("SimpleModuleLoader")

    def load_modules(self):
        self.log.info(f"Attempting to import {len(self.modules_to_load)} discovered modules...")
        for module_name, file_path in self.modules_to_load.items():
            module = None
            try:
                # First, try to import it as a standard module (for installed packages)
                module = importlib.import_module(module_name)
                self.log.info(f"Successfully imported installed module: {module_name}")
            except ImportError:
                self.log.warning(f"Module '{module_name}' not found via standard import. Attempting to load from file.")
                try:
                    # Fallback for standalone scripts or modules not in packages
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec:
                        module = importlib.util.module_from_spec(spec)
                        # Add the script's directory to the path for its local imports
                        sys.path.insert(0, os.path.dirname(file_path))
                        spec.loader.exec_module(module)
                        sys.path.pop(0) # Clean up path
                        self.log.info(f"Successfully loaded module '{module_name}' directly from file.")
                    else:
                        self.log.error(f"Could not create a module spec for {file_path}")
                        continue
                except Exception as e:
                    self.log.error(f"Failed to load module '{module_name}' from file: {e}")
                    continue
            except Exception as e:
                self.log.error(f"An unexpected error occurred while importing module {module_name}: {e}")
                continue

            if module:
                try:
                    analyzer = SimpleModuleAnalyzer(file_path)
                    analysis_result = analyzer.analyze()
                    self.module_registry[module_name] = {
                        "analysis": analysis_result,
                        "module": module,
                        "file_path": file_path
                    }
                    self.log.debug(f"Successfully registered module: {module_name}")
                except Exception as e:
                    self.log.error(f"Failed to analyze module {module_name} after loading: {e}")

        self.log.info(f"Module loading complete. Successfully loaded {len(self.module_registry)} modules.")
        return self.module_registry


# =============================================================================
# UNIFIED MODULE SYSTEM
# =============================================================================

class UnifiedModuleSystem:
    """
    Unified module system that combines all module functionality
    """
    def __init__(self, use_database: bool = False, db_path: str = "module_registry.db"):
        """
        Initialize the unified module system
        
        Args:
            use_database: Whether to use database-based module loading
            db_path: Path to the SQLite database (if use_database is True)
        """
        self.logger = logging.getLogger("UnifiedModuleSystem")
        self.use_database = use_database
        
        # Initialize components
        self.advanced_analyzer = AdvancedModuleAnalyzer()
        self.simple_analyzer = None  # Created on demand
        
        if use_database:
            self.module_loader = DatabaseModuleLoader(db_path)
        else:
            self.module_loader = None  # SimpleModuleLoader needs modules_to_load
        
        self.node_factory = NodeFactory(self.advanced_analyzer)
        
        # Registry of loaded modules
        self.loaded_modules = {}
        self.created_nodes = {}
    
    def analyze_directory(self, directory: str, recursive: bool = True, 
                         use_advanced: bool = True) -> Dict[str, Any]:
        """
        Analyze all Python files in a directory
        
        Args:
            directory: Directory path to analyze
            recursive: Whether to analyze subdirectories
            use_advanced: Whether to use advanced analyzer (vs simple)
        
        Returns:
            Dictionary of analyzed modules
        """
        if use_advanced:
            return self.advanced_analyzer.analyze_directory(directory, recursive)
        else:
            # Use simple analyzer for basic analysis
            analyzed = {}
            for root, dirs, files in os.walk(directory):
                if not recursive:
                    dirs.clear()
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        analyzer = SimpleModuleAnalyzer(file_path)
                        result = analyzer.analyze()
                        if result:
                            analyzed[result['module_name']] = result
            
            return analyzed
    
    def load_module(self, file_path: str, module_name: Optional[str] = None) -> Optional[Any]:
        """
        Load a module from file path
        
        Args:
            file_path: Path to the Python file
            module_name: Optional module name (defaults to file stem)
        
        Returns:
            Loaded module object or None
        """
        if not module_name:
            module_name = Path(file_path).stem
        
        try:
            if self.use_database and isinstance(self.module_loader, DatabaseModuleLoader):
                # Try to load from database
                module_info = self.module_loader.find_module_by_path(file_path)
                if module_info:
                    # Load the module
                    return self.module_loader.load_callable(file_path)
            
            # Direct file loading
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.loaded_modules[module_name] = module
                return module
            
        except Exception as e:
            self.logger.error(f"Failed to load module {module_name} from {file_path}: {e}")
        
        return None
    
    def create_node_from_function(self, function_path: str) -> Optional[BaseNode]:
        """
        Create a neural network node from a function
        
        Args:
            function_path: Full path to function (e.g., "module.function" or "module.Class.method")
        
        Returns:
            Created node or None
        """
        node = self.advanced_analyzer.create_function_node(function_path)
        if node:
            self.created_nodes[function_path] = node
        return node
    
    def create_nodes_from_module(self, module_name: str) -> Optional[CompositeNode]:
        """
        Create a composite node from an entire module
        
        Args:
            module_name: Name of the module
        
        Returns:
            Created composite node or None
        """
        node = self.advanced_analyzer.create_module_node(module_name)
        if node:
            self.created_nodes[f"module_{module_name}"] = node
        return node
    
    def export_analysis(self, output_file: str):
        """Export complete analysis to JSON file"""
        self.advanced_analyzer.export_analysis(output_file)
    
    def get_module_info(self, module_name: str) -> Optional[ModuleInfo]:
        """Get information about an analyzed module"""
        return self.advanced_analyzer.modules.get(module_name)
    
    def get_function_info(self, function_path: str) -> Optional[FunctionInfo]:
        """Get information about an analyzed function"""
        return self.advanced_analyzer.functions.get(function_path)
    
    def get_class_info(self, class_path: str) -> Optional[ClassInfo]:
        """Get information about an analyzed class"""
        return self.advanced_analyzer.classes.get(class_path)
    
    def connect_nodes_automatically(self):
        """Automatically connect nodes based on dependencies"""
        self.node_factory.connect_nodes_by_dependencies()
    
    def get_dependency_graph(self) -> Dict[str, Any]:
        """Get the complete dependency graph"""
        return self.advanced_analyzer.generate_node_graph()


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔧 Unified Module System Demo")
    print("=" * 60)
    
    # Create unified system
    system = UnifiedModuleSystem(use_database=False)
    
    # Demo directory (adjust path as needed)
    demo_dir = "/media/jarvis/6E7A-FA6E/polar-bear/masters/core/module"
    
    if os.path.exists(demo_dir):
        print(f"\n📁 Analyzing directory: {demo_dir}")
        
        # Analyze directory
        modules = system.analyze_directory(demo_dir, recursive=False)
        
        print(f"\n📊 Found {len(modules)} modules:")
        for name, info in modules.items():
            if isinstance(info, ModuleInfo):
                print(f"  📦 {name}:")
                print(f"     - Functions: {len(info.functions)}")
                print(f"     - Classes: {len(info.classes)}")
                print(f"     - Dependencies: {len(info.dependencies)}")
        
        # Create nodes
        print("\n🔮 Creating neural network nodes...")
        for module_name in modules:
            node = system.create_nodes_from_module(module_name)
            if node:
                print(f"  ✅ Created composite node for: {module_name}")
        
        # Connect nodes
        print("\n🔗 Connecting nodes by dependencies...")
        system.connect_nodes_automatically()
        
        # Export analysis
        output_file = "unified_module_analysis.json"
        system.export_analysis(output_file)
        print(f"\n💾 Analysis exported to: {output_file}")
        
        # Show dependency graph stats
        graph = system.get_dependency_graph()
        print(f"\n📈 Dependency Graph Statistics:")
        print(f"  - Total nodes: {len(graph['nodes'])}")
        print(f"  - Total edges: {len(graph['edges'])}")
        print(f"  - Modules: {graph['stats']['total_modules']}")
        print(f"  - Functions: {graph['stats']['total_functions']}")
        print(f"  - Classes: {graph['stats']['total_classes']}")
    
    else:
        print(f"\n❌ Demo directory not found: {demo_dir}")
    
    print("\n✨ Demo complete!")