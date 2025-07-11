#!/usr/bin/env python3
"""
Module Analyzer and Node Wrapper
Analyzes existing Python modules and creates neural network node wrappers
"""

import os
import sys
import ast
import inspect
import importlib
import importlib.util
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import re
import json
import traceback

from .logger import logger, LogChannel
from .node_base import BaseNode, AtomicNode, CompositeNode, NodeMetadata, NodeInput, NodeOutput, NodeType


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


class ModuleAnalyzer:
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
    
    def __init__(self, analyzer: ModuleAnalyzer):
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


if __name__ == "__main__":
    # Demo the module analyzer
    print("🔍 Module Analyzer Demo")
    print("=" * 50)
    
    # Create analyzer
    analyzer = ModuleAnalyzer()
    
    # Analyze a directory
    test_dir = "/home/jarvis/Documents/GitHub/polar-bear/modules/iteration6-lab-framework/modules"
    if os.path.exists(test_dir):
        print(f"\nAnalyzing directory: {test_dir}")
        modules = analyzer.analyze_directory(test_dir)
        
        print(f"\nFound {len(modules)} modules:")
        for name, info in modules.items():
            print(f"  📦 {name}: {len(info.functions)} functions, {len(info.classes)} classes")
        
        # Create nodes
        print("\nCreating neural network nodes...")
        factory = NodeFactory(analyzer)
        
        func_nodes = factory.create_all_nodes()
        print(f"  Created {len(func_nodes)} function nodes")
        
        module_nodes = factory.create_module_nodes()
        print(f"  Created {len(module_nodes)} module nodes")
        
        # Connect nodes
        factory.connect_nodes_by_dependencies()
        
        # Export analysis
        analyzer.export_analysis("module_analysis.json")
        print("\n✅ Analysis complete! Results saved to module_analysis.json")
    else:
        print(f"\n❌ Test directory not found: {test_dir}")