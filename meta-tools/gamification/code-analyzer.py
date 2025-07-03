#!/usr/bin/env python3
"""
Code Dependency Analyzer - Analyze code structure and dependencies
Maps imports, function calls, and code relationships
"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict, Counter
import json
from datetime import datetime

class CodeAnalyzer:
    def __init__(self, path="."):
        self.root = Path(path).resolve()
        self.dependencies = defaultdict(set)
        self.imports = defaultdict(set)
        self.functions = defaultdict(list)
        self.classes = defaultdict(list)
        self.complexity = {}
        self.code_stats = {
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'files_analyzed': 0
        }
        
    def analyze_code(self):
        """Analyze all code files in the directory"""
        print(f"🔍 Analyzing code in: {self.root}")
        print("-" * 50)
        
        # Language analyzers
        analyzers = {
            '.py': self._analyze_python,
            '.js': self._analyze_javascript,
            '.java': self._analyze_java,
            '.cpp': self._analyze_cpp,
            '.c': self._analyze_cpp,
            '.go': self._analyze_go
        }
        
        code_files = []
        
        # Find all code files
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
            
            for file in files:
                path = Path(root) / file
                ext = path.suffix.lower()
                
                if ext in analyzers:
                    code_files.append((path, ext))
        
        print(f"Found {len(code_files)} code files to analyze")
        
        # Analyze each file
        for i, (path, ext) in enumerate(code_files):
            print(f"\rAnalyzing file {i+1}/{len(code_files)}...", end='')
            
            try:
                analyzer = analyzers[ext]
                analyzer(path)
                self.code_stats['files_analyzed'] += 1
            except Exception as e:
                print(f"\nError analyzing {path}: {e}")
        
        print(f"\n✓ Analyzed {self.code_stats['files_analyzed']} files")
        
    def _analyze_python(self, filepath):
        """Analyze Python file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Count lines
        self._count_lines(lines)
        
        # Parse AST
        try:
            tree = ast.parse(content)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        self.imports[str(filepath)].add(module)
                        self.dependencies[module].add(str(filepath))
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        self.imports[str(filepath)].add(module)
                        self.dependencies[module].add(str(filepath))
                
                elif isinstance(node, ast.FunctionDef):
                    self.functions[str(filepath)].append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': len(node.args.args),
                        'decorators': len(node.decorator_list)
                    })
                
                elif isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    self.classes[str(filepath)].append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': methods,
                        'bases': len(node.bases)
                    })
            
            # Calculate complexity (simple version based on control structures)
            complexity = 1  # Base complexity
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            self.complexity[str(filepath)] = complexity
            
        except SyntaxError:
            pass
    
    def _analyze_javascript(self, filepath):
        """Analyze JavaScript file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
        
        self._count_lines(lines)
        
        # Simple regex-based analysis for JS
        # Find imports
        import_patterns = [
            r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'require\([\'"]([^\'"]+)[\'"]\)',
            r'import\s+[\'"]([^\'"]+)[\'"]'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                module = match.split('/')[-1].replace('.js', '')
                self.imports[str(filepath)].add(module)
                self.dependencies[module].add(str(filepath))
        
        # Find functions
        func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|function))'
        matches = re.findall(func_pattern, content)
        for match in matches:
            func_name = match[0] or match[1]
            if func_name:
                self.functions[str(filepath)].append({'name': func_name})
        
        # Find classes
        class_pattern = r'class\s+(\w+)'
        matches = re.findall(class_pattern, content)
        for match in matches:
            self.classes[str(filepath)].append({'name': match})
    
    def _analyze_java(self, filepath):
        """Analyze Java file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
        
        self._count_lines(lines)
        
        # Find imports
        import_pattern = r'import\s+(?:static\s+)?([a-zA-Z0-9_.]+);'
        matches = re.findall(import_pattern, content)
        for match in matches:
            package = match.split('.')[-1]
            self.imports[str(filepath)].add(package)
            self.dependencies[package].add(str(filepath))
        
        # Find classes
        class_pattern = r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)'
        matches = re.findall(class_pattern, content)
        for match in matches:
            self.classes[str(filepath)].append({'name': match})
    
    def _analyze_cpp(self, filepath):
        """Analyze C/C++ file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
        
        self._count_lines(lines)
        
        # Find includes
        include_pattern = r'#include\s+[<"]([^>"]+)[>"]'
        matches = re.findall(include_pattern, content)
        for match in matches:
            header = match.split('/')[-1].replace('.h', '').replace('.hpp', '')
            self.imports[str(filepath)].add(header)
            self.dependencies[header].add(str(filepath))
    
    def _analyze_go(self, filepath):
        """Analyze Go file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
        
        self._count_lines(lines)
        
        # Find imports
        import_pattern = r'import\s+(?:\(([^)]+)\)|"([^"]+)")'
        matches = re.findall(import_pattern, content)
        for match in matches:
            imports = match[0] if match[0] else match[1]
            for imp in imports.split('\n'):
                imp = imp.strip().strip('"')
                if imp:
                    package = imp.split('/')[-1]
                    self.imports[str(filepath)].add(package)
                    self.dependencies[package].add(str(filepath))
    
    def _count_lines(self, lines):
        """Count different types of lines"""
        self.code_stats['total_lines'] += len(lines)
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                self.code_stats['blank_lines'] += 1
            elif stripped.startswith(('#', '//', '/*', '*', '*/')):
                self.code_stats['comment_lines'] += 1
            else:
                self.code_stats['code_lines'] += 1
    
    def display_analysis(self):
        """Display code analysis results"""
        print("\n" + "="*80)
        print("📊 CODE ANALYSIS RESULTS")
        print("="*80)
        
        # Overall stats
        print("\n📈 Code Statistics:")
        print(f"  Total lines: {self.code_stats['total_lines']:,}")
        print(f"  Code lines: {self.code_stats['code_lines']:,} ({self._percent(self.code_stats['code_lines'], self.code_stats['total_lines'])}%)")
        print(f"  Comment lines: {self.code_stats['comment_lines']:,} ({self._percent(self.code_stats['comment_lines'], self.code_stats['total_lines'])}%)")
        print(f"  Blank lines: {self.code_stats['blank_lines']:,} ({self._percent(self.code_stats['blank_lines'], self.code_stats['total_lines'])}%)")
        
        # Most imported modules
        print("\n📦 Most Used Dependencies:")
        import_counts = Counter()
        for module, files in self.dependencies.items():
            import_counts[module] = len(files)
        
        for module, count in import_counts.most_common(10):
            print(f"  {module}: used in {count} files")
        
        # Files with most dependencies
        print("\n🔗 Files with Most Dependencies:")
        dep_counts = [(f, len(deps)) for f, deps in self.imports.items()]
        dep_counts.sort(key=lambda x: x[1], reverse=True)
        
        for filepath, count in dep_counts[:10]:
            rel_path = Path(filepath).relative_to(self.root)
            print(f"  {rel_path}: {count} dependencies")
        
        # Complexity analysis
        if self.complexity:
            print("\n🧩 Most Complex Files:")
            complex_files = sorted(self.complexity.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for filepath, complexity in complex_files:
                rel_path = Path(filepath).relative_to(self.root)
                print(f"  {rel_path}: complexity score {complexity}")
        
        # Function/Class counts
        total_functions = sum(len(funcs) for funcs in self.functions.values())
        total_classes = sum(len(classes) for classes in self.classes.values())
        
        print(f"\n🔧 Code Structure:")
        print(f"  Total functions: {total_functions}")
        print(f"  Total classes: {total_classes}")
        
        # Files with most functions
        func_counts = [(f, len(funcs)) for f, funcs in self.functions.items() if funcs]
        if func_counts:
            func_counts.sort(key=lambda x: x[1], reverse=True)
            print("\n📄 Files with Most Functions:")
            for filepath, count in func_counts[:5]:
                rel_path = Path(filepath).relative_to(self.root)
                print(f"  {rel_path}: {count} functions")
        
        # Potential issues
        self._check_issues()
        
        # Save analysis
        self._save_analysis()
    
    def _check_issues(self):
        """Check for potential code issues"""
        print("\n⚠️  Potential Issues:")
        
        issues = []
        
        # Circular dependencies (simple check)
        for module1, files1 in self.dependencies.items():
            for module2, files2 in self.dependencies.items():
                if module1 != module2:
                    # Check if both modules import each other
                    if any(f in files2 for f in files1) and any(f in files1 for f in files2):
                        issues.append(f"Possible circular dependency: {module1} ↔ {module2}")
        
        # Files with too many dependencies
        for filepath, deps in self.imports.items():
            if len(deps) > 20:
                rel_path = Path(filepath).relative_to(self.root)
                issues.append(f"High dependency count: {rel_path} ({len(deps)} dependencies)")
        
        # Very complex files
        for filepath, complexity in self.complexity.items():
            if complexity > 50:
                rel_path = Path(filepath).relative_to(self.root)
                issues.append(f"High complexity: {rel_path} (score: {complexity})")
        
        if issues:
            for issue in issues[:10]:
                print(f"  • {issue}")
        else:
            print("  ✓ No major issues detected")
    
    def _save_analysis(self):
        """Save code analysis results"""
        stats_dir = Path('.project-stats')
        stats_dir.mkdir(exist_ok=True)
        
        analysis_file = stats_dir / f"code_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'path': str(self.root),
            'stats': self.code_stats,
            'dependencies': {k: list(v) for k, v in self.dependencies.items()},
            'top_imports': dict(Counter(module for deps in self.imports.values() for module in deps).most_common(20)),
            'complexity': self.complexity,
            'function_count': sum(len(funcs) for funcs in self.functions.values()),
            'class_count': sum(len(classes) for classes in self.classes.values())
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Analysis saved to: {analysis_file}")
    
    def _percent(self, part, whole):
        """Calculate percentage"""
        return round(100 * part / whole, 1) if whole > 0 else 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze code structure and dependencies')
    parser.add_argument('path', nargs='?', default='.', help='Directory to analyze')
    
    args = parser.parse_args()
    
    analyzer = CodeAnalyzer(args.path)
    analyzer.analyze_code()
    analyzer.display_analysis()

if __name__ == "__main__":
    main()
