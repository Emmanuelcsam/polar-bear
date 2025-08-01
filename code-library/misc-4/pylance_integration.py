#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pylance Integration Script for Fiber CNN Project
Provides type hints, code analysis, and development tools
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import logging
import importlib
import inspect
from dataclasses import dataclass, field
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TypeAnalysis:
    """Data class for type analysis results"""
    file_path: str
    total_lines: int
    typed_lines: int
    type_coverage: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)

class PylanceAnalyzer:
    """Analyze Python files for Pylance integration"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.analysis_results: Dict[str, TypeAnalysis] = {}
        
    def analyze_file(self, file_path: str) -> TypeAnalysis:
        """Analyze a single Python file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        total_lines = len(lines)
        typed_lines = 0
        issues = []
        suggestions = []
        imports = []
        functions = []
        classes = []
        
        # Simple analysis (in a real implementation, you'd use ast or mypy)
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check for type hints
            if ':' in line and ('->' in line or 'List[' in line or 'Dict[' in line or 
                               'Optional[' in line or 'Tuple[' in line or 'Union[' in line):
                typed_lines += 1
            
            # Check for imports
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
            
            # Check for function definitions
            if line.startswith('def '):
                functions.append(line)
            
            # Check for class definitions
            if line.startswith('class '):
                classes.append(line)
            
            # Check for potential issues
            if 'TODO' in line or 'FIXME' in line:
                issues.append(f"Line {i}: {line}")
            
            if 'print(' in line and not line.startswith('#'):
                suggestions.append(f"Line {i}: Consider using logger instead of print")
        
        type_coverage = (typed_lines / total_lines * 100) if total_lines > 0 else 0
        
        return TypeAnalysis(
            file_path=str(file_path),
            total_lines=total_lines,
            typed_lines=typed_lines,
            type_coverage=type_coverage,
            issues=issues,
            suggestions=suggestions,
            imports=imports,
            functions=functions,
            classes=classes
        )
    
    def analyze_project(self, include_patterns: List[str] = None, 
                       exclude_patterns: List[str] = None) -> Dict[str, TypeAnalysis]:
        """Analyze all Python files in the project"""
        if include_patterns is None:
            include_patterns = ["*.py"]
        
        if exclude_patterns is None:
            exclude_patterns = ["__pycache__", "venv", ".git", "checkpoints", "logs", "temp"]
        
        python_files = []
        for pattern in include_patterns:
            python_files.extend(self.project_root.rglob(pattern))
        
        # Filter out excluded patterns
        filtered_files = []
        for file_path in python_files:
            if not any(exclude in str(file_path) for exclude in exclude_patterns):
                filtered_files.append(file_path)
        
        logger.info(f"Analyzing {len(filtered_files)} Python files...")
        
        for file_path in filtered_files:
            try:
                analysis = self.analyze_file(str(file_path))
                self.analysis_results[str(file_path)] = analysis
                logger.info(f"Analyzed: {file_path} (Type coverage: {analysis.type_coverage:.1f}%)")
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        return self.analysis_results
    
    def generate_type_stubs(self, output_dir: str = "typings") -> None:
        """Generate type stubs for the project"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Use stubgen if available
            subprocess.run([
                sys.executable, "-m", "stubgen", 
                "--output", str(output_path),
                "--recursive", str(self.project_root)
            ], check=True)
            logger.info(f"Type stubs generated in {output_path}")
        except subprocess.CalledProcessError:
            logger.warning("stubgen not available, creating basic stubs...")
            self._create_basic_stubs(output_path)
    
    def _create_basic_stubs(self, output_dir: Path) -> None:
        """Create basic type stubs manually"""
        for file_path, analysis in self.analysis_results.items():
            if analysis.type_coverage < 50:  # Only create stubs for files with low type coverage
                stub_content = self._generate_stub_content(analysis)
                stub_file = output_dir / f"{Path(file_path).stem}.pyi"
                
                with open(stub_file, 'w') as f:
                    f.write(stub_content)
                
                logger.info(f"Created stub: {stub_file}")
    
    def _generate_stub_content(self, analysis: TypeAnalysis) -> str:
        """Generate stub content for a file"""
        content = f"# Type stub for {analysis.file_path}\n"
        content += "# Generated automatically by Pylance integration script\n\n"
        
        # Add imports
        for import_line in analysis.imports:
            content += f"{import_line}\n"
        
        content += "\n"
        
        # Add function stubs
        for func_line in analysis.functions:
            # Extract function name and add basic type hints
            if 'def ' in func_line:
                func_name = func_line.split('def ')[1].split('(')[0]
                content += f"def {func_name}(*args, **kwargs) -> Any: ...\n"
        
        content += "\n"
        
        # Add class stubs
        for class_line in analysis.classes:
            if 'class ' in class_line:
                class_name = class_line.split('class ')[1].split('(')[0].split(':')[0]
                content += f"class {class_name}:\n    ...\n\n"
        
        return content
    
    def run_mypy_check(self) -> Dict[str, Any]:
        """Run mypy type checking"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "mypy", 
                str(self.project_root), 
                "--ignore-missing-imports",
                "--show-error-codes"
            ], capture_output=True, text=True, check=False)
            
            return {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
        except Exception as e:
            logger.error(f"Error running mypy: {e}")
            return {"success": False, "error": str(e)}
    
    def run_flake8_check(self) -> Dict[str, Any]:
        """Run flake8 linting"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "flake8",
                str(self.project_root),
                "--max-line-length=100",
                "--ignore=E203,W503"
            ], capture_output=True, text=True, check=False)
            
            return {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
        except Exception as e:
            logger.error(f"Error running flake8: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_report(self, output_file: str = "pylance_analysis_report.json") -> str:
        """Generate a comprehensive analysis report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "total_files": len(self.analysis_results),
            "files": {},
            "summary": {
                "total_lines": 0,
                "total_typed_lines": 0,
                "average_type_coverage": 0.0,
                "total_issues": 0,
                "total_suggestions": 0
            }
        }
        
        total_lines = 0
        total_typed_lines = 0
        total_issues = 0
        total_suggestions = 0
        
        for file_path, analysis in self.analysis_results.items():
            report["files"][file_path] = analysis.__dict__
            total_lines += analysis.total_lines
            total_typed_lines += analysis.typed_lines
            total_issues += len(analysis.issues)
            total_suggestions += len(analysis.suggestions)
        
        if self.analysis_results:
            report["summary"]["total_lines"] = total_lines
            report["summary"]["total_typed_lines"] = total_typed_lines
            report["summary"]["average_type_coverage"] = total_typed_lines / total_lines * 100
            report["summary"]["total_issues"] = total_issues
            report["summary"]["total_suggestions"] = total_suggestions
        
        # Add mypy and flake8 results
        report["mypy_check"] = self.run_mypy_check()
        report["flake8_check"] = self.run_flake8_check()
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to {output_file}")
        return output_file

class VSCodeIntegration:
    """VS Code integration utilities"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.vscode_dir = self.project_root / ".vscode"
        self.vscode_dir.mkdir(exist_ok=True)
    
    def create_launch_configurations(self) -> None:
        """Create VS Code launch configurations"""
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Debug Fiber CNN Pure",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/fiber_cnn_pure.py",
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}",
                        "CUDA_VISIBLE_DEVICES": "0"
                    },
                    "args": [
                        "--data-dir", "dataset",
                        "--reference-dir", "reference",
                        "--batch-size", "4",
                        "--epochs", "5",
                        "--image-size", "256"
                    ],
                    "justMyCode": False
                },
                {
                    "name": "Debug Fiber CNN Distributed",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/fiber_cnn_distributed.py",
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}",
                        "CUDA_VISIBLE_DEVICES": "0,1"
                    },
                    "args": [
                        "--data-dir", "dataset",
                        "--reference-dir", "reference",
                        "--batch-size", "4",
                        "--epochs", "5",
                        "--image-size", "256"
                    ],
                    "justMyCode": False
                }
            ]
        }
        
        launch_file = self.vscode_dir / "launch.json"
        with open(launch_file, 'w') as f:
            json.dump(launch_config, f, indent=2)
        
        logger.info(f"Launch configurations created: {launch_file}")
    
    def create_settings(self) -> None:
        """Create VS Code settings for Pylance"""
        settings = {
            "python.defaultInterpreterPath": "./venv/bin/python",
            "python.analysis.typeCheckingMode": "basic",
            "python.analysis.autoImportCompletions": True,
            "python.analysis.autoSearchPaths": True,
            "python.analysis.diagnosticMode": "workspace",
            "python.analysis.include": [
                "**/*.py",
                "**/*.pyi"
            ],
            "python.analysis.exclude": [
                "**/__pycache__",
                "**/venv",
                "**/.git",
                "**/checkpoints",
                "**/logs",
                "**/temp"
            ],
            "python.analysis.stubPath": "./typings",
            "python.analysis.extraPaths": [
                "./",
                "./version1",
                "./version2",
                "./version3"
            ],
            "python.linting.enabled": True,
            "python.linting.pylintEnabled": False,
            "python.linting.flake8Enabled": True,
            "python.linting.mypyEnabled": True,
            "python.formatting.provider": "black",
            "python.sortImports.args": ["--profile", "black"],
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": "explicit"
            },
            "python.analysis.inlayHints.functionReturnTypes": True,
            "python.analysis.inlayHints.variableTypes": True
        }
        
        settings_file = self.vscode_dir / "settings.json"
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        
        logger.info(f"VS Code settings created: {settings_file}")

def main():
    """Main function for Pylance integration"""
    parser = argparse.ArgumentParser(description='Pylance Integration for Fiber CNN Project')
    parser.add_argument('--project-root', type=str, default='.',
                       help='Project root directory')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze project files')
    parser.add_argument('--generate-stubs', action='store_true',
                       help='Generate type stubs')
    parser.add_argument('--create-vscode-config', action='store_true',
                       help='Create VS Code configurations')
    parser.add_argument('--run-checks', action='store_true',
                       help='Run mypy and flake8 checks')
    parser.add_argument('--report', type=str, default='pylance_analysis_report.json',
                       help='Output report file')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PylanceAnalyzer(args.project_root)
    
    if args.analyze:
        logger.info("Starting project analysis...")
        analyzer.analyze_project()
        
        # Generate report
        report_file = analyzer.generate_report(args.report)
        logger.info(f"Analysis complete. Report saved to: {report_file}")
    
    if args.generate_stubs:
        logger.info("Generating type stubs...")
        analyzer.generate_type_stubs()
    
    if args.create_vscode_config:
        logger.info("Creating VS Code configurations...")
        vscode_integration = VSCodeIntegration(args.project_root)
        vscode_integration.create_launch_configurations()
        vscode_integration.create_settings()
    
    if args.run_checks:
        logger.info("Running code quality checks...")
        
        # Run mypy
        mypy_result = analyzer.run_mypy_check()
        if mypy_result["success"]:
            logger.info("✅ MyPy check passed")
        else:
            logger.warning("⚠️ MyPy check failed")
            if mypy_result.get("stderr"):
                logger.warning(mypy_result["stderr"])
        
        # Run flake8
        flake8_result = analyzer.run_flake8_check()
        if flake8_result["success"]:
            logger.info("✅ Flake8 check passed")
        else:
            logger.warning("⚠️ Flake8 check failed")
            if flake8_result.get("stdout"):
                logger.warning(flake8_result["stdout"])
    
    if not any([args.analyze, args.generate_stubs, args.create_vscode_config, args.run_checks]):
        # Default: run all
        logger.info("Running complete Pylance integration setup...")
        
        # Analyze project
        analyzer.analyze_project()
        
        # Generate stubs
        analyzer.generate_type_stubs()
        
        # Create VS Code config
        vscode_integration = VSCodeIntegration(args.project_root)
        vscode_integration.create_launch_configurations()
        vscode_integration.create_settings()
        
        # Run checks
        mypy_result = analyzer.run_mypy_check()
        flake8_result = analyzer.run_flake8_check()
        
        # Generate report
        report_file = analyzer.generate_report(args.report)
        
        logger.info("✅ Pylance integration setup complete!")
        logger.info(f"📊 Analysis report: {report_file}")
        logger.info("🔧 VS Code configurations created")
        logger.info("📝 Type stubs generated")
        
        if mypy_result["success"]:
            logger.info("✅ MyPy type checking passed")
        else:
            logger.warning("⚠️ MyPy type checking failed")
        
        if flake8_result["success"]:
            logger.info("✅ Flake8 linting passed")
        else:
            logger.warning("⚠️ Flake8 linting failed")

if __name__ == "__main__":
    main() 