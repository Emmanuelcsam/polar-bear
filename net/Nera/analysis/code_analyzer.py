#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Analysis Engine
Comprehensive static analysis using multiple open-source tools.
"""
import ast
import re
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from ..core.models import ScriptAnalysis, SecurityFinding, CodeQualityIssue
from ..core.config import config, CACHE_DIR
from ..core.logger import logger


class CodeAnalyzer:
    """Enhanced code analyzer using multiple open-source tools."""

    def __init__(self):
        self.cache_dir = CACHE_DIR / "analysis"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = config.analysis_cache_ttl
        self._tool_cache = {}

    async def analyze_code(self, content: str, use_cache: bool = True) -> ScriptAnalysis:
        """Perform comprehensive code analysis."""
        start_time = time.time()

        # Check cache first
        if use_cache:
            cached_result = self._get_cached_analysis(content)
            if cached_result:
                return cached_result

        analysis = ScriptAnalysis()
        analysis.tools_used = []

        # Basic AST analysis
        try:
            tree = ast.parse(content)
            await self._analyze_ast(tree, content, analysis)
        except SyntaxError as e:
            analysis.errors.append({
                'type': 'syntax_error',
                'message': str(e),
                'line': e.lineno or 1,
                'column': e.offset or 0
            })

        # Run available analysis tools
        await self._run_pylint_analysis(content, analysis)
        await self._run_mypy_analysis(content, analysis)
        await self._run_ruff_analysis(content, analysis)
        await self._run_bandit_analysis(content, analysis)
        await self._run_semgrep_analysis(content, analysis)
        await self._run_friendly_traceback_analysis(content, analysis)

        # Calculate scores
        self._calculate_scores(analysis)

        # Cache the result
        if use_cache:
            self._cache_analysis(content, analysis)

        analysis.analysis_time = time.time() - start_time
        logger.debug(f"Code analysis completed in {analysis.analysis_time:.2f}s")

        return analysis

    async def _analyze_ast(self, tree: ast.AST, content: str, analysis: ScriptAnalysis):
        """Perform AST-based analysis."""
        visitor = CodeAnalysisVisitor()
        visitor.visit(tree)

        # Extract imports
        analysis.imports = visitor.imports

        # Extract dependencies
        for imp in visitor.imports:
            if '.' in imp:
                base_module = imp.split('.')[0]
            else:
                base_module = imp

            if base_module not in sys.stdlib_module_names:
                analysis.dependencies.append({
                    'name': base_module,
                    'type': 'import',
                    'line': visitor.import_lines.get(imp, 1)
                })

        # Detect potential issues
        if visitor.has_eval:
            analysis.security_issues.append(SecurityFinding(
                tool='ast_analyzer',
                severity='high',
                rule_id='eval_usage',
                message='Use of eval() detected - potential security risk',
                line=visitor.eval_lines[0] if visitor.eval_lines else 1,
                fix_suggestion='Consider using ast.literal_eval() or safer alternatives'
            ))

        if visitor.has_exec:
            analysis.security_issues.append(SecurityFinding(
                tool='ast_analyzer',
                severity='high',
                rule_id='exec_usage',
                message='Use of exec() detected - potential security risk',
                line=visitor.exec_lines[0] if visitor.exec_lines else 1,
                fix_suggestion='Consider refactoring to avoid dynamic code execution'
            ))

        # Performance hints
        if visitor.nested_loops > 2:
            analysis.performance_hints.append({
                'type': 'complexity',
                'message': f'Deep nested loops detected ({visitor.nested_loops} levels)',
                'suggestion': 'Consider refactoring to reduce complexity',
                'line': 1
            })

        analysis.complexity_score = min(visitor.complexity / 10, 10.0)

    async def _run_pylint_analysis(self, content: str, analysis: ScriptAnalysis):
        """Run Pylint analysis if available."""
        if not config.features.get('pylint', False):
            return

        try:
            from pylint.lint import Run
            from pylint.reporters.text import TextReporter
            import io

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name

            try:
                output = io.StringIO()
                reporter = TextReporter(output)

                # Run pylint with minimal configuration
                Run([
                    '--disable=all',
                    '--enable=E,W,C0103,C0114,C0115,C0116',  # Enable key checks
                    '--output-format=parseable',
                    temp_file
                ], reporter=reporter, exit=False)

                # Parse output
                for line in output.getvalue().splitlines():
                    if ':' in line and temp_file in line:
                        parts = line.split(':')
                        if len(parts) >= 4:
                            try:
                                line_num = int(parts[1])
                                col_num = int(parts[2]) if parts[2].isdigit() else 0
                                message = ':'.join(parts[3:]).strip()

                                if message.startswith('E'):
                                    issue_type = 'error'
                                elif message.startswith('W'):
                                    issue_type = 'warning'
                                else:
                                    issue_type = 'info'

                                analysis.quality_issues.append(CodeQualityIssue(
                                    tool='pylint',
                                    type=issue_type,
                                    code=message.split()[0] if message else 'unknown',
                                    message=message,
                                    line=line_num,
                                    column=col_num
                                ))
                            except (ValueError, IndexError):
                                continue

                analysis.tools_used.append('pylint')

            finally:
                Path(temp_file).unlink(missing_ok=True)

        except ImportError:
            logger.debug("Pylint not available")
        except Exception as e:
            logger.error(f"Pylint analysis failed: {e}")

    async def _run_mypy_analysis(self, content: str, analysis: ScriptAnalysis):
        """Run MyPy type checking if available."""
        if not config.features.get('mypy', False):
            return

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name

            try:
                result = subprocess.run([
                    sys.executable, '-m', 'mypy',
                    '--no-error-summary',
                    '--show-column-numbers',
                    '--ignore-missing-imports',
                    temp_file
                ], capture_output=True, text=True, timeout=30)

                for line in result.stdout.splitlines():
                    if ':' in line and temp_file in line:
                        parts = line.split(':')
                        if len(parts) >= 4:
                            try:
                                line_num = int(parts[1])
                                col_num = int(parts[2]) if parts[2].isdigit() else 0
                                message = ':'.join(parts[3:]).strip()

                                analysis.type_errors.append({
                                    'line': line_num,
                                    'column': col_num,
                                    'message': message,
                                    'tool': 'mypy'
                                })
                                analysis.mypy_errors += 1
                            except (ValueError, IndexError):
                                continue

                analysis.tools_used.append('mypy')

            finally:
                Path(temp_file).unlink(missing_ok=True)

        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("MyPy not available or timed out")
        except Exception as e:
            logger.error(f"MyPy analysis failed: {e}")

    async def _run_ruff_analysis(self, content: str, analysis: ScriptAnalysis):
        """Run Ruff linting and formatting if available."""
        if not config.features.get('ruff', False):
            return

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name

            try:
                # Check for issues
                result = subprocess.run([
                    'ruff', 'check', '--output-format=json', temp_file
                ], capture_output=True, text=True, timeout=30)

                if result.stdout:
                    import json
                    issues = json.loads(result.stdout)
                    for issue in issues:
                        analysis.quality_issues.append(CodeQualityIssue(
                            tool='ruff',
                            type='warning' if issue.get('severity') == 'warning' else 'error',
                            code=issue.get('code', 'unknown'),
                            message=issue.get('message', ''),
                            line=issue.get('location', {}).get('row', 1),
                            column=issue.get('location', {}).get('column', 0),
                            auto_fixable=issue.get('fix') is not None
                        ))

                        if issue.get('fix'):
                            analysis.ruff_fixes_available += 1

                analysis.tools_used.append('ruff')

            finally:
                Path(temp_file).unlink(missing_ok=True)

        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("Ruff not available or timed out")
        except Exception as e:
            logger.error(f"Ruff analysis failed: {e}")

    async def _run_bandit_analysis(self, content: str, analysis: ScriptAnalysis):
        """Run Bandit security analysis if available."""
        if not config.features.get('bandit', False):
            return

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name

            try:
                result = subprocess.run([
                    'bandit', '-f', 'json', '-q', temp_file
                ], capture_output=True, text=True, timeout=30)

                if result.stdout:
                    import json
                    bandit_result = json.loads(result.stdout)

                    for issue in bandit_result.get('results', []):
                        analysis.security_issues.append(SecurityFinding(
                            tool='bandit',
                            severity=issue.get('issue_severity', 'medium').lower(),
                            rule_id=issue.get('test_id', 'unknown'),
                            message=issue.get('issue_text', ''),
                            line=issue.get('line_number', 1),
                            confidence=issue.get('issue_confidence', 'medium').lower()
                        ))

                analysis.tools_used.append('bandit')

            finally:
                Path(temp_file).unlink(missing_ok=True)

        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("Bandit not available or timed out")
        except Exception as e:
            logger.error(f"Bandit analysis failed: {e}")

    async def _run_semgrep_analysis(self, content: str, analysis: ScriptAnalysis):
        """Run Semgrep security analysis if available."""
        if not config.features.get('semgrep', False):
            return

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name

            try:
                result = subprocess.run([
                    'semgrep', '--json', '--config=auto', temp_file
                ], capture_output=True, text=True, timeout=30)

                if result.stdout:
                    import json
                    semgrep_result = json.loads(result.stdout)

                    for finding in semgrep_result.get('results', []):
                        analysis.security_issues.append(SecurityFinding(
                            tool='semgrep',
                            severity=finding.get('extra', {}).get('severity', 'medium').lower(),
                            rule_id=finding.get('check_id', 'unknown'),
                            message=finding.get('extra', {}).get('message', ''),
                            line=finding.get('start', {}).get('line', 1),
                            column=finding.get('start', {}).get('col', 0),
                            fix_suggestion=finding.get('extra', {}).get('fix')
                        ))

                analysis.tools_used.append('semgrep')

            finally:
                Path(temp_file).unlink(missing_ok=True)

        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("Semgrep not available or timed out")
        except Exception as e:
            logger.error(f"Semgrep analysis failed: {e}")

    async def _run_friendly_traceback_analysis(self, content: str, analysis: ScriptAnalysis):
        """Use friendly-traceback for enhanced error interpretation."""
        if not config.features.get('friendly_traceback', False):
            return

        try:
            import friendly_traceback

            # Try to execute the code in a safe environment to catch runtime errors
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(content)
                    temp_file = f.name

                # Install friendly as the exception handler
                friendly_traceback.install()

                # This is a basic check - in a real implementation,
                # you might want to run this in a sandboxed environment
                result = subprocess.run([
                    sys.executable, '-c',
                    f'''
import friendly_traceback
friendly_traceback.install()
exec(open("{temp_file}").read())
'''
                ], capture_output=True, text=True, timeout=10)

                if result.stderr and 'Friendly' in result.stderr:
                    # Parse friendly traceback output for enhanced error messages
                    analysis.suggestions.append({
                        'type': 'friendly_error',
                        'message': 'Enhanced error analysis available',
                        'details': result.stderr[:500],  # Limit length
                        'tool': 'friendly_traceback'
                    })

                Path(temp_file).unlink(missing_ok=True)
                analysis.tools_used.append('friendly_traceback')

            except subprocess.TimeoutExpired:
                logger.debug("Friendly traceback analysis timed out")

        except ImportError:
            logger.debug("Friendly traceback not available")
        except Exception as e:
            logger.error(f"Friendly traceback analysis failed: {e}")

    def _calculate_scores(self, analysis: ScriptAnalysis):
        """Calculate various quality scores."""
        # Code quality score (0-10)
        error_penalty = len(analysis.errors) * 2
        warning_penalty = len(analysis.warnings) * 1
        quality_issue_penalty = len(analysis.quality_issues) * 1.5

        quality_score = max(0, 10 - error_penalty - warning_penalty - quality_issue_penalty)
        analysis.code_quality_score = quality_score

        # Security score (0-10)
        critical_security = len([i for i in analysis.security_issues if i.severity == 'critical'])
        high_security = len([i for i in analysis.security_issues if i.severity == 'high'])
        medium_security = len([i for i in analysis.security_issues if i.severity == 'medium'])

        security_penalty = critical_security * 5 + high_security * 3 + medium_security * 1
        security_score = max(0, 10 - security_penalty)
        analysis.security_score = security_score

        # Maintainability score based on complexity and quality
        maintainability = (quality_score + (10 - (analysis.complexity_score or 0))) / 2
        analysis.maintainability_score = maintainability

    def _get_cached_analysis(self, content: str) -> Optional[ScriptAnalysis]:
        """Get cached analysis result if available and fresh."""
        import hashlib
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_file = self.cache_dir / f"{content_hash}.json"

        if cache_file.exists():
            try:
                import json
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                # Check if cache is still fresh
                if time.time() - data.get('cached_at', 0) < self.cache_ttl:
                    # Reconstruct ScriptAnalysis object
                    analysis = ScriptAnalysis()
                    for key, value in data.items():
                        if key != 'cached_at' and hasattr(analysis, key):
                            setattr(analysis, key, value)
                    return analysis
            except Exception as e:
                logger.debug(f"Failed to load cached analysis: {e}")

        return None

    def _cache_analysis(self, content: str, analysis: ScriptAnalysis):
        """Cache analysis result."""
        try:
            import hashlib
            import json

            content_hash = hashlib.md5(content.encode()).hexdigest()
            cache_file = self.cache_dir / f"{content_hash}.json"

            data = analysis.to_dict()
            data['cached_at'] = time.time()

            with open(cache_file, 'w') as f:
                json.dump(data, f)

        except Exception as e:
            logger.debug(f"Failed to cache analysis: {e}")


class CodeAnalysisVisitor(ast.NodeVisitor):
    """AST visitor for code analysis."""

    def __init__(self):
        self.imports = []
        self.import_lines = {}
        self.has_eval = False
        self.has_exec = False
        self.eval_lines = []
        self.exec_lines = []
        self.nested_loops = 0
        self.current_loop_depth = 0
        self.complexity = 1

    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(alias.name)
            self.import_lines[alias.name] = node.lineno
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Visit from...import statements."""
        if node.module:
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                self.imports.append(full_name)
                self.import_lines[full_name] = node.lineno
        self.generic_visit(node)

    def visit_Call(self, node):
        """Visit function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id == 'eval':
                self.has_eval = True
                self.eval_lines.append(node.lineno)
            elif node.func.id == 'exec':
                self.has_exec = True
                self.exec_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_For(self, node):
        """Visit for loops."""
        self.current_loop_depth += 1
        self.nested_loops = max(self.nested_loops, self.current_loop_depth)
        self.complexity += 1
        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_While(self, node):
        """Visit while loops."""
        self.current_loop_depth += 1
        self.nested_loops = max(self.nested_loops, self.current_loop_depth)
        self.complexity += 1
        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_If(self, node):
        """Visit if statements."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Visit class definitions."""
        self.complexity += 1
        self.generic_visit(node)


# Global analyzer instance
analyzer = CodeAnalyzer()
