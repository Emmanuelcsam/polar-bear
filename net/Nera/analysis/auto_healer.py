#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-Healing Engine for Neural Nexus IDE
Intelligent error detection and automatic code fixing using open-source tools.
"""
import re
import ast
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..core.models import ScriptAnalysis
from ..core.config import config
from ..core.logger import logger
from .code_analyzer import analyzer


class AutoHealer:
    """Enhanced auto-healing engine using multiple open-source tools."""

    def __init__(self):
        self.max_attempts = config.max_heal_attempts
        self.healing_patterns = self._load_healing_patterns()
        self.ruff_available = config.features.get('ruff', False)

    def _load_healing_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load common error patterns and their fixes."""
        return {
            # Syntax errors
            'invalid_syntax': {
                'pattern': r'invalid syntax.*\(.*line (\d+)\)',
                'fix_type': 'syntax'
            },
            'missing_colon': {
                'pattern': r'expected.*:.*\(.*line (\d+)\)',
                'fix_type': 'add_colon'
            },
            'unmatched_parentheses': {
                'pattern': r'unexpected EOF while parsing|unmatched.*\(',
                'fix_type': 'parentheses'
            },

            # Import errors
            'module_not_found': {
                'pattern': r"No module named '([^']+)'",
                'fix_type': 'install_module'
            },
            'import_error': {
                'pattern': r"cannot import name '([^']+)' from '([^']+)'",
                'fix_type': 'fix_import'
            },

            # Name errors
            'name_not_defined': {
                'pattern': r"name '([^']+)' is not defined",
                'fix_type': 'fix_name'
            },

            # Type errors
            'type_error': {
                'pattern': r"'([^']+)' object.*",
                'fix_type': 'type_fix'
            },

            # Indentation errors
            'indentation_error': {
                'pattern': r'IndentationError.*line (\d+)',
                'fix_type': 'fix_indentation'
            },

            # Common logical errors
            'division_by_zero': {
                'pattern': r'division by zero',
                'fix_type': 'add_zero_check'
            }
        }

    async def auto_heal_code(self, content: str, error_message: str,
                           analysis: Optional[ScriptAnalysis] = None) -> Optional[str]:
        """Attempt to automatically heal code based on error message and analysis."""
        if analysis is None:
            analysis = await analyzer.analyze_code(content)

        # Try different healing strategies
        healing_strategies = [
            self._heal_with_ruff,
            self._heal_syntax_errors,
            self._heal_import_errors,
            self._heal_name_errors,
            self._heal_indentation_errors,
            self._heal_common_patterns
        ]

        for strategy in healing_strategies:
            try:
                fixed_code = await strategy(content, error_message, analysis)
                if fixed_code and fixed_code != content:
                    # Verify the fix doesn't introduce new errors
                    if await self._verify_fix(fixed_code):
                        logger.info(f"Successfully healed code using {strategy.__name__}")
                        return fixed_code
            except Exception as e:
                logger.debug(f"Healing strategy {strategy.__name__} failed: {e}")

        return None

    async def _heal_with_ruff(self, content: str, error_message: str,
                            analysis: ScriptAnalysis) -> Optional[str]:
        """Use Ruff's auto-fix capabilities."""
        if not self.ruff_available:
            return None

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name

            try:
                # Run ruff with auto-fix
                result = subprocess.run([
                    'ruff', 'check', '--fix', '--unsafe-fixes', temp_file
                ], capture_output=True, text=True, timeout=30)

                # Read the fixed content
                fixed_content = Path(temp_file).read_text()

                if fixed_content != content:
                    return fixed_content

            finally:
                Path(temp_file).unlink(missing_ok=True)

        except Exception as e:
            logger.debug(f"Ruff auto-fix failed: {e}")

        return None

    async def _heal_syntax_errors(self, content: str, error_message: str,
                                analysis: ScriptAnalysis) -> Optional[str]:
        """Fix common syntax errors."""
        lines = content.splitlines()

        # Fix missing colons
        if 'expected' in error_message and ':' in error_message:
            line_match = re.search(r'line (\d+)', error_message)
            if line_match:
                line_num = int(line_match.group(1)) - 1
                if 0 <= line_num < len(lines):
                    line = lines[line_num]
                    # Check for common cases where colon is missing
                    if re.match(r'^\s*(if|elif|else|for|while|def|class|try|except|finally|with)\b', line):
                        if not line.rstrip().endswith(':'):
                            lines[line_num] = line.rstrip() + ':'
                            return '\n'.join(lines)

        # Fix unmatched parentheses
        if 'unmatched' in error_message or 'unexpected EOF' in error_message:
            # Count parentheses, brackets, and braces
            open_chars = {'(': ')', '[': ']', '{': '}'}
            stack = []

            for i, line in enumerate(lines):
                for char in line:
                    if char in open_chars:
                        stack.append((char, i))
                    elif char in open_chars.values():
                        if stack and open_chars[stack[-1][0]] == char:
                            stack.pop()

            # Add missing closing characters
            if stack:
                missing_chars = [open_chars[char] for char, _ in reversed(stack)]
                if lines:
                    lines[-1] += ''.join(missing_chars)
                    return '\n'.join(lines)

        return None

    async def _heal_import_errors(self, content: str, error_message: str,
                                analysis: ScriptAnalysis) -> Optional[str]:
        """Fix import-related errors."""
        lines = content.splitlines()

        # Handle module not found errors
        module_match = re.search(r"No module named '([^']+)'", error_message)
        if module_match:
            module_name = module_match.group(1)

            # Suggest common alternatives
            common_alternatives = {
                'cv2': 'opencv-python',
                'PIL': 'Pillow',
                'sklearn': 'scikit-learn',
                'yaml': 'PyYAML',
                'bs4': 'beautifulsoup4'
            }

            if module_name in common_alternatives:
                # Add a comment suggesting the correct package
                for i, line in enumerate(lines):
                    if f'import {module_name}' in line or f'from {module_name}' in line:
                        comment = f"# Install with: pip install {common_alternatives[module_name]}"
                        lines.insert(i, comment)
                        return '\n'.join(lines)

        # Handle import name errors
        import_match = re.search(r"cannot import name '([^']+)' from '([^']+)'", error_message)
        if import_match:
            name, module = import_match.groups()

            # Try to suggest alternative imports
            alternatives = {
                ('json', 'JSONDecodeError'): 'from json.decoder import JSONDecodeError',
                ('requests', 'get'): 'import requests  # Use requests.get()',
                ('os', 'path'): 'import os.path',
            }

            alt_import = alternatives.get((module, name))
            if alt_import:
                for i, line in enumerate(lines):
                    if f'from {module} import' in line and name in line:
                        lines[i] = alt_import
                        return '\n'.join(lines)

        return None

    async def _heal_name_errors(self, content: str, error_message: str,
                              analysis: ScriptAnalysis) -> Optional[str]:
        """Fix name-related errors."""
        lines = content.splitlines()

        name_match = re.search(r"name '([^']+)' is not defined", error_message)
        if name_match:
            undefined_name = name_match.group(1)

            # Common misspellings and their corrections
            common_fixes = {
                'lenght': 'length',
                'lengh': 'length',
                'pirnt': 'print',
                'prin': 'print',
                'improt': 'import',
                'slef': 'self',
                'True': 'True',  # Ensure boolean literals are capitalized
                'False': 'False',
                'None': 'None'
            }

            if undefined_name in common_fixes:
                corrected_name = common_fixes[undefined_name]
                for i, line in enumerate(lines):
                    if undefined_name in line:
                        # Use word boundaries to avoid partial replacements
                        pattern = r'\b' + re.escape(undefined_name) + r'\b'
                        lines[i] = re.sub(pattern, corrected_name, line)
                return '\n'.join(lines)

            # Check if it's a variable that should be defined
            if undefined_name.islower() and len(undefined_name) > 2:
                # Find where it's first used and suggest initialization
                for i, line in enumerate(lines):
                    if undefined_name in line and '=' in line:
                        # Suggest initialization
                        if i > 0:
                            indent = len(line) - len(line.lstrip())
                            init_line = ' ' * indent + f"{undefined_name} = None  # Initialize variable"
                            lines.insert(i, init_line)
                            return '\n'.join(lines)

        return None

    async def _heal_indentation_errors(self, content: str, error_message: str,
                                     analysis: ScriptAnalysis) -> Optional[str]:
        """Fix indentation errors."""
        lines = content.splitlines()

        line_match = re.search(r'line (\d+)', error_message)
        if line_match:
            line_num = int(line_match.group(1)) - 1

            if 0 <= line_num < len(lines):
                # Analyze indentation pattern
                indent_levels = []
                for line in lines:
                    if line.strip():  # Skip empty lines
                        indent = len(line) - len(line.lstrip())
                        indent_levels.append(indent)

                if indent_levels:
                    # Find the most common indentation (probably 4 spaces)
                    common_indent = max(set(indent_levels), key=indent_levels.count)

                    # Fix the problematic line
                    problem_line = lines[line_num]
                    if problem_line.strip():
                        # Determine expected indentation based on previous lines
                        expected_indent = 0
                        if line_num > 0:
                            prev_line = lines[line_num - 1].rstrip()
                            if prev_line.endswith(':'):
                                expected_indent = common_indent
                            else:
                                prev_indent = len(lines[line_num - 1]) - len(lines[line_num - 1].lstrip())
                                expected_indent = prev_indent

                        # Apply the fix
                        content_without_indent = problem_line.lstrip()
                        lines[line_num] = ' ' * expected_indent + content_without_indent
                        return '\n'.join(lines)

        return None

    async def _heal_common_patterns(self, content: str, error_message: str,
                                  analysis: ScriptAnalysis) -> Optional[str]:
        """Fix other common error patterns."""
        lines = content.splitlines()

        # Division by zero protection
        if 'division by zero' in error_message:
            for i, line in enumerate(lines):
                if '/' in line and 'if' not in line:
                    # Find division operations and add zero checks
                    parts = line.split('/')
                    if len(parts) == 2:
                        left, right = parts
                        right_var = right.strip().split()[0]

                        # Add zero check
                        indent = len(line) - len(line.lstrip())
                        check_line = ' ' * indent + f"if {right_var} != 0:"
                        protected_line = ' ' * (indent + 4) + line.strip()
                        else_line = ' ' * (indent + 4) + "result = 0  # Handle division by zero"

                        lines[i:i+1] = [check_line, protected_line, ' ' * indent + 'else:', else_line]
                        return '\n'.join(lines)

        return None

    async def _verify_fix(self, fixed_code: str) -> bool:
        """Verify that the fix doesn't introduce new syntax errors."""
        try:
            ast.parse(fixed_code)
            return True
        except SyntaxError:
            return False

    def generate_fix_suggestions(self, content: str, error_message: str,
                               analysis: ScriptAnalysis) -> List[Dict[str, Any]]:
        """Generate human-readable fix suggestions."""
        suggestions = []

        # Pattern-based suggestions
        for pattern_name, pattern_info in self.healing_patterns.items():
            if re.search(pattern_info['pattern'], error_message, re.IGNORECASE):
                suggestion = self._get_suggestion_for_pattern(pattern_name, error_message)
                if suggestion:
                    suggestions.append(suggestion)

        # Analysis-based suggestions
        if analysis.security_issues:
            suggestions.append({
                'type': 'security',
                'title': 'Security Issues Detected',
                'description': f'Found {len(analysis.security_issues)} security issues',
                'priority': 'high',
                'auto_fixable': False
            })

        if analysis.ruff_fixes_available > 0:
            suggestions.append({
                'type': 'formatting',
                'title': 'Auto-Formatting Available',
                'description': f'{analysis.ruff_fixes_available} issues can be auto-fixed',
                'priority': 'medium',
                'auto_fixable': True
            })

        return suggestions

    def _get_suggestion_for_pattern(self, pattern_name: str, error_message: str) -> Optional[Dict[str, Any]]:
        """Get suggestion for a specific error pattern."""
        suggestions_map = {
            'module_not_found': {
                'type': 'import',
                'title': 'Missing Module',
                'description': 'Install the required module using pip',
                'priority': 'high',
                'auto_fixable': False
            },
            'name_not_defined': {
                'type': 'variable',
                'title': 'Undefined Variable',
                'description': 'Variable used before definition',
                'priority': 'high',
                'auto_fixable': True
            },
            'invalid_syntax': {
                'type': 'syntax',
                'title': 'Syntax Error',
                'description': 'Check for missing colons, parentheses, or quotes',
                'priority': 'critical',
                'auto_fixable': True
            },
            'indentation_error': {
                'type': 'indentation',
                'title': 'Indentation Error',
                'description': 'Fix indentation to match Python requirements',
                'priority': 'high',
                'auto_fixable': True
            }
        }

        return suggestions_map.get(pattern_name)


# Global auto-healer instance
auto_healer = AutoHealer()
