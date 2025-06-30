#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Models for Neural Nexus IDE Server
Enhanced models with validation and serialization support.
"""
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Union
from pathlib import Path

try:
    import asyncio
    HAS_ASYNCIO = True
except ImportError:
    HAS_ASYNCIO = False


@dataclass
class ScriptProcess:
    """Represents a running script process with enhanced monitoring."""
    process: Optional[Any]  # asyncio.subprocess.Process when available
    script_id: str
    start_time: float
    output_lines: List[str] = field(default_factory=list)
    error_lines: List[str] = field(default_factory=list)
    script_name: str = ""
    project_id: Optional[str] = None
    status: str = "running"  # running, completed, failed, terminated
    exit_code: Optional[int] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


@dataclass
class SecurityFinding:
    """Represents a security issue found during analysis."""
    tool: str  # bandit, semgrep, etc.
    severity: str  # low, medium, high, critical
    rule_id: str
    message: str
    line: int
    column: Optional[int] = None
    fix_suggestion: Optional[str] = None
    confidence: str = "medium"


@dataclass
class CodeQualityIssue:
    """Represents a code quality issue."""
    tool: str  # pylint, ruff, mypy, etc.
    type: str  # error, warning, info, style
    code: str  # error code (e.g., E302, W291)
    message: str
    line: int
    column: Optional[int] = None
    fix_suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ScriptAnalysis:
    """Enhanced results of script analysis with comprehensive insights."""
    # Basic analysis
    errors: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)

    # Enhanced analysis
    security_issues: List[SecurityFinding] = field(default_factory=list)
    quality_issues: List[CodeQualityIssue] = field(default_factory=list)
    type_errors: List[Dict[str, Any]] = field(default_factory=list)
    performance_hints: List[Dict[str, Any]] = field(default_factory=list)

    # Scores and metrics
    code_quality_score: Optional[float] = None
    security_score: Optional[float] = None
    maintainability_score: Optional[float] = None
    complexity_score: Optional[float] = None

    # Tool-specific results
    pylint_score: Optional[float] = None
    mypy_errors: int = 0
    ruff_fixes_available: int = 0

    # Analysis metadata
    analysis_time: float = field(default_factory=time.time)
    tools_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)

        # Convert SecurityFinding and CodeQualityIssue objects
        result['security_issues'] = [asdict(issue) for issue in self.security_issues]
        result['quality_issues'] = [asdict(issue) for issue in self.quality_issues]

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis results."""
        return {
            'total_errors': len(self.errors) + len(self.quality_issues),
            'security_issues': len(self.security_issues),
            'critical_security_issues': len([i for i in self.security_issues if i.severity == 'critical']),
            'warnings': len(self.warnings),
            'suggestions': len(self.suggestions),
            'code_quality_score': self.code_quality_score,
            'security_score': self.security_score,
            'auto_fixable_issues': len([i for i in self.quality_issues if i.auto_fixable]),
            'tools_used': self.tools_used
        }


@dataclass
class Project:
    """Represents a project containing multiple scripts with enhanced metadata."""
    id: str
    name: str
    scripts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    main_script: Optional[str] = None
    created: float = field(default_factory=time.time)
    modified: float = field(default_factory=time.time)
    dependencies: Set[str] = field(default_factory=set)
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    license: str = "MIT"

    # Project metadata
    python_version: Optional[str] = None
    requirements_file: Optional[str] = None
    readme_file: Optional[str] = None

    # Analysis results for project
    last_analysis: Optional[ScriptAnalysis] = None
    project_health_score: Optional[float] = None

    def get_script_count(self) -> int:
        """Get the number of scripts in the project."""
        return len(self.scripts)

    def get_total_lines(self) -> int:
        """Get total lines of code across all scripts."""
        total = 0
        for script_info in self.scripts.values():
            content = script_info.get('content', '')
            total += len(content.splitlines())
        return total

    def update_modified_time(self):
        """Update the modified timestamp."""
        self.modified = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['dependencies'] = list(self.dependencies)  # Convert set to list
        if self.last_analysis:
            result['last_analysis'] = self.last_analysis.to_dict()
        return result


@dataclass
class TerminalSession:
    """Represents a terminal session with history and state."""
    session_id: str
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    history: List[str] = field(default_factory=list)
    working_directory: str = "."
    environment_vars: Dict[str, str] = field(default_factory=dict)
    active: bool = True

    def add_command(self, command: str):
        """Add a command to the history."""
        self.history.append(command)
        self.last_used = time.time()

        # Keep only last 1000 commands
        if len(self.history) > 1000:
            self.history = self.history[-1000:]


@dataclass
class PerformanceMetrics:
    """Server performance metrics tracking."""
    start_time: float = field(default_factory=time.time)
    requests_handled: int = 0
    scripts_executed: int = 0
    analysis_runs: int = 0
    errors_encountered: int = 0
    auto_heal_attempts: int = 0
    successful_heals: int = 0

    # Timing metrics
    avg_analysis_time: float = 0.0
    avg_execution_time: float = 0.0
    avg_response_time: float = 0.0

    # Resource usage
    peak_memory_usage: float = 0.0
    current_memory_usage: float = 0.0
    cpu_usage_percent: float = 0.0

    def update_request_metrics(self, response_time: float):
        """Update request handling metrics."""
        self.requests_handled += 1
        self.avg_response_time = (
            (self.avg_response_time * (self.requests_handled - 1) + response_time)
            / self.requests_handled
        )

    def update_analysis_metrics(self, analysis_time: float):
        """Update analysis timing metrics."""
        self.analysis_runs += 1
        self.avg_analysis_time = (
            (self.avg_analysis_time * (self.analysis_runs - 1) + analysis_time)
            / self.analysis_runs
        )

    def update_execution_metrics(self, execution_time: float):
        """Update script execution metrics."""
        self.scripts_executed += 1
        self.avg_execution_time = (
            (self.avg_execution_time * (self.scripts_executed - 1) + execution_time)
            / self.scripts_executed
        )

    def get_uptime(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'uptime': self.get_uptime(),
            'requests_handled': self.requests_handled,
            'scripts_executed': self.scripts_executed,
            'analysis_runs': self.analysis_runs,
            'errors_encountered': self.errors_encountered,
            'auto_heal_attempts': self.auto_heal_attempts,
            'successful_heals': self.successful_heals,
            'success_rate': (
                self.successful_heals / max(self.auto_heal_attempts, 1) * 100
                if self.auto_heal_attempts > 0 else 0
            ),
            'avg_analysis_time': self.avg_analysis_time,
            'avg_execution_time': self.avg_execution_time,
            'avg_response_time': self.avg_response_time,
            'memory_usage_mb': self.current_memory_usage,
            'peak_memory_mb': self.peak_memory_usage,
            'cpu_usage_percent': self.cpu_usage_percent
        }


def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate a unique ID for scripts, projects, etc."""
    timestamp = str(int(time.time() * 1000))
    hash_input = f"{prefix}_{timestamp}_{time.time()}"
    hash_obj = hashlib.md5(hash_input.encode())
    return f"{prefix}_{hash_obj.hexdigest()[:length]}" if prefix else hash_obj.hexdigest()[:length]
