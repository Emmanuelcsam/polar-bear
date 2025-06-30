#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Nexus IDE Server v6.0
Enhanced backend with performance optimization, static analysis,
security scanning, and comprehensive multi-script project management.
"""

import asyncio
import os
import sys
import subprocess
import tempfile
import ast
import re
import shutil
import platform
import time
import importlib
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime
import hashlib

# Performance optimizations - set event loop policy early
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    HAS_UVLOOP = True
except ImportError:
    HAS_UVLOOP = False

# Enhanced JSON handling
try:
    import orjson
    def json_dumps(obj) -> str:
        return orjson.dumps(obj).decode()
    def json_loads(data: str) -> Any:
        return orjson.loads(data)
    HAS_ORJSON = True
except ImportError:
    import json
    json_dumps = json.dumps
    json_loads = json.loads
    HAS_ORJSON = False

# Enhanced logging
try:
    from loguru import logger
    HAS_LOGURU = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    HAS_LOGURU = False

# Core FastAPI and server dependencies
try:
    import uvicorn
except ImportError:
    uvicorn = None

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = WebSocket = WebSocketDisconnect = Request = HTTPException = None
    CORSMiddleware = StaticFiles = FileResponse = HTMLResponse = JSONResponse = None
    FASTAPI_AVAILABLE = False

# Rate limiting and security
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False

try:
    import aiofiles
except ImportError:
    aiofiles = None

try:
    import psutil
except ImportError:
    psutil = None

# Static analysis and security tools
try:
    import semgrep
    HAS_SEMGREP = True
except ImportError:
    HAS_SEMGREP = False

try:
    import bandit
    from bandit.core import manager
    HAS_BANDIT = True
except ImportError:
    HAS_BANDIT = False

# Ruff for formatting and linting
try:
    import ruff
    HAS_RUFF = True
except ImportError:
    HAS_RUFF = False

# Enhanced data serialization
try:
    import msgspec
    HAS_MSGSPEC = True
except ImportError:
    HAS_MSGSPEC = False

# Optional imports
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Configuration
HOME_DIR = Path.home() / ".neural_nexus_server"
SCRIPTS_DIR = HOME_DIR / "scripts"
LOGS_DIR = HOME_DIR / "logs"
TEMP_DIR = HOME_DIR / "temp"
PROJECTS_DIR = HOME_DIR / "projects"

# Create directories
for dir_path in [SCRIPTS_DIR, LOGS_DIR, TEMP_DIR, PROJECTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Enhanced logging setup with performance optimizations
if HAS_LOGURU:
    # Remove default logger
    logger.remove()

    # Add structured JSON logging for production
    logger.add(
        LOGS_DIR / f"server_{datetime.now():%Y%m%d_%H%M%S}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="10 MB",
        compression="zip",
        serialize=True,  # JSON format for easier parsing
        retention="30 days"
    )

    # Console output with colors
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
else:
    # Fallback to standard logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / f"server_{datetime.now():%Y%m%d_%H%M%S}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

@dataclass
class ScriptProcess:
    """Represents a running script process"""
    process: asyncio.subprocess.Process
    script_id: str
    start_time: float
    output_lines: List[str] = field(default_factory=list)
    error_lines: List[str] = field(default_factory=list)
    script_name: str = ""
    project_id: Optional[str] = None

@dataclass
class ScriptAnalysis:
    """Enhanced results of script analysis with security and performance insights"""
    errors: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)

    # New security and analysis fields
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    type_errors: List[Dict[str, Any]] = field(default_factory=list)
    performance_hints: List[Dict[str, Any]] = field(default_factory=list)
    code_quality_score: Optional[float] = None
    semgrep_findings: List[Dict[str, Any]] = field(default_factory=list)
    bandit_findings: List[Dict[str, Any]] = field(default_factory=list)
    ruff_suggestions: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Project:
    """Represents a project containing multiple scripts"""
    id: str
    name: str
    scripts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    main_script: Optional[str] = None
    created: float = field(default_factory=time.time)
    dependencies: Set[str] = field(default_factory=set)

class NeuralNexusServer:
    """Enhanced main server class with security, performance, and static analysis"""

    def __init__(self, port: int = 8765):
        if FastAPI is None:
            raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn websockets")

        self.port = port

        # Enhanced FastAPI app with custom JSON encoder
        if HAS_ORJSON:
            self.app = FastAPI(
                title="Neural Nexus IDE Server",
                version="6.0",
                description="AI-Powered Development Environment with Security & Performance",
                default_response_class=JSONResponse
            )
        else:
            self.app = FastAPI(
                title="Neural Nexus IDE Server",
                version="6.0",
                description="AI-Powered Development Environment"
            )

        # Rate limiting setup
        if HAS_SLOWAPI:
            self.limiter = Limiter(key_func=get_remote_address)
            self.app.state.limiter = self.limiter
            self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        else:
            self.limiter = None

        # Core application state
        self.clients: Set[Any] = set()
        self.running_scripts: Dict[str, ScriptProcess] = {}
        self.script_cache: Dict[str, Dict[str, Any]] = {}
        self.installed_packages: Set[str] = set()
        self.auto_heal_mode: Dict[str, bool] = {}
        self.heal_attempts: Dict[str, int] = {}
        self.projects: Dict[str, Project] = {}
        self.active_project: Optional[str] = None
        self._cached_api_key: Optional[str] = None
        self.max_heal_attempts: int = 10
        self.terminal_history: Dict[str, List[str]] = {}

        # Security and analysis features
        self.security_enabled: bool = True
        self.analysis_cache: Dict[str, ScriptAnalysis] = {}
        self.last_security_scan: Dict[str, float] = {}

        # Performance monitoring
        self.performance_stats = {
            'requests_handled': 0,
            'scripts_executed': 0,
            'analysis_runs': 0,
            'start_time': time.time()
        }

        # Setup routes with enhanced security
        self.setup_routes()

        # Initialize package list
        asyncio.create_task(self.update_installed_packages())

        # Add temp directory to Python path for imports
        if str(TEMP_DIR) not in sys.path:
            sys.path.insert(0, str(TEMP_DIR))
        if str(PROJECTS_DIR) not in sys.path:
            sys.path.insert(0, str(PROJECTS_DIR))

    def setup_routes(self):
        """Setup FastAPI routes with enhanced security, rate limiting, and performance"""

        # Enhanced CORS with security headers
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        # Security headers middleware
        @self.app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            response = await call_next(request)

            # Security headers to prevent XSS, clickjacking, etc.
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

            # Content Security Policy (CSP)
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' cdnjs.cloudflare.com; "
                "style-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; "
                "img-src 'self' data: blob:; "
                "connect-src 'self' ws: wss:; "
                "font-src 'self' cdnjs.cloudflare.com"
            )
            response.headers["Content-Security-Policy"] = csp

            # COOP/COEP headers for enhanced security
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
            response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"

            return response

        # Performance monitoring middleware
        @self.app.middleware("http")
        async def performance_monitoring(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time

            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)

            # Update performance stats
            self.performance_stats['requests_handled'] += 1

            return response

        # WebSocket endpoint with rate limiting
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket(websocket)

        # Enhanced health check endpoint with detailed status
        rate_limit_decorator = self.limiter.limit("10/minute") if self.limiter else lambda f: f

        @self.app.get("/health")
        @rate_limit_decorator
        async def health_check(request: Request):
            uptime = time.time() - self.performance_stats['start_time']

            return {
                "status": "ok",
                "message": "Neural Nexus IDE Server v6.0 - Enhanced with Security & Performance",
                "version": "6.0",
                "uptime_seconds": round(uptime, 2),
                "performance": {
                    "requests_handled": self.performance_stats['requests_handled'],
                    "scripts_executed": self.performance_stats['scripts_executed'],
                    "analysis_runs": self.performance_stats['analysis_runs'],
                    "active_clients": len(self.clients),
                    "running_scripts": len(self.running_scripts),
                    "cached_projects": len(self.projects)
                },
                "features": {
                    "openai": HAS_OPENAI,
                    "uvloop": HAS_UVLOOP,
                    "orjson": HAS_ORJSON,
                    "loguru": HAS_LOGURU,
                    "rate_limiting": HAS_SLOWAPI,
                    "semgrep": HAS_SEMGREP,
                    "bandit": HAS_BANDIT,
                    "ruff": HAS_RUFF,
                    "msgspec": HAS_MSGSPEC,
                    "psutil": psutil is not None
                }
            }

        # Serve index.html at root with proper headers
        @self.app.get("/", response_class=HTMLResponse)
        async def serve_index():
            index_path = Path("static/index.html")

            # If file doesn't exist, create it
            if not index_path.exists():
                index_path.parent.mkdir(exist_ok=True)
                # Get the HTML content from the artifact
                html_content = self.get_complete_html()
                index_path.write_text(html_content, encoding='utf-8')
                logger.info("Created index.html file")

            # Read and return with proper headers
            content = index_path.read_text(encoding='utf-8')
            return HTMLResponse(
                content=content,
                headers={
                    "Content-Type": "text/html; charset=utf-8",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )

        # API endpoint to get server info
        @self.app.get("/api/info")
        async def get_info():
            return {
                "server": "Neural Nexus IDE",
                "version": "6.0",
                "running_scripts": len(self.running_scripts),
                "loaded_scripts": len(self.script_cache),
                "projects": len(self.projects),
                "clients": len(self.clients)
            }

        # New API endpoint for code formatting
        @self.app.post("/api/format")
        @rate_limit_decorator
        async def format_code(request: Request):
            try:
                data = await request.json()
                content = data.get('content', '')

                if not content:
                    return {"error": "No content provided"}

                formatted_code = await self.format_code_with_ruff(content)
                return {
                    "formatted_code": formatted_code,
                    "success": True
                }

            except Exception as e:
                logger.error(f"Code formatting error: {e}")
                return {"error": str(e), "success": False}

        # New API endpoint for security analysis
        @self.app.post("/api/security-scan")
        @rate_limit_decorator
        async def security_scan(request: Request):
            try:
                data = await request.json()
                content = data.get('content', '')

                if not content:
                    return {"error": "No content provided"}

                # Run quick security analysis
                analysis = ScriptAnalysis()

                if HAS_BANDIT:
                    await self.run_bandit_analysis(content, analysis)
                if HAS_SEMGREP:
                    await self.run_semgrep_analysis(content, analysis)

                return {
                    "security_issues": analysis.security_issues,
                    "bandit_findings": analysis.bandit_findings,
                    "semgrep_findings": analysis.semgrep_findings,
                    "total_issues": len(analysis.security_issues),
                    "success": True
                }

            except Exception as e:
                logger.error(f"Security scan error: {e}")
                return {"error": str(e), "success": False}

        # Mount static files if directory exists
        if Path("static").exists():
            self.app.mount("/static", StaticFiles(directory="static"), name="static")

    def get_complete_html(self) -> str:
        """Get the complete HTML content from the artifact"""
        try:
            # Check if there's an existing index.html file
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                return html_path.read_text(encoding='utf-8')

            # Check in static directory
            static_html_path = Path("static/index.html")
            if static_html_path.exists():
                return static_html_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read HTML file: {e}")

        # Return the embedded HTML content
        # Use the complete HTML from the static/index.html attachment
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Nexus IDE - AI-Powered Development Environment</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #252525;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --accent: #00ff88;
            --error: #ff3366;
            --warning: #ffaa00;
            --success: #00ff88;
            --border: #333;
        }

        body {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            overflow: hidden;
            height: 100vh;
        }

        .container {
            display: grid;
            grid-template-columns: 250px 1fr 400px;
            grid-template-rows: 60px 1fr 30px;
            height: 100vh;
            gap: 1px;
            background: var(--border);
        }

        .header {
            grid-column: 1 / -1;
            background: var(--bg-secondary);
            display: flex;
            align-items: center;
            padding: 0 20px;
            border-bottom: 1px solid var(--border);
        }

        .logo {
            font-size: 20px;
            font-weight: bold;
            color: var(--accent);
            margin-right: 40px;
        }

        .header-actions {
            display: flex;
            gap: 10px;
            margin-left: auto;
        }

        .btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 16px;
            cursor: pointer;
            border-radius: 4px;
            font-size: 12px;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .btn:hover {
            background: var(--accent);
            color: var(--bg-primary);
            transform: translateY(-1px);
        }

        .btn-primary {
            background: var(--accent);
            color: var(--bg-primary);
        }

        .sidebar {
            background: var(--bg-secondary);
            overflow-y: auto;
            padding: 10px;
        }

        .section {
            margin-bottom: 20px;
        }

        .section-title {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .script-list {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .script-item {
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s;
            border: 1px solid transparent;
        }

        .script-item:hover {
            background: #333;
            transform: translateX(2px);
            border-color: var(--accent);
        }

        .script-item.active {
            background: var(--accent);
            color: var(--bg-primary);
        }

        .editor-container {
            background: var(--bg-primary);
            display: flex;
            flex-direction: column;
        }

        #codeEditor {
            width: 100%;
            height: 100%;
            background: var(--bg-primary);
            color: var(--text-primary);
            border: none;
            padding: 20px;
            font-family: inherit;
            font-size: 14px;
            line-height: 1.6;
            resize: none;
            outline: none;
        }

        .tools-panel {
            background: var(--bg-secondary);
            overflow-y: auto;
            padding: 20px;
        }

        .status-bar {
            grid-column: 1 / -1;
            background: var(--bg-secondary);
            display: flex;
            align-items: center;
            padding: 0 20px;
            font-size: 12px;
            color: var(--text-secondary);
            border-top: 1px solid var(--border);
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="logo">⚡ Neural Nexus IDE v6.0</div>
            <div class="header-actions">
                <button class="btn" onclick="newScript()">📝 New Script</button>
                <button class="btn" onclick="formatCode()">🎨 Format</button>
                <button class="btn" onclick="securityScan()">🔒 Security</button>
                <button class="btn btn-primary" onclick="runScript()">▶️ Run</button>
                <button class="btn" onclick="stopScript()">⏹️ Stop</button>
            </div>
        </header>

        <aside class="sidebar">
            <div class="section">
                <div class="section-title">Scripts</div>
                <div class="script-list" id="scriptList">
                    <!-- Scripts will be populated here -->
                </div>
            </div>
        </aside>

        <main class="editor-container">
            <textarea id="codeEditor" placeholder="Enter your Python code here..."></textarea>
        </main>

        <aside class="tools-panel">
            <div class="section-title">Output</div>
            <div id="output" style="background: #000; color: #fff; padding: 10px; border-radius: 4px; min-height: 200px; font-family: monospace; white-space: pre-wrap; overflow-y: auto;"></div>
        </aside>

        <footer class="status-bar">
            <span id="status">Ready</span>
        </footer>
    </div>

    <script>
        // State Management
        let scripts = {};
        let activeScriptId = null;
        let isRunning = false;
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 10;

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            setupWebSocket();
        });

        // WebSocket Connection with retry logic
        function setupWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;

            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                reconnectAttempts = 0;  // Reset on successful connection
                updateStatus('Connected to Neural Nexus IDE Server', 'success');
                console.log('Connected to Neural Nexus IDE Server');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleServerMessage(data);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                updateStatus('Connection error', 'error');
            };

            ws.onclose = () => {
                updateStatus('Disconnected', 'error');

                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
                    setTimeout(setupWebSocket, delay);
                    updateStatus(`Reconnecting in ${delay/1000}s (attempt ${reconnectAttempts})`, 'warning');
                }
            };
        }

        // Server Message Handler
        function handleServerMessage(data) {
            switch(data.type) {
                case 'connected':
                    updateStatus(data.message, 'success');
                    break;
                case 'output':
                    addOutput(data.text, data.outputType);
                    break;
                case 'scriptComplete':
                    updateStatus(data.success ? 'Script completed successfully' : 'Script failed', data.success ? 'success' : 'error');
                    isRunning = false;
                    break;
                case 'healed':
                    document.getElementById('codeEditor').value = data.healedCode;
                    updateStatus('Code auto-healed by AI', 'success');
                    break;
                case 'error':
                    updateStatus(data.message, 'error');
                    break;
                default:
                    console.log('Received:', data);
            }
        }

        // Script Management
        function newScript() {
            const name = prompt('Script name (e.g., main.py, utils.py):');
            if (!name) return;

            const id = 'script_' + Date.now();
            const script = {
                id: id,
                name: name,
                content: '',
                status: 'idle'
            };

            scripts[id] = script;
            addScriptToList(script);
            selectScript(id);
        }

        function addScriptToList(script) {
            const list = document.getElementById('scriptList');
            const item = document.createElement('div');
            item.className = 'script-item';
            item.id = 'script_' + script.id;
            item.innerHTML = `<span>${script.name}</span>`;
            item.onclick = () => selectScript(script.id);
            list.appendChild(item);
        }

        function selectScript(id) {
            activeScriptId = id;
            const script = scripts[id];

            // Update UI
            document.querySelectorAll('.script-item').forEach(el => {
                el.classList.remove('active');
            });
            document.getElementById('script_' + id).classList.add('active');

            document.getElementById('codeEditor').value = script.content;
        }

        // Script Execution
        function runScript() {
            if (!activeScriptId) {
                newScript();
                if (!activeScriptId) return;
            }

            if (isRunning) {
                updateStatus('A script is already running', 'warning');
                return;
            }

            const script = scripts[activeScriptId];
            script.content = document.getElementById('codeEditor').value;

            isRunning = true;
            clearOutput();
            updateStatus('Running script...', 'info');

            ws.send(JSON.stringify({
                action: 'run',
                scriptId: activeScriptId,
                content: script.content,
                scriptName: script.name
            }));
        }

        function stopScript() {
            if (!isRunning) {
                updateStatus('No script is running', 'warning');
                return;
            }

            ws.send(JSON.stringify({
                action: 'stop',
                scriptId: activeScriptId
            }));

            isRunning = false;
            updateStatus('Script stopped', 'info');
        }

        // Enhanced Features - Code Formatting
        async function formatCode() {
            if (!activeScriptId) {
                updateStatus('No script selected', 'warning');
                return;
            }

            const content = document.getElementById('codeEditor').value;
            if (!content.trim()) {
                updateStatus('No code to format', 'warning');
                return;
            }

            updateStatus('Formatting code...', 'info');

            try {
                const response = await fetch('/api/format', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ content: content })
                });

                const result = await response.json();

                if (result.success) {
                    document.getElementById('codeEditor').value = result.formatted_code;
                    if (activeScriptId && scripts[activeScriptId]) {
                        scripts[activeScriptId].content = result.formatted_code;
                    }
                    updateStatus('Code formatted successfully!', 'success');
                } else {
                    updateStatus('Formatting failed: ' + (result.error || 'Unknown error'), 'error');
                }
            } catch (error) {
                console.error('Format error:', error);
                updateStatus('Formatting failed: ' + error.message, 'error');
            }
        }

        // Enhanced Features - Security Scanning
        async function securityScan() {
            if (!activeScriptId) {
                updateStatus('No script selected', 'warning');
                return;
            }

            const content = document.getElementById('codeEditor').value;
            if (!content.trim()) {
                updateStatus('No code to scan', 'warning');
                return;
            }

            updateStatus('Running security scan...', 'info');
            clearOutput();

            try {
                const response = await fetch('/api/security-scan', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ content: content })
                });

                const result = await response.json();

                if (result.success) {
                    const issueCount = result.total_issues || 0;

                    if (issueCount === 0) {
                        addOutput('✅ No security issues found!', 'success');
                        updateStatus('Security scan complete - no issues found', 'success');
                    } else {
                        addOutput(`⚠️  Found ${issueCount} security issue(s):`, 'warning');

                        // Display Bandit findings
                        if (result.bandit_findings && result.bandit_findings.length > 0) {
                            addOutput('\\n🔍 Bandit Security Analysis:', 'info');
                            result.bandit_findings.forEach(finding => {
                                addOutput(`  Line ${finding.line}: ${finding.message} (${finding.severity})`, 'warning');
                            });
                        }

                        // Display Semgrep findings
                        if (result.semgrep_findings && result.semgrep_findings.length > 0) {
                            addOutput('\\n🛡️  Semgrep Security Analysis:', 'info');
                            result.semgrep_findings.forEach(finding => {
                                addOutput(`  Line ${finding.line}: ${finding.message} (${finding.severity})`, 'warning');
                            });
                        }

                        updateStatus(`Security scan complete - ${issueCount} issues found`, 'warning');
                    }
                } else {
                    addOutput('❌ Security scan failed: ' + (result.error || 'Unknown error'), 'error');
                    updateStatus('Security scan failed', 'error');
                }
            } catch (error) {
                console.error('Security scan error:', error);
                addOutput('❌ Security scan error: ' + error.message, 'error');
                updateStatus('Security scan failed', 'error');
            }
        }

        // UI Updates
        function updateStatus(message, type = 'info') {
            const status = document.getElementById('status');
            status.textContent = message;

            // Remove existing classes
            status.className = '';
            status.classList.add(type);
        }

        function addOutput(text, type = 'normal') {
            const output = document.getElementById('output');
            const line = document.createElement('div');
            line.textContent = text;
            line.className = 'output-line ' + type;
            output.appendChild(line);
            output.scrollTop = output.scrollHeight;
        }

        function clearOutput() {
            document.getElementById('output').innerHTML = '';
        }

        // Auto-save editor content
        document.getElementById('codeEditor').addEventListener('input', () => {
            if (activeScriptId && scripts[activeScriptId]) {
                scripts[activeScriptId].content = document.getElementById('codeEditor').value;
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    runScript();
                } else if (e.key === 's') {
                    e.preventDefault();
                    updateStatus('Auto-saved', 'success');
                }
            }
        });
    </script>
</body>
</html>'''

    async def handle_websocket(self, websocket):
        """Handle WebSocket connections with improved error handling"""
        await websocket.accept()
        self.clients.add(websocket)
        client_id = id(websocket)
        self.terminal_history[client_id] = []

        logger.info(f"Client connected. Total clients: {len(self.clients)}")

        # Send initial connection success message
        await websocket.send_json({
            'type': 'connected',
            'message': 'Connected to Neural Nexus IDE Server v5.0'
        })

        try:
            while True:
                data = await websocket.receive_json()
                await self.process_message(websocket, data)
        except WebSocketDisconnect:
            self.clients.remove(websocket)
            if client_id in self.terminal_history:
                del self.terminal_history[client_id]
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            logger.error(traceback.format_exc())
            if websocket in self.clients:
                self.clients.remove(websocket)
            if client_id in self.terminal_history:
                del self.terminal_history[client_id]

    async def process_message(self, websocket, data: Dict[str, Any]):
        """Process incoming WebSocket messages with enhanced error handling"""
        action = data.get('action')
        logger.info(f"Processing action: {action}")

        try:
            # Core actions
            if action == 'run':
                await self.run_script(websocket, data)
            elif action == 'stop':
                await self.stop_script(websocket, data)
            elif action == 'analyze':
                await self.analyze_script(websocket, data)
            elif action == 'autoHeal':
                await self.auto_heal_script(websocket, data)
            elif action == 'installDeps':
                await self.install_dependencies(websocket, data)
            elif action == 'terminal':
                await self.execute_terminal_command(websocket, data)
            elif action == 'openVSCode':
                await self.open_in_vscode(websocket, data)
            elif action == 'aiAnalyze':
                await self.ai_analyze(websocket, data)
            elif action == 'copilot':
                await self.copilot_suggest(websocket, data)
            elif action == 'toggleAutoHeal':
                await self.toggle_auto_heal(websocket, data)
            elif action == 'fixNow':
                await self.fix_and_apply(websocket, data)
            # Project management
            elif action == 'createProject':
                await self.create_project(websocket, data)
            elif action == 'addToProject':
                await self.add_script_to_project(websocket, data)
            elif action == 'runProject':
                await self.run_project(websocket, data)
            elif action == 'saveAllScripts':
                await self.save_all_scripts(websocket, data)
            # New actions
            elif action == 'exportProject':
                await self.export_project(websocket, data)
            elif action == 'importProject':
                await self.import_project(websocket, data)
            elif action == 'getScriptGraph':
                await self.get_script_graph(websocket, data)
            else:
                await websocket.send_json({
                    'type': 'error',
                    'message': f'Unknown action: {action}'
                })
        except Exception as e:
            logger.error(f"Error processing {action}: {e}")
            logger.error(traceback.format_exc())
            await websocket.send_json({
                'type': 'error',
                'message': f'Error: {str(e)}'
            })

    async def run_script(self, websocket, data: Dict[str, Any]):
        """Run a Python script with enhanced error handling and project support"""
        script_id = data['scriptId']
        content = data['content']
        script_name = data.get('scriptName', f"{script_id}.py")
        project_id = data.get('projectId')

        # Cache API key if provided
        if 'apiKey' in data:
            self._cached_api_key = data['apiKey']

        # Update max heal attempts if provided
        if 'maxHealAttempts' in data:
            self.max_heal_attempts = data['maxHealAttempts']

        # Stop existing process if running
        if script_id in self.running_scripts:
            await self.stop_script(websocket, {'scriptId': script_id})

        # Determine script location
        if project_id:
            project_dir = PROJECTS_DIR / f"project_{project_id}"
            project_dir.mkdir(exist_ok=True)

            # Add project directory to Python path
            if str(project_dir) not in sys.path:
                sys.path.insert(0, str(project_dir))

            # Save all project scripts to ensure imports work
            if project_id in self.projects:
                await self.save_project_scripts(project_id)

            script_path = project_dir / script_name
        else:
            # Regular single script
            script_path = TEMP_DIR / f"{script_id}.py"

        # Save script with proper encoding
        try:
            if aiofiles:
                async with aiofiles.open(script_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
            else:
                script_path.write_text(content, encoding='utf-8')
        except Exception as e:
            await websocket.send_json({
                'type': 'output',
                'scriptId': script_id,
                'text': f"Failed to save script: {e}",
                'outputType': 'error'
            })
            return

        # Cache the script content
        self.script_cache[script_id] = {
            'content': content,
            'name': script_name,
            'path': str(script_path),
            'project_id': project_id
        }

        # For module reloading
        if script_name.endswith('.py'):
            module_name = script_name[:-3]
            if module_name in sys.modules:
                try:
                    importlib.reload(sys.modules[module_name])
                except Exception as e:
                    logger.warning(f"Failed to reload module {module_name}: {e}")
                    if module_name in sys.modules:
                        del sys.modules[module_name]

        # Determine Python command - prefer uv if available
        python_cmd = self.get_python_command()

        try:
            # Create process with proper working directory and environment
            cwd = str(script_path.parent)

            # Enhanced environment setup
            env = os.environ.copy()
            env['PYTHONPATH'] = cwd + os.pathsep + os.environ.get('PYTHONPATH', '')
            env['PYTHONUNBUFFERED'] = '1'  # Ensure real-time output

            # Add project directory to PYTHONPATH if in a project
            if project_id:
                project_root = str(PROJECTS_DIR / f"project_{project_id}")
                env['PYTHONPATH'] = project_root + os.pathsep + env['PYTHONPATH']

            process = await asyncio.create_subprocess_exec(
                *python_cmd, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env
            )

            # Store process info
            script_process = ScriptProcess(
                process=process,
                script_id=script_id,
                start_time=asyncio.get_event_loop().time(),
                script_name=script_name,
                project_id=project_id
            )
            self.running_scripts[script_id] = script_process

            # Send start message
            await websocket.send_json({
                'type': 'output',
                'scriptId': script_id,
                'text': f"--- Running {script_name} ---",
                'outputType': 'info'
            })

            # Monitor output
            asyncio.create_task(self.monitor_script_output(websocket, script_process))

        except Exception as e:
            await websocket.send_json({
                'type': 'output',
                'scriptId': script_id,
                'text': f"Failed to start script: {e}",
                'outputType': 'error'
            })

    def get_python_command(self) -> List[str]:
        """Get the appropriate Python command"""
        if shutil.which("uv"):
            return ["uv", "run", "python"]
        elif platform.system() == "Windows":
            return [sys.executable]
        else:
            # Try python3 first, then python
            if shutil.which("python3"):
                return ["python3"]
            else:
                return [sys.executable]

    async def monitor_script_output(self, websocket, script_process: ScriptProcess):
        """Monitor script output with improved error handling"""
        async def read_stream(stream, output_type):
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break

                    try:
                        text = line.decode('utf-8', errors='replace').rstrip()
                    except:
                        text = str(line).rstrip()

                    # Store output
                    if output_type == 'error':
                        script_process.error_lines.append(text)
                    else:
                        script_process.output_lines.append(text)

                    # Send to client
                    await websocket.send_json({
                        'type': 'output',
                        'scriptId': script_process.script_id,
                        'text': text,
                        'outputType': output_type
                    })
            except Exception as e:
                logger.error(f"Error reading stream: {e}")

        # Read both stdout and stderr
        try:
            await asyncio.gather(
                read_stream(script_process.process.stdout, 'normal'),
                read_stream(script_process.process.stderr, 'error')
            )
        except Exception as e:
            logger.error(f"Error monitoring output: {e}")

        # Wait for process to complete
        try:
            returncode = await script_process.process.wait()
        except:
            returncode = -1

        # Send completion message
        await websocket.send_json({
            'type': 'scriptComplete',
            'scriptId': script_process.script_id,
            'success': returncode == 0,
            'returnCode': returncode
        })

        # Cleanup
        if script_process.script_id in self.running_scripts:
            del self.running_scripts[script_process.script_id]

        # Auto-heal if enabled and script failed
        if returncode != 0 and self.auto_heal_mode.get(script_process.script_id, False):
            await self.handle_auto_heal(websocket, script_process)

    async def handle_auto_heal(self, websocket, script_process: ScriptProcess):
        """Handle automatic healing for failed scripts"""
        script_id = script_process.script_id
        error_text = '\n'.join(script_process.error_lines)
        output_text = '\n'.join(script_process.output_lines)

        # Get script content
        cached = self.script_cache.get(script_id, {})
        content = cached.get('content', '')

        if not content:
            content = await self.read_script_content(script_id)

        # Store error for auto-heal
        self.script_cache[script_id] = {
            **cached,
            'last_error': error_text,
            'last_output': output_text,
            'content': content
        }

        # Increment heal attempts
        self.heal_attempts[script_id] = self.heal_attempts.get(script_id, 0) + 1

        # Check if we should continue healing
        if self.heal_attempts[script_id] <= self.max_heal_attempts:
            await websocket.send_json({
                'type': 'autoHealStatus',
                'scriptId': script_id,
                'message': f'Auto-heal attempt {self.heal_attempts[script_id]}/{self.max_heal_attempts}...',
                'attempts': self.heal_attempts[script_id]
            })

            # Trigger automatic healing
            await asyncio.sleep(1)  # Brief pause
            await self.auto_heal_and_retry(websocket, script_id)
        else:
            # Max attempts reached
            await websocket.send_json({
                'type': 'autoHealStatus',
                'scriptId': script_id,
                'message': 'Max auto-heal attempts reached. Manual intervention required.',
                'attempts': self.heal_attempts[script_id],
                'stopped': True
            })
            self.auto_heal_mode[script_id] = False

    async def auto_heal_and_retry(self, websocket, script_id: str):
        """Automatically heal and retry a failed script with enhanced logic"""
        cached = self.script_cache.get(script_id, {})
        content = cached.get('content', '')
        error_text = cached.get('last_error', '')
        output_text = cached.get('last_output', '')

        if not content:
            return

        # First, analyze the script for syntax errors
        analysis = await self.analyze_script_content(content)

        # Use stored API key if available
        api_key = self._cached_api_key

        if not api_key:
            await websocket.send_json({
                'type': 'error',
                'message': 'API key required for auto-heal. Please configure in Settings.'
            })
            self.auto_heal_mode[script_id] = False
            return

        # Build comprehensive error description
        error_desc = self.build_error_description(analysis, error_text, output_text)

        # Call AI to fix the code
        fixed_code = await self.get_fixed_code(content, error_desc, api_key)

        if fixed_code and fixed_code != content:
            # Save the fixed code
            script_path = Path(cached.get('path', TEMP_DIR / f"{script_id}.py"))

            try:
                if aiofiles:
                    async with aiofiles.open(script_path, 'w', encoding='utf-8') as f:
                        await f.write(fixed_code)
                else:
                    script_path.write_text(fixed_code, encoding='utf-8')
            except Exception as e:
                logger.error(f"Failed to save fixed code: {e}")
                return

            # Update cache
            self.script_cache[script_id]['content'] = fixed_code

            # Send the fixed code to client
            await websocket.send_json({
                'type': 'healed',
                'scriptId': script_id,
                'healedCode': fixed_code
            })

            # Wait a moment then re-run
            await asyncio.sleep(0.5)

            # Re-run the script
            await self.run_script(websocket, {
                'scriptId': script_id,
                'content': fixed_code,
                'scriptName': cached.get('name', f"{script_id}.py"),
                'projectId': cached.get('project_id')
            })
        else:
            # Couldn't fix the code
            await websocket.send_json({
                'type': 'autoHealStatus',
                'scriptId': script_id,
                'message': 'Unable to fix the code automatically',
                'stopped': True
            })
            self.auto_heal_mode[script_id] = False

    def build_error_description(self, analysis: ScriptAnalysis, error_text: str, output_text: str) -> str:
        """Build a comprehensive error description for AI"""
        error_desc = ""

        # Add syntax errors
        if analysis.errors:
            error_desc += "Syntax/Static Errors:\n"
            for err in analysis.errors:
                error_desc += f"  Line {err.get('line', 0)}: {err['message']}\n"

        # Add warnings
        if analysis.warnings:
            error_desc += "\nWarnings:\n"
            for warn in analysis.warnings:
                error_desc += f"  Line {warn.get('line', 0)}: {warn['message']}\n"

        # Add runtime errors
        if error_text:
            error_desc += "\nRuntime Errors:\n" + error_text

        # Add output for context (last 1000 chars)
        if output_text:
            error_desc += "\nScript Output:\n" + output_text[-1000:]

        return error_desc

    async def get_fixed_code(self, content: str, error_desc: str, api_key: str) -> Optional[str]:
        """Get fixed code from AI with improved prompt engineering"""
        if not HAS_OPENAI or not api_key:
            return None

        prompt = f"""Fix this Python code that has errors. Return ONLY the complete fixed Python code, no explanations.

Original Code:
```python
{content}
```

Errors and Issues:
{error_desc}

Instructions:
1. Fix ALL syntax errors
2. Fix ALL runtime errors based on the error messages
3. Add proper error handling where needed
4. Ensure all imports are correct and available
5. Handle edge cases that might cause errors
6. Maintain the original functionality and intent
7. Return the COMPLETE fixed code

Fixed Code:"""

        try:
            client = OpenAI(api_key=api_key)

            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4000
                )
            )

            fixed_code = response.choices[0].message.content.strip()

            # Clean up response - remove markdown code blocks
            fixed_code = self.clean_ai_code_response(fixed_code)

            return fixed_code

        except Exception as e:
            logger.error(f"Failed to get fixed code: {e}")
            return None

    def clean_ai_code_response(self, code: str) -> str:
        """Clean up AI response to extract pure code"""
        # Remove markdown code blocks
        if '```' in code:
            # Find code between triple backticks
            pattern = r'```(?:python)?\n?(.*?)```'
            match = re.search(pattern, code, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Remove any leading/trailing quotes
        code = code.strip('"\'')

        return code.strip()

    async def analyze_script_content(self, content: str) -> ScriptAnalysis:
        """Enhanced script analysis with security scanning, type checking, and performance hints"""
        analysis = ScriptAnalysis()

        # Update performance stats
        self.performance_stats['analysis_runs'] += 1

        try:
            # Parse AST
            tree = ast.parse(content)

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    analysis.imports.append(node.module)

            # Perform basic analysis
            await self.perform_advanced_analysis(content, tree, analysis)

            # Run security analysis with Bandit
            if HAS_BANDIT and self.security_enabled:
                await self.run_bandit_analysis(content, analysis)

            # Run Semgrep security scan
            if HAS_SEMGREP and self.security_enabled:
                await self.run_semgrep_analysis(content, analysis)

            # Run Ruff linting and formatting suggestions
            if HAS_RUFF:
                await self.run_ruff_analysis(content, analysis)

            # Run Pyright type checking
            await self.run_type_analysis(content, analysis)

            # Calculate code quality score
            analysis.code_quality_score = self.calculate_quality_score(analysis)

        except SyntaxError as e:
            analysis.errors.append({
                'type': 'SyntaxError',
                'line': e.lineno or 0,
                'message': str(e.msg),
                'offset': e.offset,
                'text': e.text,
                'severity': 'error'
            })
        except Exception as e:
            analysis.errors.append({
                'type': 'ParseError',
                'line': 0,
                'message': str(e),
                'severity': 'error'
            })

        return analysis

    async def run_bandit_analysis(self, content: str, analysis: ScriptAnalysis):
        """Run Bandit security analysis on the code"""
        try:
            # Create temporary file for Bandit analysis
            temp_file = TEMP_DIR / f"bandit_temp_{hash(content) % 10000}.py"
            temp_file.write_text(content, encoding='utf-8')

            # Run Bandit analysis
            cmd = [sys.executable, "-m", "bandit", "-f", "json", str(temp_file)]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if stdout:
                try:
                    bandit_results = json_loads(stdout.decode())

                    for result in bandit_results.get('results', []):
                        analysis.bandit_findings.append({
                            'type': 'SecurityIssue',
                            'line': result.get('line_number', 0),
                            'message': result.get('issue_text', ''),
                            'severity': result.get('issue_severity', 'medium').lower(),
                            'confidence': result.get('issue_confidence', 'medium').lower(),
                            'test_id': result.get('test_id', ''),
                            'test_name': result.get('test_name', ''),
                            'code': result.get('code', ''),
                            'tool': 'bandit'
                        })

                        # Also add to security_issues for unified access
                        analysis.security_issues.append(analysis.bandit_findings[-1])

                except Exception as e:
                    logger.warning(f"Failed to parse Bandit results: {e}")

            # Cleanup
            temp_file.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Bandit analysis failed: {e}")

    async def run_semgrep_analysis(self, content: str, analysis: ScriptAnalysis):
        """Run Semgrep security and bug analysis"""
        try:
            # Create temporary file for Semgrep analysis
            temp_file = TEMP_DIR / f"semgrep_temp_{hash(content) % 10000}.py"
            temp_file.write_text(content, encoding='utf-8')

            # Run Semgrep with Python security rules
            cmd = [
                sys.executable, "-m", "semgrep",
                "--config=auto",  # Use automatic rule selection
                "--json",
                "--no-git-ignore",
                str(temp_file)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if stdout:
                try:
                    semgrep_results = json_loads(stdout.decode())

                    for result in semgrep_results.get('results', []):
                        severity = result.get('extra', {}).get('severity', 'info')

                        finding = {
                            'type': 'SecurityVulnerability' if severity in ['error', 'warning'] else 'CodeIssue',
                            'line': result.get('start', {}).get('line', 0),
                            'message': result.get('extra', {}).get('message', ''),
                            'severity': severity.lower(),
                            'rule_id': result.get('check_id', ''),
                            'category': result.get('extra', {}).get('metadata', {}).get('category', []),
                            'confidence': result.get('extra', {}).get('metadata', {}).get('confidence', 'medium'),
                            'tool': 'semgrep'
                        }

                        analysis.semgrep_findings.append(finding)
                        analysis.security_issues.append(finding)

                except Exception as e:
                    logger.warning(f"Failed to parse Semgrep results: {e}")

            # Cleanup
            temp_file.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Semgrep analysis failed: {e}")

    async def run_ruff_analysis(self, content: str, analysis: ScriptAnalysis):
        """Run Ruff linting and get formatting suggestions"""
        try:
            # Create temporary file for Ruff analysis
            temp_file = TEMP_DIR / f"ruff_temp_{hash(content) % 10000}.py"
            temp_file.write_text(content, encoding='utf-8')

            # Run Ruff check
            cmd = [sys.executable, "-m", "ruff", "check", "--output-format=json", str(temp_file)]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if stdout:
                try:
                    ruff_results = json_loads(stdout.decode())

                    for result in ruff_results:
                        severity = 'warning' if result.get('code', '').startswith('W') else 'error'

                        finding = {
                            'type': 'StyleIssue',
                            'line': result.get('location', {}).get('row', 0),
                            'column': result.get('location', {}).get('column', 0),
                            'message': result.get('message', ''),
                            'severity': severity,
                            'rule_code': result.get('code', ''),
                            'rule_url': result.get('url', ''),
                            'fixable': result.get('fix', {}).get('applicability') == 'automatic',
                            'tool': 'ruff'
                        }

                        analysis.ruff_suggestions.append(finding)

                        # Add to warnings for unified display
                        analysis.warnings.append(finding)

                except Exception as e:
                    logger.warning(f"Failed to parse Ruff results: {e}")

            # Cleanup
            temp_file.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Ruff analysis failed: {e}")

    async def run_type_analysis(self, content: str, analysis: ScriptAnalysis):
        """Run Pyright type checking analysis"""
        try:
            # Create temporary file for type checking
            temp_file = TEMP_DIR / f"pyright_temp_{hash(content) % 10000}.py"
            temp_file.write_text(content, encoding='utf-8')

            # Run Pyright type checking
            cmd = [sys.executable, "-m", "pyright", "--outputjson", str(temp_file)]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if stdout:
                try:
                    pyright_results = json_loads(stdout.decode())

                    for diagnostic in pyright_results.get('generalDiagnostics', []):
                        severity = diagnostic.get('severity', 'information').lower()

                        type_error = {
                            'type': 'TypeError',
                            'line': diagnostic.get('range', {}).get('start', {}).get('line', 0) + 1,
                            'column': diagnostic.get('range', {}).get('start', {}).get('character', 0),
                            'message': diagnostic.get('message', ''),
                            'severity': severity,
                            'rule': diagnostic.get('rule', ''),
                            'tool': 'pyright'
                        }

                        analysis.type_errors.append(type_error)

                        # Add to appropriate category based on severity
                        if severity == 'error':
                            analysis.errors.append(type_error)
                        else:
                            analysis.warnings.append(type_error)

                except Exception as e:
                    logger.warning(f"Failed to parse Pyright results: {e}")

            # Cleanup
            temp_file.unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Pyright analysis failed: {e}")

    def calculate_quality_score(self, analysis: ScriptAnalysis) -> float:
        """Calculate a code quality score based on analysis results"""
        try:
            # Base score
            score = 100.0

            # Deduct points for issues
            score -= len(analysis.errors) * 15  # Syntax/parse errors are severe
            score -= len([w for w in analysis.warnings if w.get('severity') == 'error']) * 10
            score -= len([w for w in analysis.warnings if w.get('severity') == 'warning']) * 5
            score -= len(analysis.security_issues) * 20  # Security issues are critical
            score -= len([t for t in analysis.type_errors if t.get('severity') == 'error']) * 8

            # Slight deductions for style issues
            score -= len([r for r in analysis.ruff_suggestions if r.get('severity') == 'warning']) * 2

            # Bonus points for good practices
            if any('docstring' in w.get('message', '').lower() for w in analysis.suggestions):
                score += 5  # Has docstring suggestions (implies some structure)

            # Ensure score is between 0 and 100
            return max(0.0, min(100.0, score))

        except Exception as e:
            logger.warning(f"Failed to calculate quality score: {e}")
            return 50.0  # Default neutral score

    async def format_code_with_ruff(self, content: str) -> str:
        """Format code using Ruff formatter"""
        try:
            if not HAS_RUFF:
                logger.warning("Ruff not available, returning original content")
                return content

            # Create temporary file for formatting
            temp_file = TEMP_DIR / f"format_temp_{hash(content) % 10000}.py"
            temp_file.write_text(content, encoding='utf-8')

            # Run Ruff format
            cmd = [sys.executable, "-m", "ruff", "format", "--stdin-filename", str(temp_file)]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate(input=content.encode())

            if process.returncode == 0 and stdout:
                formatted_code = stdout.decode('utf-8')
                logger.debug("Code formatted successfully with Ruff")
                return formatted_code
            else:
                logger.warning(f"Ruff formatting failed: {stderr.decode() if stderr else 'Unknown error'}")
                return content

        except Exception as e:
            logger.warning(f"Code formatting error: {e}")
            return content
        finally:
            # Cleanup
            if 'temp_file' in locals():
                temp_file.unlink(missing_ok=True)

    async def perform_advanced_analysis(self, content: str, tree: ast.AST, analysis: ScriptAnalysis):
        """Perform advanced code analysis"""
        lines = content.split('\n')

        # Check for common issues
        for i, line in enumerate(lines, 1):
            # Unused imports
            if line.strip().startswith(('import ', 'from ')):
                module_match = re.search(r'import\s+(\w+)|from\s+(\w+)', line)
                if module_match:
                    module = module_match.group(1) or module_match.group(2)
                    # Simple check - look for module usage in rest of file
                    remaining_content = '\n'.join(lines[i:])
                    if module not in remaining_content:
                        analysis.warnings.append({
                            'type': 'UnusedImport',
                            'line': i,
                            'message': f"Import '{module}' appears to be unused"
                        })

            # Long lines
            if len(line) > 120:
                analysis.warnings.append({
                    'type': 'LineTooLong',
                    'line': i,
                    'message': f"Line exceeds 120 characters ({len(line)} chars)"
                })

            # Missing docstrings for functions/classes
            stripped = line.strip()
            if stripped.startswith(('def ', 'class ')):
                # Check if next non-empty line is a docstring
                for j in range(i, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not next_line.startswith(('"""', "'''")):
                        analysis.suggestions.append({
                            'type': 'MissingDocstring',
                            'line': i,
                            'message': f"Consider adding a docstring"
                        })
                        break
                    elif next_line.startswith(('"""', "'''")):
                        break

        # Check for security issues
        security_patterns = [
            (r'eval\s*\(', 'Avoid using eval() - security risk'),
            (r'exec\s*\(', 'Avoid using exec() - security risk'),
            (r'__import__\s*\(', 'Avoid using __import__() - use import statement'),
            (r'pickle\.load', 'Be careful with pickle.load() - security risk'),
        ]

        for pattern, message in security_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    analysis.warnings.append({
                        'type': 'SecurityWarning',
                        'line': i,
                        'message': message
                    })

    async def stop_script(self, websocket, data: Dict[str, Any]):
        """Stop a running script with improved cleanup"""
        script_id = data['scriptId']

        if script_id in self.running_scripts:
            process = self.running_scripts[script_id].process
            try:
                # Try graceful termination first
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force kill if termination fails
                    process.kill()
                    await process.wait()
            except Exception as e:
                logger.error(f"Error stopping script: {e}")

            # Cleanup
            if script_id in self.running_scripts:
                del self.running_scripts[script_id]

            await websocket.send_json({
                'type': 'output',
                'scriptId': script_id,
                'text': '--- Script terminated ---',
                'outputType': 'info'
            })

            # Reset auto-heal if it was active
            if script_id in self.auto_heal_mode:
                self.auto_heal_mode[script_id] = False
            if script_id in self.heal_attempts:
                self.heal_attempts[script_id] = 0

    async def analyze_script(self, websocket, data: Dict[str, Any]):
        """Analyze a script for errors and dependencies"""
        script_id = data['scriptId']
        content = data['content']

        analysis = await self.analyze_script_content(content)

        # Check dependencies
        await self.update_installed_packages()

        for imp in analysis.imports:
            base_module = imp.split('.')[0]

            # Skip standard library modules
            if base_module in sys.stdlib_module_names:
                continue

            # Check if installed
            if base_module.lower() not in self.installed_packages:
                analysis.dependencies.append({
                    'name': base_module,
                    'status': 'missing'
                })
            else:
                analysis.dependencies.append({
                    'name': base_module,
                    'status': 'installed'
                })

        # Send results
        await websocket.send_json({
            'type': 'analyzed',
            'scriptId': script_id,
            'errors': analysis.errors,
            'dependencies': analysis.dependencies,
            'warnings': analysis.warnings,
            'suggestions': analysis.suggestions
        })

    async def toggle_auto_heal(self, websocket, data: Dict[str, Any]):
        """Toggle continuous auto-heal mode for a script"""
        script_id = data['scriptId']
        enabled = data.get('enabled', False)

        self.auto_heal_mode[script_id] = enabled

        if enabled:
            self.heal_attempts[script_id] = 0
            await websocket.send_json({
                'type': 'autoHealStatus',
                'scriptId': script_id,
                'message': 'Continuous auto-heal enabled',
                'enabled': True
            })
        else:
            if script_id in self.heal_attempts:
                del self.heal_attempts[script_id]
            await websocket.send_json({
                'type': 'autoHealStatus',
                'scriptId': script_id,
                'message': 'Continuous auto-heal disabled',
                'enabled': False
            })

    async def fix_and_apply(self, websocket, data: Dict[str, Any]):
        """Immediately fix and apply code changes"""
        script_id = data['scriptId']
        content = data['content']
        api_key = data.get('apiKey')

        if not HAS_OPENAI or not api_key:
            await websocket.send_json({
                'type': 'error',
                'message': 'OpenAI API key required'
            })
            return

        # Cache the API key for auto-heal
        self._cached_api_key = api_key

        # Analyze the script first
        analysis = await self.analyze_script_content(content)

        prompt = f"""Analyze and FIX this Python code. Return ONLY the complete improved/fixed code.

```python
{content}
```

Current Issues Found:
{self.format_analysis_for_prompt(analysis)}

Fix ALL of these issues:
1. Syntax errors
2. Import errors
3. Logic errors
4. Add comprehensive error handling
5. Improve code quality and readability
6. Fix any potential bugs
7. Add type hints where appropriate
8. Ensure the code follows PEP 8 style guidelines

Return the COMPLETE fixed and improved code:"""

        try:
            client = OpenAI(api_key=api_key)

            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=4000
                )
            )

            fixed_code = response.choices[0].message.content.strip()
            fixed_code = self.clean_ai_code_response(fixed_code)

            # Send the fixed code back
            await websocket.send_json({
                'type': 'healed',
                'scriptId': script_id,
                'healedCode': fixed_code,
                'message': 'Code analyzed and fixed!'
            })

        except Exception as e:
            logger.error(f"Fix and apply error: {e}")
            error_msg = self.format_openai_error(str(e))

            await websocket.send_json({
                'type': 'error',
                'message': f'Failed to fix code: {error_msg}'
            })

    def format_analysis_for_prompt(self, analysis: ScriptAnalysis) -> str:
        """Format analysis results for AI prompt"""
        result = []

        if analysis.errors:
            result.append("Errors:")
            for err in analysis.errors:
                result.append(f"  - Line {err.get('line', 0)}: {err['message']}")

        if analysis.warnings:
            result.append("\nWarnings:")
            for warn in analysis.warnings:
                result.append(f"  - Line {warn.get('line', 0)}: {warn['message']}")

        if analysis.suggestions:
            result.append("\nSuggestions:")
            for sug in analysis.suggestions:
                result.append(f"  - Line {sug.get('line', 0)}: {sug['message']}")

        return '\n'.join(result) if result else "No issues found"

    def format_openai_error(self, error_msg: str) -> str:
        """Format OpenAI error messages to be user-friendly"""
        if "Invalid API key" in error_msg or "Incorrect API key" in error_msg:
            return "Invalid OpenAI API key. Please check your API key in Settings."
        elif "quota" in error_msg.lower():
            return "OpenAI API quota exceeded. Please check your usage limits."
        elif "model" in error_msg.lower():
            return "Model access error. Make sure your API key has access to the required model."
        else:
            return error_msg

    async def auto_heal_script(self, websocket, data: Dict[str, Any]):
        """Manually triggered auto-heal (legacy compatibility)"""
        script_id = data['scriptId']
        content = data['content']
        errors = data.get('errors', [])
        api_key = data.get('apiKey')

        if not HAS_OPENAI or not api_key:
            await websocket.send_json({
                'type': 'error',
                'message': 'OpenAI API key required for auto-heal'
            })
            return

        # Get cached error if available
        cached = self.script_cache.get(script_id, {})
        runtime_error = cached.get('last_error', '')

        # Build error description
        error_desc = '\n'.join([
            f"Line {e.get('line', 0)}: {e['message']}"
            for e in errors
        ])

        if runtime_error:
            error_desc += f"\n\nRuntime error:\n{runtime_error}"

        # Get fixed code
        fixed_code = await self.get_fixed_code(content, error_desc, api_key)

        if fixed_code:
            await websocket.send_json({
                'type': 'healed',
                'scriptId': script_id,
                'healedCode': fixed_code
            })
        else:
            await websocket.send_json({
                'type': 'error',
                'message': 'Failed to heal code'
            })

    async def install_dependencies(self, websocket, data: Dict[str, Any]):
        """Install missing dependencies with improved package management"""
        packages = data['packages']

        # Determine package manager
        cmd = self.get_pip_command() + ["install"] + packages

        await websocket.send_json({
            'type': 'terminalOutput',
            'output': f"Installing packages: {' '.join(packages)}"
        })

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Stream output
            async def read_stream(stream):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    await websocket.send_json({
                        'type': 'terminalOutput',
                        'output': line.decode('utf-8', errors='replace').rstrip()
                    })

            await asyncio.gather(
                read_stream(process.stdout),
                read_stream(process.stderr)
            )

            returncode = await process.wait()

            if returncode == 0:
                await websocket.send_json({
                    'type': 'terminalOutput',
                    'output': 'Installation completed successfully!'
                })
                # Update installed packages
                await self.update_installed_packages()
            else:
                await websocket.send_json({
                    'type': 'terminalOutput',
                    'output': f'Installation failed with code {returncode}'
                })

        except Exception as e:
            await websocket.send_json({
                'type': 'terminalOutput',
                'output': f'Error: {str(e)}'
            })

    def get_pip_command(self) -> List[str]:
        """Get the appropriate pip command"""
        if shutil.which("uv"):
            return ["uv", "pip"]
        else:
            return [sys.executable, "-m", "pip"]

    async def execute_terminal_command(self, websocket, data: Dict[str, Any]):
        """Execute terminal command with enhanced shell support"""
        command = data['command']
        terminal_type = data.get('type', 'auto')

        # Security check
        if self.is_dangerous_command(command):
            await websocket.send_json({
                'type': 'terminalOutput',
                'output': 'Error: Potentially dangerous command blocked',
                'error': True
            })
            return

        # Store command in history
        client_id = id(websocket)
        if client_id in self.terminal_history:
            self.terminal_history[client_id].append(command)

        logger.info(f"Executing terminal command: {command} (type: {terminal_type})")

        # Get shell command
        shell_cmd = self.get_shell_command(command, terminal_type)

        if not shell_cmd:
            await websocket.send_json({
                'type': 'terminalOutput',
                'output': f'Error: Terminal type "{terminal_type}" not supported',
                'error': True
            })
            return

        try:
            # Create process
            process = await asyncio.create_subprocess_exec(
                *shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(TEMP_DIR)
            )

            # Get output with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()
                await websocket.send_json({
                    'type': 'terminalOutput',
                    'output': 'Command timed out after 30 seconds',
                    'error': True
                })
                return

            # Decode output
            output = self.decode_terminal_output(stdout, stderr)

            # Send output back
            await websocket.send_json({
                'type': 'terminalOutput',
                'output': output.strip() if output else '(no output)',
                'error': process.returncode != 0
            })

        except FileNotFoundError:
            await websocket.send_json({
                'type': 'terminalOutput',
                'output': f'Error: Shell not found for terminal type "{terminal_type}"',
                'error': True
            })
        except Exception as e:
            logger.error(f"Terminal command error: {e}")
            await websocket.send_json({
                'type': 'terminalOutput',
                'output': f'Error: {str(e)}',
                'error': True
            })

    def is_dangerous_command(self, command: str) -> bool:
        """Check if command is potentially dangerous"""
        dangerous_patterns = [
            r'rm\s+-rf\s+/',
            r'rm\s+-rf\s+~',
            r'format\s+[cC]:',
            r'del\s+/f\s+/s\s+/q',
            r'sudo\s+rm\s+-rf',
            r'rmdir\s+/s\s+/q',
            r'>\s*/dev/s[a-z]+',  # Fixed: was /dev/sda
            r'dd\s+if=/dev/zero\s+of=/dev/',
            r':(){ :|:& };:',  # Fork bomb
        ]

        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, command_lower):
                return True

        return False

    def get_shell_command(self, command: str, terminal_type: str) -> Optional[List[str]]:
        """Get shell command based on terminal type"""
        if platform.system() == 'Windows':
            if terminal_type == 'gitbash':
                # Try to find Git Bash
                git_path = shutil.which('git')
                if git_path:
                    bash_path = Path(git_path).parent.parent / 'bin' / 'bash.exe'
                    if bash_path.exists():
                        return [str(bash_path), '-c', command]
                return ['cmd', '/c', command]  # Fallback
            elif terminal_type == 'wsl':
                if shutil.which('wsl'):
                    return ['wsl', 'bash', '-c', command]
                return None
            elif terminal_type == 'powershell':
                return ['powershell', '-Command', command]
            elif terminal_type == 'cmd' or terminal_type == 'auto':
                return ['cmd', '/c', command]
        else:
            # Unix-like systems
            if terminal_type == 'bash' or terminal_type == 'auto':
                if shutil.which('bash'):
                    return ['bash', '-c', command]
                else:
                    return ['sh', '-c', command]
            elif terminal_type == 'zsh' and shutil.which('zsh'):
                return ['zsh', '-c', command]
            else:
                return ['sh', '-c', command]

        return None

    def decode_terminal_output(self, stdout: bytes, stderr: bytes) -> str:
        """Decode terminal output with proper encoding handling"""
        output = ''

        # Try to decode stdout
        if stdout:
            for encoding in ['utf-8', 'cp1252', 'latin-1']:
                try:
                    output = stdout.decode(encoding, errors='replace')
                    break
                except:
                    continue

        # Add stderr if present
        if stderr:
            error_output = ''
            for encoding in ['utf-8', 'cp1252', 'latin-1']:
                try:
                    error_output = stderr.decode(encoding, errors='replace')
                    break
                except:
                    continue

            if error_output:
                if output:
                    output += '\n' + error_output
                else:
                    output = error_output

        return output

    async def open_in_vscode(self, websocket, data: Dict[str, Any]):
        """Open script in VS Code or default editor"""
        script_id = data['scriptId']
        content = data['content']

        # Save to file
        script_name = self.script_cache.get(script_id, {}).get('name', f"{script_id}.py")
        script_path = SCRIPTS_DIR / script_name

        try:
            if aiofiles:
                async with aiofiles.open(script_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
            else:
                script_path.write_text(content, encoding='utf-8')
        except Exception as e:
            await websocket.send_json({
                'type': 'error',
                'message': f'Failed to save script: {str(e)}'
            })
            return

        # Try to open in editor
        opened = False

        # Try VS Code first
        if shutil.which('code'):
            try:
                subprocess.Popen(['code', str(script_path)])
                await websocket.send_json({
                    'type': 'info',
                    'message': 'Opened in VS Code'
                })
                opened = True
            except:
                pass

        # Try other editors
        if not opened:
            editors = ['subl', 'atom', 'notepad++', 'gedit', 'kate']
            for editor in editors:
                if shutil.which(editor):
                    try:
                        subprocess.Popen([editor, str(script_path)])
                        await websocket.send_json({
                            'type': 'info',
                            'message': f'Opened in {editor}'
                        })
                        opened = True
                        break
                    except:
                        pass

        # Use system default
        if not opened:
            try:
                if platform.system() == 'Darwin':
                    subprocess.Popen(['open', str(script_path)])
                elif platform.system() == 'Windows':
                    os.startfile(str(script_path))
                else:
                    subprocess.Popen(['xdg-open', str(script_path)])

                await websocket.send_json({
                    'type': 'info',
                    'message': 'Opened in default editor'
                })
            except Exception as e:
                await websocket.send_json({
                    'type': 'error',
                    'message': f'Could not open editor: {str(e)}'
                })

    async def ai_analyze(self, websocket, data: Dict[str, Any]):
        """Analyze code with AI for comprehensive improvements"""
        script_id = data['scriptId']
        content = data['content']
        api_key = data['apiKey']

        if not HAS_OPENAI or not api_key:
            await websocket.send_json({
                'type': 'error',
                'message': 'OpenAI API key required'
            })
            return

        # First do local analysis
        analysis = await self.analyze_script_content(content)

        prompt = f"""Analyze this Python code comprehensively and provide detailed suggestions:

```python
{content}
```

Current Analysis Results:
{self.format_analysis_for_prompt(analysis)}

Please provide:
1. Code quality improvements
2. Performance optimizations
3. Best practices violations
4. Potential bugs or issues
5. Security concerns
6. Suggested refactoring
7. Missing error handling
8. Documentation improvements

Format your response as a structured list with clear sections."""

        try:
            client = OpenAI(api_key=api_key)

            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
            )

            suggestion = response.choices[0].message.content

            await websocket.send_json({
                'type': 'aiSuggestion',
                'scriptId': script_id,
                'suggestion': suggestion
            })

        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            error_msg = self.format_openai_error(str(e))

            await websocket.send_json({
                'type': 'error',
                'message': f'AI analysis failed: {error_msg}'
            })

    async def copilot_suggest(self, websocket, data: Dict[str, Any]):
        """Get GitHub Copilot suggestions (placeholder for future implementation)"""
        script_id = data['scriptId']
        content = data['content']
        cursor_position = data.get('cursorPosition', 0)

        # For now, provide a helpful message about Copilot integration
        await websocket.send_json({
            'type': 'copilotSuggestion',
            'scriptId': script_id,
            'suggestion': """GitHub Copilot integration requires:
1. GitHub Copilot subscription
2. VS Code with Copilot extension
3. Or GitHub CLI with Copilot enabled

For now, you can:
- Use the "Open in VS Code" button to edit with Copilot
- Use the AI Analysis feature for code suggestions
- Enable Auto-Heal for automatic error fixing"""
        })

    async def update_installed_packages(self):
        """Update list of installed packages"""
        try:
            cmd = self.get_pip_command() + ["list", "--format=json"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                try:
                    packages = json.loads(stdout.decode())
                    self.installed_packages = {pkg['name'].lower() for pkg in packages}
                except:
                    # Fallback to parsing non-JSON format
                    await self.update_installed_packages_fallback()
            else:
                await self.update_installed_packages_fallback()

        except Exception as e:
            logger.error(f"Failed to update package list: {e}")

    async def update_installed_packages_fallback(self):
        """Fallback method to update installed packages"""
        try:
            cmd = self.get_pip_command() + ["list"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                lines = stdout.decode().split('\n')[2:]  # Skip header
                self.installed_packages = set()
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            self.installed_packages.add(parts[0].lower())

        except Exception as e:
            logger.error(f"Failed to update package list (fallback): {e}")

    async def read_script_content(self, script_id: str) -> str:
        """Read script content from file"""
        # Check cache first
        if script_id in self.script_cache:
            return self.script_cache[script_id].get('content', '')

        # Try to read from file
        script_path = TEMP_DIR / f"{script_id}.py"
        if script_path.exists():
            try:
                if aiofiles:
                    async with aiofiles.open(script_path, 'r', encoding='utf-8') as f:
                        return await f.read()
                else:
                    return script_path.read_text(encoding='utf-8')
            except:
                pass

        return ""

    async def create_project(self, websocket, data: Dict[str, Any]):
        """Create a new project"""
        project_name = data['projectName']
        project_id = f"project_{int(time.time() * 1000)}_{hashlib.md5(project_name.encode()).hexdigest()[:8]}"

        project = Project(
            id=project_id,
            name=project_name
        )

        self.projects[project_id] = project

        # Create project directory
        project_dir = PROJECTS_DIR / f"project_{project_id}"
        project_dir.mkdir(exist_ok=True)

        # Create __init__.py for proper Python package
        init_file = project_dir / "__init__.py"
        try:
            if aiofiles:
                async with aiofiles.open(init_file, 'w', encoding='utf-8') as f:
                    await f.write("# Project package initialization\n")
            else:
                init_file.write_text("# Project package initialization\n", encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to create __init__.py: {e}")

        await websocket.send_json({
            'type': 'projectCreated',
            'projectId': project_id,
            'projectName': project_name
        })

        logger.info(f"Created project: {project_name} ({project_id})")

    async def add_script_to_project(self, websocket, data: Dict[str, Any]):
        """Add a script to a project"""
        project_id = data['projectId']
        script_id = data['scriptId']
        script_name = data['scriptName']
        content = data['content']
        is_main = data.get('isMain', False)

        if project_id not in self.projects:
            await websocket.send_json({
                'type': 'error',
                'message': 'Project not found'
            })
            return

        project = self.projects[project_id]

        # Sanitize script name for Python module
        if not script_name.endswith('.py'):
            script_name += '.py'

        # Ensure valid Python module name
        module_name = script_name[:-3]  # Remove .py
        module_name = re.sub(r'[^a-zA-Z0-9_]', '_', module_name)
        if module_name and module_name[0].isdigit():
            module_name = f"m_{module_name}"
        script_name = f"{module_name}.py"

        # Add to project
        project.scripts[script_id] = {
            'id': script_id,
            'name': script_name,
            'content': content,
            'is_main': is_main
        }

        if is_main:
            project.main_script = script_id

        # Save script to project directory
        project_dir = PROJECTS_DIR / f"project_{project_id}"
        script_path = project_dir / script_name

        try:
            if aiofiles:
                async with aiofiles.open(script_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
            else:
                script_path.write_text(content, encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to save script to project: {e}")

        # Update project dependencies
        analysis = await self.analyze_script_content(content)
        for imp in analysis.imports:
            base_module = imp.split('.')[0]
            if base_module not in sys.stdlib_module_names:
                project.dependencies.add(base_module)

        await websocket.send_json({
            'type': 'scriptAddedToProject',
            'projectId': project_id,
            'scriptId': script_id,
            'scriptName': script_name
        })

    async def save_project_scripts(self, project_id: str):
        """Save all scripts in a project to disk"""
        if project_id not in self.projects:
            logger.warning(f"Project {project_id} not found")
            return

        project = self.projects[project_id]
        project_dir = PROJECTS_DIR / f"project_{project_id}"

        for script_id, script_info in project.scripts.items():
            script_path = project_dir / script_info['name']
            try:
                if aiofiles:
                    async with aiofiles.open(script_path, 'w', encoding='utf-8') as f:
                        await f.write(script_info['content'])
                else:
                    script_path.write_text(script_info['content'], encoding='utf-8')
            except Exception as e:
                logger.error(f"Failed to save script {script_info['name']}: {e}")

    async def run_project(self, websocket, data: Dict[str, Any]):
        """Run the main script of a project"""
        project_id = data['projectId']

        if project_id not in self.projects:
            await websocket.send_json({
                'type': 'error',
                'message': 'Project not found'
            })
            return

        project = self.projects[project_id]
        main_script_id = project.main_script

        if not main_script_id or main_script_id not in project.scripts:
            await websocket.send_json({
                'type': 'error',
                'message': 'No main script defined for project'
            })
            return

        # Save all project scripts first
        await self.save_project_scripts(project_id)

        # Install project dependencies if needed
        missing_deps = []
        await self.update_installed_packages()

        for dep in project.dependencies:
            if dep.lower() not in self.installed_packages:
                missing_deps.append(dep)

        if missing_deps:
            await websocket.send_json({
                'type': 'info',
                'message': f'Installing missing dependencies: {", ".join(missing_deps)}'
            })
            await self.install_dependencies(websocket, {'packages': missing_deps})

        # Run the main script
        main_script = project.scripts[main_script_id]
        await self.run_script(websocket, {
            'scriptId': main_script_id,
            'content': main_script['content'],
            'scriptName': main_script['name'],
            'projectId': project_id,
            'apiKey': data.get('apiKey')
        })

    async def save_all_scripts(self, websocket, data: Dict[str, Any]):
        """Save all scripts from the client"""
        scripts = data.get('scripts', {})

        for script_id, script_data in scripts.items():
            # Save to cache
            self.script_cache[script_id] = {
                'content': script_data.get('content', ''),
                'name': script_data.get('name', f'{script_id}.py'),
                'project_id': script_data.get('projectId')
            }

            # Save to disk
            script_path = TEMP_DIR / f"{script_id}.py"
            try:
                if aiofiles:
                    async with aiofiles.open(script_path, 'w', encoding='utf-8') as f:
                        await f.write(script_data.get('content', ''))
                else:
                    script_path.write_text(script_data.get('content', ''), encoding='utf-8')
            except Exception as e:
                logger.error(f"Failed to save script {script_id}: {e}")

        await websocket.send_json({
            'type': 'info',
            'message': f'Saved {len(scripts)} scripts'
        })

    async def export_project(self, websocket, data: Dict[str, Any]):
        """Export a project as a zip file"""
        project_id = data['projectId']

        if project_id not in self.projects:
            await websocket.send_json({
                'type': 'error',
                'message': 'Project not found'
            })
            return

        # Implementation would create a zip file of the project
        # For now, just acknowledge
        await websocket.send_json({
            'type': 'info',
            'message': 'Project export feature coming soon!'
        })

    async def import_project(self, websocket, data: Dict[str, Any]):
        """Import a project from a zip file"""
        # Implementation would extract and load a project
        # For now, just acknowledge
        await websocket.send_json({
            'type': 'info',
            'message': 'Project import feature coming soon!'
        })

    async def get_script_graph(self, websocket, data: Dict[str, Any]):
        """Get script dependency graph data"""
        # This would analyze imports and create a graph structure
        # For now, return basic structure
        graph_data = {
            'nodes': [],
            'edges': []
        }

        # Add nodes for each script
        for script_id, script_info in self.script_cache.items():
            graph_data['nodes'].append({
                'id': script_id,
                'label': script_info.get('name', script_id),
                'type': 'script'
            })

        # Add nodes for projects
        for project_id, project in self.projects.items():
            graph_data['nodes'].append({
                'id': project_id,
                'label': project.name,
                'type': 'project'
            })

            # Add edges from project to its scripts
            for script_id in project.scripts:
                graph_data['edges'].append({
                    'from': project_id,
                    'to': script_id
                })

        await websocket.send_json({
            'type': 'scriptGraph',
            'graph': graph_data
        })

def install_requirements():
    """Install required packages"""
    required = [
        'fastapi',
        'uvicorn[standard]',
        'websockets',
        'aiofiles',
        'psutil'
    ]

    optional = [
        'openai',
        'requests'
    ]

    print("Checking required packages...")

    for package in required:
        try:
            __import__(package.split('[')[0])
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    print("\nOptional packages (for enhanced features):")
    for package in optional:
        try:
            mod = __import__(package)
            if package == 'openai':
                version = getattr(mod, '__version__', 'unknown')
                print(f"✓ {package} is installed (version {version})")
                if version.startswith('0.'):
                    print(f"  ⚠️  Old OpenAI API detected. Run: pip install --upgrade openai")
            else:
                print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed (optional)")
            response = input(f"Install {package}? (y/n): ")
            if response.lower() == 'y':
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])

async def main(port: int = 8765):
    """Enhanced main entry point with performance and security features"""
    if uvicorn is None:
        print("Error: uvicorn is required. Install with: pip install uvicorn")
        return

    try:
        server = NeuralNexusServer(port=port)
    except ImportError as e:
        print(f"Error: {e}")
        print("\nRun with --install flag to install requirements:")
        print(f"  python {sys.argv[0]} --install")
        return

    # Create static directory
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    # Save HTML file if it doesn't exist
    html_path = static_dir / "index.html"
    if not html_path.exists():
        print("Creating enhanced index.html...")
        html_path.write_text(server.get_complete_html(), encoding='utf-8')

    # Enhanced uvicorn configuration
    config = uvicorn.Config(
        app=server.app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
        ws_max_size=16384 * 1024,  # 16MB max message size
        # Enhanced performance settings
        loop="uvloop" if HAS_UVLOOP else "asyncio",
        http="httptools",
        ws="websockets",
        access_log=True,
        use_colors=True
    )

    server_instance = uvicorn.Server(config)

    # Enhanced startup banner
    print(f"""
╔══════════════════════════════════════════════════╗
║          Neural Nexus IDE Server v6.0            ║
║        🚀 Enhanced with Security & Performance 🚀 ║
╠══════════════════════════════════════════════════╣
║  Web Interface: http://localhost:{port:<5}            ║
║  WebSocket: ws://localhost:{port:<5}/ws               ║
║  Health Check: http://localhost:{port:<5}/health      ║
║  API Docs: http://localhost:{port:<5}/docs            ║
║                                                  ║
║  🆕 Enhanced Features v6.0:                      ║
║  {'✓' if HAS_ORJSON else '✗'} Ultra-fast JSON (orjson) - 6x faster         ║
║  {'✓' if HAS_UVLOOP else '✗'} Enhanced event loop (uvloop) - 4x faster     ║
║  {'✓' if HAS_SEMGREP else '✗'} Security scanning (Semgrep)                 ║
║  {'✓' if HAS_BANDIT else '✗'} Vulnerability detection (Bandit)             ║
║  {'✓' if HAS_RUFF else '✗'} Auto-formatting & linting (Ruff)              ║
║  {'✓' if HAS_LOGURU else '✗'} Structured logging (loguru)                  ║
║  {'✓' if HAS_SLOWAPI else '✗'} Rate limiting & security headers            ║
║  ✓ Real-time performance monitoring             ║
║                                                  ║
║  🔒 Security Features:                           ║
║  • Content Security Policy (CSP) headers        ║
║  • COOP/COEP protection against Spectre         ║
║  • Rate limiting on API endpoints               ║
║  • Comprehensive vulnerability scanning         ║
║  • Safe code execution environment              ║
║                                                  ║
║  🎨 IDE Features:                                ║
║  • One-click code formatting                    ║
║  • Real-time security analysis                  ║
║  • Multi-script project management              ║
║  • Continuous auto-healing with AI              ║
║  • Terminal integration                         ║
║  • VS Code integration                          ║
║                                                  ║
║  Open http://localhost:{port:<5} in your browser!     ║
╚══════════════════════════════════════════════════╝
    """)

    # Log feature availability
    features_available = []
    features_missing = []

    feature_checks = [
        ("orjson (6x faster JSON)", HAS_ORJSON),
        ("uvloop (4x faster event loop)", HAS_UVLOOP),
        ("loguru (structured logging)", HAS_LOGURU),
        ("slowapi (rate limiting)", HAS_SLOWAPI),
        ("semgrep (security scanning)", HAS_SEMGREP),
        ("bandit (vulnerability detection)", HAS_BANDIT),
        ("ruff (fast linting & formatting)", HAS_RUFF),
        ("msgspec (fast serialization)", HAS_MSGSPEC),
    ]

    for feature_name, is_available in feature_checks:
        if is_available:
            features_available.append(feature_name)
        else:
            features_missing.append(feature_name)

    if features_available:
        logger.info(f"✅ Enhanced features available: {', '.join(features_available)}")

    if features_missing:
        logger.warning(f"⚠️  Missing optional features: {', '.join(features_missing)}")
        logger.info("💡 Install with: pip install orjson uvloop loguru slowapi semgrep bandit ruff msgspec")

    await server_instance.serve()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neural Nexus IDE Server")
    parser.add_argument("--install", action="store_true", help="Install requirements")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--setup", action="store_true", help="Run initial setup")
    args = parser.parse_args()

    if args.install:
        install_requirements()
    elif args.setup:
        # Create all necessary directories and files
        print("Setting up Neural Nexus IDE...")
        for dir_path in [HOME_DIR, SCRIPTS_DIR, LOGS_DIR, TEMP_DIR, PROJECTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created {dir_path}")

        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)
        print(f"✓ Created {static_dir}")

        print("\nSetup complete! Run the server with:")
        print(f"  python {sys.argv[0]}")
    else:
        try:
            asyncio.run(main(port=args.port))
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
