#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Nexus IDE Server v6.0 - Modular Edition
Enhanced Python IDE with AI-powered auto-healing and comprehensive code analysis.

This modular version separates concerns into:
- Core: Configuration, logging, models, utilities
- Analysis: Code analysis and auto-healing engines
- Server: FastAPI app and WebSocket handling
"""

import sys
import asyncio
import argparse
import traceback
from pathlib import Path

# Set up uvloop early for maximum performance benefit
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("✓ Enhanced event loop (uvloop) activated")
except ImportError:
    print("○ Using standard asyncio event loop")

# Import core modules
try:
    from core.config import config, HOME_DIR, SCRIPTS_DIR, LOGS_DIR, TEMP_DIR, PROJECTS_DIR
    from core.logger import logger
    from core.models import PerformanceMetrics
    from server.app import create_app
    from server.websocket_handler import websocket_handler
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all dependencies are installed.")
    sys.exit(1)

# Additional imports for server functionality
try:
    import uvicorn
    from fastapi import WebSocket
    HAS_UVICORN = True
except ImportError:
    print("Error: uvicorn is required for running the server")
    print("Install with: pip install uvicorn[standard]")
    HAS_UVICORN = False


class NeuralNexusServer:
    """Main server orchestrator for the modular Neural Nexus IDE."""

    def __init__(self, port: int = 8765):
        self.port = port
        self.app = None
        self.metrics = PerformanceMetrics()

        # Initialize the FastAPI application
        try:
            self.app = create_app()
            self._setup_websocket_routes()
            self._setup_static_files()
            logger.info("Neural Nexus Server initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise

    def _setup_websocket_routes(self):
        """Setup WebSocket routes."""
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication."""
            await websocket_handler.handle_connection(websocket)

    def _setup_static_files(self):
        """Setup static file serving."""
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse

        # Create static directory if it doesn't exist
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)

        # Create enhanced HTML file if it doesn't exist
        html_path = static_dir / "index.html"
        if not html_path.exists():
            self._create_enhanced_html(html_path)

        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

        # Serve index.html at root
        @self.app.get("/", response_class=FileResponse)
        async def read_index():
            return FileResponse('static/index.html')

    def _create_enhanced_html(self, html_path: Path):
        """Create enhanced HTML interface."""
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Nexus IDE v6.0 - Enhanced</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header h1 {
            color: white;
            font-size: 2rem;
            font-weight: 300;
        }

        .header .subtitle {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }

        .main-content {
            flex: 1;
            display: flex;
            padding: 2rem;
            gap: 2rem;
        }

        .sidebar {
            width: 300px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 1.5rem;
            height: fit-content;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .editor-area {
            flex: 1;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .feature-list {
            list-style: none;
        }

        .feature-list li {
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
            color: #333;
        }

        .feature-list li:before {
            content: "✨ ";
            color: #667eea;
        }

        .status-panel {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            margin: 0.5rem 0;
            font-size: 0.9rem;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: transform 0.2s;
        }

        .btn:hover {
            transform: translateY(-2px);
        }

        .code-editor {
            width: 100%;
            height: 400px;
            font-family: 'Consolas', 'Monaco', monospace;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 1rem;
            resize: vertical;
        }

        .output-area {
            background: #1e1e1e;
            color: #d4d4d4;
            border-radius: 6px;
            padding: 1rem;
            margin-top: 1rem;
            height: 200px;
            overflow-y: auto;
            font-family: 'Consolas', 'Monaco', monospace;
        }

        .controls {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
            flex-wrap: wrap;
        }

        .connection-status {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .connected {
            background: #d4edda;
            color: #155724;
        }

        .disconnected {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Neural Nexus IDE v6.0</h1>
        <div class="subtitle">Enhanced Python IDE with AI-powered auto-healing and comprehensive analysis</div>
    </div>

    <div class="main-content">
        <div class="sidebar">
            <h3>🚀 Enhanced Features</h3>
            <ul class="feature-list">
                <li>Ultra-fast JSON processing (orjson)</li>
                <li>Enhanced event loop (uvloop)</li>
                <li>Security scanning (Semgrep, Bandit)</li>
                <li>Auto-formatting (Ruff)</li>
                <li>Type checking (MyPy)</li>
                <li>Advanced linting (Pylint)</li>
                <li>Friendly error messages</li>
                <li>Real-time auto-healing</li>
                <li>Project management</li>
                <li>Performance monitoring</li>
            </ul>

            <div class="status-panel">
                <h4>🔧 System Status</h4>
                <div class="status-item">
                    <span>Connection:</span>
                    <span id="connection-status" class="connection-status disconnected">Disconnected</span>
                </div>
                <div class="status-item">
                    <span>Server Version:</span>
                    <span>v6.0</span>
                </div>
                <div class="status-item">
                    <span>Auto-Heal:</span>
                    <span id="auto-heal-status">Disabled</span>
                </div>
            </div>
        </div>

        <div class="editor-area">
            <h3>💻 Code Editor</h3>
            <textarea id="code-editor" class="code-editor" placeholder="# Write your Python code here...
print('Hello, Neural Nexus!')

# Try some code with errors to test auto-healing
# Example: undefined_variable + 5"></textarea>

            <div class="controls">
                <button class="btn" onclick="runCode()">▶️ Run Code</button>
                <button class="btn" onclick="analyzeCode()">🔍 Analyze</button>
                <button class="btn" onclick="formatCode()">🎨 Format</button>
                <button class="btn" onclick="securityScan()">🔒 Security Scan</button>
                <button class="btn" onclick="toggleAutoHeal()">🚑 Toggle Auto-Heal</button>
                <button class="btn" onclick="stopScript()">⏹️ Stop</button>
            </div>

            <div id="output" class="output-area">
                Neural Nexus IDE v6.0 Enhanced Edition
                =====================================

                🌟 Welcome to the enhanced Python IDE with AI-powered features!

                Features available:
                • Real-time code analysis and auto-healing
                • Security vulnerability scanning
                • Advanced linting and formatting
                • Type checking and error detection
                • Performance monitoring

                Ready to start coding! 🚀
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let currentScriptId = null;
        let autoHealEnabled = false;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;

            ws = new WebSocket(wsUrl);

            ws.onopen = function(event) {
                updateConnectionStatus(true);
                appendOutput('✅ Connected to Neural Nexus Server');
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onclose = function(event) {
                updateConnectionStatus(false);
                appendOutput('❌ Disconnected from server');
                setTimeout(connect, 3000); // Reconnect after 3 seconds
            };

            ws.onerror = function(error) {
                appendOutput('⚠️ Connection error: ' + error);
            };
        }

        function updateConnectionStatus(connected) {
            const statusEl = document.getElementById('connection-status');
            if (connected) {
                statusEl.textContent = 'Connected';
                statusEl.className = 'connection-status connected';
            } else {
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'connection-status disconnected';
            }
        }

        function handleMessage(data) {
            switch(data.type) {
                case 'output':
                    appendOutput('📤 ' + data.content);
                    break;
                case 'error':
                    appendOutput('❌ ' + data.content, 'error');
                    break;
                case 'scriptStarted':
                    appendOutput('🚀 Script started');
                    currentScriptId = data.scriptId;
                    break;
                case 'scriptCompleted':
                    appendOutput(`✅ Script completed (exit code: ${data.exitCode}, runtime: ${data.runtime.toFixed(2)}s)`);
                    currentScriptId = null;
                    break;
                case 'analysisResult':
                    displayAnalysisResult(data.analysis);
                    break;
                case 'autoHealSuccess':
                    appendOutput('🚑 Auto-heal successful! Code has been fixed.');
                    document.getElementById('code-editor').value = data.fixedCode;
                    break;
                case 'autoHealFailed':
                    appendOutput('⚠️ Auto-heal failed: ' + data.message);
                    break;
                case 'formattedCode':
                    if (data.formattedCode !== data.originalCode) {
                        document.getElementById('code-editor').value = data.formattedCode;
                        appendOutput('🎨 Code formatted successfully');
                    } else {
                        appendOutput('ℹ️ Code is already properly formatted');
                    }
                    break;
                case 'securityScanResult':
                    displaySecurityResults(data);
                    break;
                default:
                    console.log('Unknown message type:', data.type);
            }
        }

        function appendOutput(text, type = 'info') {
            const output = document.getElementById('output');
            const timestamp = new Date().toLocaleTimeString();
            output.innerHTML += `[${timestamp}] ${text}\n`;
            output.scrollTop = output.scrollHeight;
        }

        function runCode() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                appendOutput('❌ Not connected to server');
                return;
            }

            const code = document.getElementById('code-editor').value;
            const scriptId = 'script_' + Date.now();

            ws.send(JSON.stringify({
                type: 'run_script',
                scriptId: scriptId,
                content: code,
                scriptName: scriptId + '.py'
            }));
        }

        function analyzeCode() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                appendOutput('❌ Not connected to server');
                return;
            }

            const code = document.getElementById('code-editor').value;
            const scriptId = 'analysis_' + Date.now();

            ws.send(JSON.stringify({
                type: 'analyze_script',
                scriptId: scriptId,
                content: code
            }));
        }

        function formatCode() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                appendOutput('❌ Not connected to server');
                return;
            }

            const code = document.getElementById('code-editor').value;

            ws.send(JSON.stringify({
                type: 'format_code',
                content: code
            }));
        }

        function securityScan() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                appendOutput('❌ Not connected to server');
                return;
            }

            const code = document.getElementById('code-editor').value;

            ws.send(JSON.stringify({
                type: 'security_scan',
                content: code
            }));
        }

        function toggleAutoHeal() {
            autoHealEnabled = !autoHealEnabled;
            const status = document.getElementById('auto-heal-status');
            status.textContent = autoHealEnabled ? 'Enabled' : 'Disabled';

            if (currentScriptId && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'toggle_auto_heal',
                    scriptId: currentScriptId,
                    enabled: autoHealEnabled
                }));
            }

            appendOutput(`🚑 Auto-heal ${autoHealEnabled ? 'enabled' : 'disabled'}`);
        }

        function stopScript() {
            if (currentScriptId && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'stop_script',
                    scriptId: currentScriptId
                }));
            }
        }

        function displayAnalysisResult(analysis) {
            appendOutput('🔍 Analysis completed:');
            appendOutput(`   • Errors: ${analysis.errors.length}`);
            appendOutput(`   • Security issues: ${analysis.security_issues.length}`);
            appendOutput(`   • Quality score: ${analysis.code_quality_score?.toFixed(1) || 'N/A'}`);
            appendOutput(`   • Tools used: ${analysis.tools_used.join(', ')}`);
        }

        function displaySecurityResults(data) {
            appendOutput('🔒 Security scan completed:');
            appendOutput(`   • Issues found: ${data.securityIssues.length}`);
            appendOutput(`   • Security score: ${data.securityScore?.toFixed(1) || 'N/A'}`);

            if (data.securityIssues.length > 0) {
                data.securityIssues.forEach(issue => {
                    appendOutput(`   ⚠️ Line ${issue.line}: ${issue.message}`);
                });
            }
        }

        // Connect on page load
        document.addEventListener('DOMContentLoaded', connect);
    </script>
</body>
</html>'''

        html_path.write_text(html_content, encoding='utf-8')
        logger.info("Enhanced HTML interface created")


def install_requirements():
    """Install required packages with enhanced package list."""
    import subprocess

    required_packages = [
        'fastapi',
        'uvicorn[standard]',
        'websockets',
        'aiofiles',
        'psutil'
    ]

    optional_packages = [
        'uvloop',           # Enhanced event loop
        'orjson',           # Fast JSON
        'loguru',           # Enhanced logging
        'slowapi',          # Rate limiting
        'mypy',             # Type checking
        'pylint',           # Advanced linting
        'ruff',             # Fast linting and formatting
        'bandit',           # Security analysis
        'semgrep',          # Security scanning
        'friendly-traceback', # Enhanced error messages
        'openai',           # AI integration
        'requests'          # HTTP requests
    ]

    print("Installing required packages...")
    for package in required_packages:
        try:
            __import__(package.split('[')[0])
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")

    print("\nInstalling optional enhancement packages...")
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"⚠ Optional package {package} failed to install: {e}")


async def main(port: int = 8765):
    """Enhanced main entry point with comprehensive startup."""
    if not HAS_UVICORN:
        print("Error: uvicorn is required but not available")
        print("Install with: pip install uvicorn[standard]")
        return

    try:
        # Initialize the modular server
        server = NeuralNexusServer(port=port)

        # Enhanced uvicorn configuration
        config_dict = {
            "app": server.app,
            "host": config.host,
            "port": port,
            "log_level": "debug" if config.debug else "info",
            "ws_ping_interval": 20,
            "ws_ping_timeout": 20,
            "ws_max_size": 16384 * 1024,  # 16MB
            "access_log": True,
            "use_colors": True,
        }

        # Add performance enhancements if available
        if config.use_uvloop:
            config_dict["loop"] = "uvloop"

        try:
            config_dict["http"] = "httptools"
            config_dict["ws"] = "websockets"
        except:
            pass

        uvicorn_config = uvicorn.Config(**config_dict)
        server_instance = uvicorn.Server(uvicorn_config)

        # Enhanced startup banner
        feature_status = config.get_feature_status()
        available_features = [k for k, v in feature_status.items() if v == '✓']
        missing_features = [k for k, v in feature_status.items() if v == '✗']

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                Neural Nexus IDE Server v6.0                  ║
║              🚀 Modular Enhanced Edition 🚀                  ║
╠══════════════════════════════════════════════════════════════╣
║  Web Interface: http://{config.host}:{port:<5}                     ║
║  WebSocket: ws://{config.host}:{port:<5}/ws                        ║
║  Health Check: http://{config.host}:{port:<5}/health               ║
║  API Docs: http://{config.host}:{port:<5}/docs                     ║
║                                                              ║
║  🏗️  Modular Architecture:                                   ║
║  • Core: Configuration, logging, models, utilities          ║
║  • Analysis: Code analysis and auto-healing engines         ║
║  • Server: FastAPI app and WebSocket handling               ║
║                                                              ║
║  ✅ Available Features ({len(available_features)}):                           ║""")

        for i, feature in enumerate(available_features):
            if i < 8:  # Limit display
                print(f"║  • {feature:<56} ║")

        if missing_features:
            print(f"║                                                              ║")
            print(f"║  ⚠️  Missing Optional Features ({len(missing_features)}):                    ║")
            for i, feature in enumerate(missing_features[:4]):  # Limit display
                print(f"║  • {feature:<56} ║")

        print(f"""║                                                              ║
║  🔒 Security Features:                                       ║
║  • Content Security Policy headers                          ║
║  • Rate limiting on API endpoints                           ║
║  • Comprehensive vulnerability scanning                     ║
║  • Safe code execution environment                          ║
║                                                              ║
║  🎨 Enhanced IDE Features:                                   ║
║  • Real-time code analysis with multiple tools              ║
║  • Intelligent auto-healing with pattern recognition        ║
║  • Multi-project management system                          ║
║  • Performance monitoring and metrics                       ║
║  • Advanced error interpretation                            ║
║                                                              ║
║  Open http://{config.host}:{port:<5} in your browser to start!        ║
╚══════════════════════════════════════════════════════════════╝""")

        # Log feature availability
        if available_features:
            logger.info(f"✅ Enhanced features available: {', '.join(available_features[:5])}")

        if missing_features:
            logger.warning(f"⚠️  Missing optional features: {', '.join(missing_features[:5])}")
            logger.info("💡 Install missing features with: python neural_nexus_server.py --install")

        # Start the server
        await server_instance.serve()

    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        traceback.print_exc()


def setup_environment():
    """Setup the Neural Nexus environment."""
    print("Setting up Neural Nexus IDE environment...")

    # Create all necessary directories
    directories = [HOME_DIR, SCRIPTS_DIR, LOGS_DIR, TEMP_DIR, PROJECTS_DIR]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

    # Create static directory for web interface
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    print(f"✓ Created directory: {static_dir}")

    print("\n✅ Environment setup complete!")
    print(f"📁 Home directory: {HOME_DIR}")
    print(f"📜 Scripts directory: {SCRIPTS_DIR}")
    print(f"📋 Logs directory: {LOGS_DIR}")
    print(f"🔄 Temp directory: {TEMP_DIR}")
    print(f"📦 Projects directory: {PROJECTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neural Nexus IDE Server v6.0 - Enhanced Modular Edition"
    )
    parser.add_argument("--install", action="store_true",
                      help="Install required and optional packages")
    parser.add_argument("--setup", action="store_true",
                      help="Setup the Neural Nexus environment")
    parser.add_argument("--port", type=int, default=8765,
                      help="Server port (default: 8765)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                      help="Server host (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode")

    args = parser.parse_args()

    # Override config with command line arguments
    if args.host != "127.0.0.1":
        config.host = args.host
    if args.debug:
        config.debug = True

    try:
        if args.install:
            install_requirements()
        elif args.setup:
            setup_environment()
        else:
            # Run the server
            asyncio.run(main(port=args.port))

    except KeyboardInterrupt:
        print("\n👋 Neural Nexus Server shutting down...")
        logger.info("Server shutdown by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        logger.critical(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
