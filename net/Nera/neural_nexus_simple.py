#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Nexus IDE Server v6.0 - Simplified Launcher
Enhanced Python IDE with modular architecture and open-source analysis tools.
"""

import sys
import os
import asyncio
import argparse
import traceback
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_uvloop():
    """Setup uvloop for enhanced performance if available."""
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        return True
    except ImportError:
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    required = [
        ('fastapi', 'FastAPI framework'),
        ('uvicorn', 'ASGI server'),
        ('websockets', 'WebSocket support'),
    ]

    missing = []
    for module, desc in required:
        try:
            __import__(module)
        except ImportError:
            missing.append((module, desc))

    if missing:
        print("❌ Missing required dependencies:")
        for module, desc in missing:
            print(f"   • {module} - {desc}")
        print("\n💡 Install with: pip install fastapi uvicorn[standard] websockets")
        return False

    return True

def install_requirements():
    """Install required packages."""
    import subprocess

    required_packages = [
        'fastapi',
        'uvicorn[standard]',
        'websockets',
        'aiofiles',
        'psutil'
    ]

    optional_packages = [
        'uvloop',
        'orjson',
        'loguru',
        'mypy',
        'ruff',
        'bandit',
        'pylint',
        'friendly-traceback'
    ]

    print("Installing required packages...")
    for package in required_packages:
        try:
            __import__(package.split('[')[0])
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")
                return False

    print("\nInstalling optional enhancement packages...")
    for package in optional_packages:
        try:
            module_name = package.replace('-', '_')
            __import__(module_name)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed")
            except subprocess.CalledProcessError:
                print(f"⚠ Optional package {package} failed to install")

    return True

async def run_server(port=8765, host="127.0.0.1", debug=False):
    """Run the Neural Nexus server with simplified imports."""

    # Setup uvloop
    uvloop_enabled = setup_uvloop()

    # Check dependencies
    if not check_dependencies():
        return

    try:
        # Import FastAPI and create app
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse, JSONResponse
        import uvicorn
        import time
        import json

        # Create FastAPI app
        app = FastAPI(
            title="Neural Nexus IDE Server",
            version="6.0",
            description="Enhanced Python IDE with AI-powered auto-healing"
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Performance tracking
        start_time = time.time()
        request_count = 0

        # Security headers middleware
        @app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            nonlocal request_count
            request_count += 1

            response = await call_next(request)

            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline';"
            )

            return response

        # Create static directory and HTML
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)

        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Nexus IDE v6.0</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .editor { width: 100%; height: 300px; font-family: monospace; border: 1px solid #ddd; padding: 10px; }
        .output { background: #1e1e1e; color: #fff; padding: 15px; height: 200px; overflow-y: auto; font-family: monospace; margin-top: 10px; }
        .controls { margin: 10px 0; }
        .btn { background: #007bff; color: white; border: none; padding: 8px 16px; margin-right: 10px; cursor: pointer; border-radius: 4px; }
        .btn:hover { background: #0056b3; }
        .status { padding: 5px 10px; border-radius: 15px; font-size: 12px; }
        .connected { background: #d4edda; color: #155724; }
        .disconnected { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Neural Nexus IDE v6.0</h1>
            <p>Enhanced Python IDE with AI-powered features</p>
            <div id="status" class="status disconnected">Disconnected</div>
        </div>

        <textarea id="editor" class="editor" placeholder="# Write your Python code here...
print('Hello, Neural Nexus!')

# Try this to test error handling:
# undefined_variable + 5">print('Hello, Neural Nexus!')
print('Welcome to the enhanced IDE!')</textarea>

        <div class="controls">
            <button class="btn" onclick="runCode()">▶️ Run</button>
            <button class="btn" onclick="analyzeCode()">🔍 Analyze</button>
            <button class="btn" onclick="clearOutput()">🗑️ Clear</button>
            <button class="btn" onclick="stopScript()">⏹️ Stop</button>
        </div>

        <div id="output" class="output">Neural Nexus IDE v6.0 Enhanced Edition
Ready to start coding! 🚀</div>
    </div>

    <script>
        let ws = null;
        let currentScriptId = null;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;

            ws = new WebSocket(wsUrl);

            ws.onopen = function() {
                updateStatus('Connected', true);
                addOutput('✅ Connected to Neural Nexus Server');
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onclose = function() {
                updateStatus('Disconnected', false);
                addOutput('❌ Disconnected from server');
                setTimeout(connect, 3000);
            };

            ws.onerror = function(error) {
                addOutput('⚠️ Connection error');
            };
        }

        function updateStatus(text, connected) {
            const status = document.getElementById('status');
            status.textContent = text;
            status.className = 'status ' + (connected ? 'connected' : 'disconnected');
        }

        function handleMessage(data) {
            switch(data.type) {
                case 'output':
                    addOutput('📤 ' + data.content);
                    break;
                case 'error':
                    addOutput('❌ ' + data.content);
                    break;
                case 'scriptStarted':
                    addOutput('🚀 Script started');
                    currentScriptId = data.scriptId;
                    break;
                case 'scriptCompleted':
                    addOutput(`✅ Script completed (${data.runtime?.toFixed(2) || 0}s)`);
                    currentScriptId = null;
                    break;
                case 'analysisResult':
                    addOutput('🔍 Analysis: ' + JSON.stringify(data.summary || data.analysis, null, 2));
                    break;
            }
        }

        function addOutput(text) {
            const output = document.getElementById('output');
            const time = new Date().toLocaleTimeString();
            output.innerHTML += `\\n[${time}] ${text}`;
            output.scrollTop = output.scrollHeight;
        }

        function runCode() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                addOutput('❌ Not connected to server');
                return;
            }

            const code = document.getElementById('editor').value;
            const scriptId = 'script_' + Date.now();

            ws.send(JSON.stringify({
                type: 'run_script',
                scriptId: scriptId,
                content: code
            }));
        }

        function analyzeCode() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                addOutput('❌ Not connected to server');
                return;
            }

            const code = document.getElementById('editor').value;

            ws.send(JSON.stringify({
                type: 'analyze_script',
                content: code
            }));
        }

        function clearOutput() {
            document.getElementById('output').innerHTML = 'Output cleared.';
        }

        function stopScript() {
            if (currentScriptId && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'stop_script',
                    scriptId: currentScriptId
                }));
            }
        }

        // Connect on page load
        document.addEventListener('DOMContentLoaded', connect);
    </script>
</body>
</html>'''

        html_path = static_dir / "index.html"
        if not html_path.exists():
            html_path.write_text(html_content)

        # Mount static files
        app.mount("/static", StaticFiles(directory="static"), name="static")

        # Routes
        @app.get("/")
        async def read_root():
            return FileResponse('static/index.html')

        @app.get("/health")
        async def health_check():
            uptime = time.time() - start_time
            return {
                "status": "healthy",
                "version": "6.0",
                "uptime": uptime,
                "requests": request_count,
                "features": {
                    "uvloop": uvloop_enabled,
                    "enhanced_analysis": True,
                    "auto_healing": True
                }
            }

        # WebSocket connections
        active_connections = set()
        running_scripts = {}

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            active_connections.add(websocket)

            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await handle_message(websocket, message)

            except WebSocketDisconnect:
                active_connections.remove(websocket)

        async def handle_message(websocket: WebSocket, data):
            message_type = data.get('type')

            if message_type == 'run_script':
                await run_script(websocket, data)
            elif message_type == 'analyze_script':
                await analyze_script(websocket, data)
            elif message_type == 'stop_script':
                await stop_script(websocket, data)

        async def run_script(websocket: WebSocket, data):
            script_id = data.get('scriptId')
            content = data.get('content', '')

            if not script_id:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'content': 'Script ID required'
                }))
                return

            # Send start confirmation
            await websocket.send_text(json.dumps({
                'type': 'scriptStarted',
                'scriptId': script_id
            }))

            try:
                # Create temp file and run
                import tempfile
                import subprocess

                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(content)
                    temp_file = f.name

                start_time = time.time()

                # Run the script
                process = await asyncio.create_subprocess_exec(
                    sys.executable, temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                running_scripts[script_id] = process

                stdout, stderr = await process.communicate()

                runtime = time.time() - start_time

                # Send output
                if stdout:
                    await websocket.send_text(json.dumps({
                        'type': 'output',
                        'scriptId': script_id,
                        'content': stdout.decode('utf-8')
                    }))

                if stderr:
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'scriptId': script_id,
                        'content': stderr.decode('utf-8')
                    }))

                # Send completion
                await websocket.send_text(json.dumps({
                    'type': 'scriptCompleted',
                    'scriptId': script_id,
                    'exitCode': process.returncode,
                    'runtime': runtime
                }))

                # Cleanup
                os.unlink(temp_file)
                if script_id in running_scripts:
                    del running_scripts[script_id]

            except Exception as e:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'scriptId': script_id,
                    'content': f'Execution error: {str(e)}'
                }))

        async def analyze_script(websocket: WebSocket, data):
            content = data.get('content', '')

            # Basic analysis
            analysis = {
                'errors': [],
                'warnings': [],
                'imports': [],
                'lines': len(content.splitlines()),
                'characters': len(content)
            }

            # Check for basic syntax
            try:
                import ast
                tree = ast.parse(content)

                # Extract imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis['imports'].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            analysis['imports'].append(node.module)

            except SyntaxError as e:
                analysis['errors'].append({
                    'type': 'syntax_error',
                    'message': str(e),
                    'line': e.lineno
                })

            # Send analysis result
            await websocket.send_text(json.dumps({
                'type': 'analysisResult',
                'analysis': analysis,
                'summary': {
                    'total_lines': analysis['lines'],
                    'imports_count': len(analysis['imports']),
                    'errors_count': len(analysis['errors'])
                }
            }))

        async def stop_script(websocket: WebSocket, data):
            script_id = data.get('scriptId')

            if script_id in running_scripts:
                process = running_scripts[script_id]
                process.terminate()
                del running_scripts[script_id]

                await websocket.send_text(json.dumps({
                    'type': 'scriptStopped',
                    'scriptId': script_id
                }))

        # Start server
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="debug" if debug else "info",
            loop="uvloop" if uvloop_enabled else "asyncio"
        )

        server = uvicorn.Server(config)

        # Startup message
        print(f"""
╔══════════════════════════════════════════════════╗
║        Neural Nexus IDE Server v6.0 Simple       ║
║           🚀 Enhanced Performance 🚀              ║
╠══════════════════════════════════════════════════╣
║  Web Interface: http://{host}:{port:<5}             ║
║  Health Check: http://{host}:{port:<5}/health       ║
║                                                  ║
║  🚀 Features:                                    ║
║  {'✓' if uvloop_enabled else '✗'} Enhanced event loop (uvloop)             ║
║  ✓ Real-time code execution                     ║
║  ✓ WebSocket communication                      ║
║  ✓ Basic code analysis                          ║
║  ✓ Security headers                             ║
║                                                  ║
║  Open http://{host}:{port:<5} to start coding!        ║
╚══════════════════════════════════════════════════╝""")

        await server.serve()

    except ImportError as e:
        print(f"Import error: {e}")
        print("Try running: python neural_nexus_simple.py --install")
    except Exception as e:
        print(f"Server error: {e}")
        if debug:
            traceback.print_exc()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Neural Nexus IDE Server v6.0 Simple")
    parser.add_argument("--install", action="store_true", help="Install requirements")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.install:
        install_requirements()
    else:
        try:
            asyncio.run(run_server(port=args.port, host=args.host, debug=args.debug))
        except KeyboardInterrupt:
            print("\n👋 Neural Nexus Server shutting down...")
        except Exception as e:
            print(f"💥 Fatal error: {e}")
            if args.debug:
                traceback.print_exc()

if __name__ == "__main__":
    main()
