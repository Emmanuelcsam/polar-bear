#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket Handler for Neural Nexus IDE
Real-time communication with enhanced security and performance.
"""
import asyncio
import json
import time
from typing import Set, Dict, Any, Optional

try:
    from fastapi import WebSocket, WebSocketDisconnect
    HAS_WEBSOCKET = True
except ImportError:
    WebSocket = WebSocketDisconnect = None
    HAS_WEBSOCKET = False

from ..core.config import config
from ..core.logger import logger
from ..core.json_utils import dumps, loads
from ..core.models import ScriptProcess, Project, generate_id
from ..analysis.code_analyzer import analyzer
from ..analysis.auto_healer import auto_healer


class WebSocketManager:
    """Enhanced WebSocket manager with connection pooling and security."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        self.max_connections = 100
        self.message_rate_limit = 100  # messages per minute per connection

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Too many connections")
            return False

        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = {
            'connected_at': time.time(),
            'message_count': 0,
            'last_message_time': time.time()
        }

        logger.info(f"WebSocket connection established. Total: {len(self.active_connections)}")
        return True

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_metadata.pop(websocket, None)
            logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            message_str = dumps(message)
            await websocket.send_text(message_str)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        message_str = dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)

        # Clean up failed connections
        for connection in disconnected:
            self.disconnect(connection)

    def check_rate_limit(self, websocket: WebSocket) -> bool:
        """Check if connection is within rate limits."""
        metadata = self.connection_metadata.get(websocket)
        if not metadata:
            return False

        current_time = time.time()

        # Reset counter if a minute has passed
        if current_time - metadata['last_message_time'] > 60:
            metadata['message_count'] = 0
            metadata['last_message_time'] = current_time

        metadata['message_count'] += 1

        return metadata['message_count'] <= self.message_rate_limit


class WebSocketHandler:
    """Enhanced WebSocket message handler."""

    def __init__(self):
        self.manager = WebSocketManager()
        self.running_scripts: Dict[str, ScriptProcess] = {}
        self.projects: Dict[str, Project] = {}
        self.script_cache: Dict[str, Dict[str, Any]] = {}
        self.auto_heal_mode: Dict[str, bool] = {}
        self.heal_attempts: Dict[str, int] = {}

    async def handle_connection(self, websocket: WebSocket):
        """Handle WebSocket connection lifecycle."""
        if not await self.manager.connect(websocket):
            return

        try:
            while True:
                # Receive message
                try:
                    data = await websocket.receive_text()
                    message = loads(data)
                except json.JSONDecodeError as e:
                    await self.manager.send_personal_message({
                        'type': 'error',
                        'message': f'Invalid JSON: {str(e)}'
                    }, websocket)
                    continue

                # Rate limiting check
                if not self.manager.check_rate_limit(websocket):
                    await self.manager.send_personal_message({
                        'type': 'error',
                        'message': 'Rate limit exceeded'
                    }, websocket)
                    continue

                # Process message
                await self.process_message(websocket, message)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.manager.disconnect(websocket)

    async def process_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Process incoming WebSocket messages."""
        message_type = data.get('type')

        try:
            # Route messages to appropriate handlers
            handlers = {
                'run_script': self.handle_run_script,
                'stop_script': self.handle_stop_script,
                'analyze_script': self.handle_analyze_script,
                'auto_heal': self.handle_auto_heal,
                'toggle_auto_heal': self.handle_toggle_auto_heal,
                'create_project': self.handle_create_project,
                'add_script_to_project': self.handle_add_script_to_project,
                'run_project': self.handle_run_project,
                'format_code': self.handle_format_code,
                'security_scan': self.handle_security_scan,
                'get_suggestions': self.handle_get_suggestions
            }

            handler = handlers.get(message_type)
            if handler:
                await handler(websocket, data)
            else:
                await self.manager.send_personal_message({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }, websocket)

        except Exception as e:
            logger.error(f"Error processing message {message_type}: {e}")
            await self.manager.send_personal_message({
                'type': 'error',
                'message': f'Processing error: {str(e)}'
            }, websocket)

    async def handle_run_script(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle script execution request."""
        script_id = data.get('scriptId')
        content = data.get('content', '')
        script_name = data.get('scriptName', f'{script_id}.py')

        if not script_id:
            await self.manager.send_personal_message({
                'type': 'error',
                'message': 'Script ID is required'
            }, websocket)
            return

        # Cache the script
        self.script_cache[script_id] = {
            'content': content,
            'name': script_name
        }

        # Create and start script process
        try:
            # Save script to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                temp_file = f.name

            # Start process
            process = await asyncio.create_subprocess_exec(
                'python', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            script_process = ScriptProcess(
                process=process,
                script_id=script_id,
                start_time=time.time(),
                script_name=script_name
            )

            self.running_scripts[script_id] = script_process

            # Send start confirmation
            await self.manager.send_personal_message({
                'type': 'scriptStarted',
                'scriptId': script_id
            }, websocket)

            # Monitor output
            await self.monitor_script_output(websocket, script_process)

        except Exception as e:
            logger.error(f"Failed to start script {script_id}: {e}")
            await self.manager.send_personal_message({
                'type': 'error',
                'message': f'Failed to start script: {str(e)}'
            }, websocket)

    async def monitor_script_output(self, websocket: WebSocket, script_process: ScriptProcess):
        """Monitor script output and handle auto-healing."""
        process = script_process.process

        try:
            # Read output streams
            stdout, stderr = await process.communicate()

            stdout_text = stdout.decode('utf-8') if stdout else ''
            stderr_text = stderr.decode('utf-8') if stderr else ''

            script_process.output_lines = stdout_text.splitlines()
            script_process.error_lines = stderr_text.splitlines()
            script_process.exit_code = process.returncode
            script_process.status = 'completed' if process.returncode == 0 else 'failed'

            # Send output
            if stdout_text:
                await self.manager.send_personal_message({
                    'type': 'output',
                    'scriptId': script_process.script_id,
                    'content': stdout_text
                }, websocket)

            if stderr_text:
                await self.manager.send_personal_message({
                    'type': 'error',
                    'scriptId': script_process.script_id,
                    'content': stderr_text
                }, websocket)

            # Handle auto-healing if enabled and there were errors
            if (process.returncode != 0 and
                self.auto_heal_mode.get(script_process.script_id, False)):
                await self.attempt_auto_heal(websocket, script_process, stderr_text)

            # Send completion
            await self.manager.send_personal_message({
                'type': 'scriptCompleted',
                'scriptId': script_process.script_id,
                'exitCode': process.returncode,
                'runtime': time.time() - script_process.start_time
            }, websocket)

        except Exception as e:
            logger.error(f"Error monitoring script output: {e}")
            await self.manager.send_personal_message({
                'type': 'error',
                'scriptId': script_process.script_id,
                'message': f'Monitoring error: {str(e)}'
            }, websocket)
        finally:
            # Cleanup
            if script_process.script_id in self.running_scripts:
                del self.running_scripts[script_process.script_id]

    async def attempt_auto_heal(self, websocket: WebSocket, script_process: ScriptProcess, error_text: str):
        """Attempt to auto-heal script errors."""
        script_id = script_process.script_id

        # Check heal attempt limits
        attempts = self.heal_attempts.get(script_id, 0)
        if attempts >= config.max_heal_attempts:
            await self.manager.send_personal_message({
                'type': 'autoHealFailed',
                'scriptId': script_id,
                'message': 'Maximum heal attempts reached'
            }, websocket)
            return

        self.heal_attempts[script_id] = attempts + 1

        # Get script content
        script_info = self.script_cache.get(script_id)
        if not script_info:
            return

        content = script_info['content']

        # Attempt healing
        try:
            await self.manager.send_personal_message({
                'type': 'autoHealStarted',
                'scriptId': script_id,
                'attempt': attempts + 1
            }, websocket)

            # Analyze and heal
            analysis = await analyzer.analyze_code(content)
            fixed_code = await auto_healer.auto_heal_code(content, error_text, analysis)

            if fixed_code and fixed_code != content:
                # Send fixed code back
                await self.manager.send_personal_message({
                    'type': 'autoHealSuccess',
                    'scriptId': script_id,
                    'fixedCode': fixed_code,
                    'attempt': attempts + 1
                }, websocket)

                # Update cache
                self.script_cache[script_id]['content'] = fixed_code

                # Optionally re-run the script
                await self.manager.send_personal_message({
                    'type': 'info',
                    'message': 'Code healed successfully. Click Run to test the fix.'
                }, websocket)

            else:
                await self.manager.send_personal_message({
                    'type': 'autoHealFailed',
                    'scriptId': script_id,
                    'message': 'Could not automatically fix the error'
                }, websocket)

        except Exception as e:
            logger.error(f"Auto-heal error: {e}")
            await self.manager.send_personal_message({
                'type': 'autoHealFailed',
                'scriptId': script_id,
                'message': f'Heal error: {str(e)}'
            }, websocket)

    async def handle_stop_script(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle script stop request."""
        script_id = data.get('scriptId')

        if script_id in self.running_scripts:
            script_process = self.running_scripts[script_id]
            if script_process.process:
                script_process.process.terminate()
                script_process.status = 'terminated'

            await self.manager.send_personal_message({
                'type': 'scriptStopped',
                'scriptId': script_id
            }, websocket)
        else:
            await self.manager.send_personal_message({
                'type': 'error',
                'message': f'Script {script_id} not found'
            }, websocket)

    async def handle_analyze_script(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle script analysis request."""
        script_id = data.get('scriptId')
        content = data.get('content', '')

        try:
            analysis = await analyzer.analyze_code(content)

            await self.manager.send_personal_message({
                'type': 'analysisResult',
                'scriptId': script_id,
                'analysis': analysis.to_dict()
            }, websocket)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            await self.manager.send_personal_message({
                'type': 'error',
                'message': f'Analysis failed: {str(e)}'
            }, websocket)

    async def handle_auto_heal(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle manual auto-heal request."""
        script_id = data.get('scriptId')
        content = data.get('content', '')
        error_message = data.get('errorMessage', '')

        try:
            analysis = await analyzer.analyze_code(content)
            fixed_code = await auto_healer.auto_heal_code(content, error_message, analysis)

            if fixed_code and fixed_code != content:
                await self.manager.send_personal_message({
                    'type': 'healedCode',
                    'scriptId': script_id,
                    'originalCode': content,
                    'healedCode': fixed_code
                }, websocket)
            else:
                suggestions = auto_healer.generate_fix_suggestions(content, error_message, analysis)
                await self.manager.send_personal_message({
                    'type': 'healSuggestions',
                    'scriptId': script_id,
                    'suggestions': suggestions
                }, websocket)

        except Exception as e:
            logger.error(f"Auto-heal error: {e}")
            await self.manager.send_personal_message({
                'type': 'error',
                'message': f'Auto-heal failed: {str(e)}'
            }, websocket)

    async def handle_toggle_auto_heal(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle auto-heal mode toggle."""
        script_id = data.get('scriptId')
        enabled = data.get('enabled', False)

        self.auto_heal_mode[script_id] = enabled

        await self.manager.send_personal_message({
            'type': 'autoHealToggled',
            'scriptId': script_id,
            'enabled': enabled
        }, websocket)

    async def handle_create_project(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle project creation request."""
        project_name = data.get('projectName')

        if not project_name:
            await self.manager.send_personal_message({
                'type': 'error',
                'message': 'Project name is required'
            }, websocket)
            return

        project_id = generate_id('project')
        project = Project(id=project_id, name=project_name)
        self.projects[project_id] = project

        await self.manager.send_personal_message({
            'type': 'projectCreated',
            'projectId': project_id,
            'projectName': project_name
        }, websocket)

    async def handle_add_script_to_project(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle adding script to project."""
        project_id = data.get('projectId')
        script_id = data.get('scriptId')
        script_name = data.get('scriptName')
        content = data.get('content', '')

        if project_id not in self.projects:
            await self.manager.send_personal_message({
                'type': 'error',
                'message': 'Project not found'
            }, websocket)
            return

        project = self.projects[project_id]
        project.scripts[script_id] = {
            'id': script_id,
            'name': script_name,
            'content': content
        }
        project.update_modified_time()

        await self.manager.send_personal_message({
            'type': 'scriptAddedToProject',
            'projectId': project_id,
            'scriptId': script_id
        }, websocket)

    async def handle_run_project(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle project execution request."""
        project_id = data.get('projectId')

        if project_id not in self.projects:
            await self.manager.send_personal_message({
                'type': 'error',
                'message': 'Project not found'
            }, websocket)
            return

        project = self.projects[project_id]
        main_script = project.main_script

        if not main_script or main_script not in project.scripts:
            await self.manager.send_personal_message({
                'type': 'error',
                'message': 'No main script defined for project'
            }, websocket)
            return

        # Run the main script
        script_info = project.scripts[main_script]
        await self.handle_run_script(websocket, {
            'scriptId': main_script,
            'content': script_info['content'],
            'scriptName': script_info['name']
        })

    async def handle_format_code(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle code formatting request."""
        content = data.get('content', '')

        # This would integrate with Ruff for formatting
        # For now, return original content
        await self.manager.send_personal_message({
            'type': 'formattedCode',
            'originalCode': content,
            'formattedCode': content,  # Would be formatted by Ruff
            'tool': 'ruff' if config.auto_format_enabled else 'none'
        }, websocket)

    async def handle_security_scan(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle security scan request."""
        content = data.get('content', '')

        try:
            analysis = await analyzer.analyze_code(content)

            await self.manager.send_personal_message({
                'type': 'securityScanResult',
                'securityIssues': [issue.__dict__ for issue in analysis.security_issues],
                'securityScore': analysis.security_score,
                'recommendations': []  # Would include security recommendations
            }, websocket)

        except Exception as e:
            logger.error(f"Security scan error: {e}")
            await self.manager.send_personal_message({
                'type': 'error',
                'message': f'Security scan failed: {str(e)}'
            }, websocket)

    async def handle_get_suggestions(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle code suggestions request."""
        content = data.get('content', '')

        try:
            analysis = await analyzer.analyze_code(content)
            suggestions = auto_healer.generate_fix_suggestions(content, '', analysis)

            await self.manager.send_personal_message({
                'type': 'codeSuggestions',
                'suggestions': suggestions,
                'qualityScore': analysis.code_quality_score
            }, websocket)

        except Exception as e:
            logger.error(f"Suggestions error: {e}")
            await self.manager.send_personal_message({
                'type': 'error',
                'message': f'Failed to get suggestions: {str(e)}'
            }, websocket)


# Global WebSocket handler instance
websocket_handler = WebSocketHandler()
