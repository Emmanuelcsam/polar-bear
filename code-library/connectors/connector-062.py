#!/usr/bin/env python3
"""
Hivemind Connector for: modules/iteration9-pytorch-production
Connector ID: 399044f4
Depth Level: 2
"""

import socket
import json
import threading
import time
import os
import sys
import subprocess
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Import enhanced integration
try:
    from enhanced_integration import EnhancedConnector as EnhancedIntegration, EventType, Event
    ENHANCED_INTEGRATION_AVAILABLE = True
except ImportError:
    ENHANCED_INTEGRATION_AVAILABLE = False

class HivemindConnector:
    def __init__(self):
        self.connector_id = "399044f4"
        self.directory = Path("/home/jarvis/Documents/GitHub/polar-bear/modules/iteration9-pytorch-production")
        self.port = 10050
        self.parent_port = 10004
        self.depth = 2
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {}
        self.scripts_in_directory = {}
        self.enhanced_integration = None
        self.event_data = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"Connector_{self.connector_id}")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def start(self):
        """Start the connector"""
        self.logger.info(f"Starting connector {self.connector_id} on port {self.port}")
        
        # Initialize enhanced integration if available
        if ENHANCED_INTEGRATION_AVAILABLE:
            self._initialize_enhanced_integration()
        
        # Start listening for connections
        self.listen_thread = threading.Thread(target=self.listen_for_connections)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        # Connect to parent
        self.connect_to_parent()
        
        # Scan directory
        self.scan_directory()
        
        # Heartbeat loop
        self.heartbeat_loop()
    
    def _initialize_enhanced_integration(self):
        """Initialize enhanced integration system"""
        try:
            self.enhanced_integration = EnhancedIntegration()
            
            # Subscribe to events
            self.enhanced_integration.event_bus.subscribe(
                EventType.STATUS, self._handle_status_event, f'hivemind_{self.connector_id}'
            )
            self.enhanced_integration.event_bus.subscribe(
                EventType.PROGRESS, self._handle_progress_event, f'hivemind_{self.connector_id}'
            )
            self.enhanced_integration.event_bus.subscribe(
                EventType.ERROR, self._handle_error_event, f'hivemind_{self.connector_id}'
            )
            
            # Load scripts with enhanced integration
            for script_name, script_path in self.scripts_in_directory.items():
                if script_name not in ['connector.py', 'hivemind_connector.py', 
                                      'enhanced_integration.py', 'setup.py']:
                    try:
                        self.enhanced_integration.load_script(script_path)
                        self.logger.info(f"Loaded {script_name} with enhanced integration")
                    except Exception as e:
                        self.logger.error(f"Failed to load {script_name}: {e}")
            
            self.logger.info("Enhanced integration initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced integration: {e}")
            self.enhanced_integration = None
    
    def _handle_status_event(self, event: Event):
        """Handle status events"""
        self.event_data[f"{event.source}_status"] = event.data
        self.logger.info(f"Status from {event.source}: {event.data}")
    
    def _handle_progress_event(self, event: Event):
        """Handle progress events"""
        progress = event.data.get('progress', 0) * 100
        message = event.data.get('message', '')
        self.logger.info(f"Progress from {event.source}: {progress:.1f}% - {message}")
        
        # Forward to parent if connected
        if self.parent_socket:
            try:
                forward_msg = {
                    'command': 'forward_event',
                    'connector_id': self.connector_id,
                    'event_type': 'progress',
                    'source': event.source,
                    'data': event.data
                }
                self.parent_socket.send(json.dumps(forward_msg).encode())
            except:
                pass
    
    def _handle_error_event(self, event: Event):
        """Handle error events"""
        self.logger.error(f"Error from {event.source}: {event.data}")
        
    def listen_for_connections(self):
        """Listen for incoming connections"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('localhost', self.port))
        self.socket.listen(5)
        
        while self.running:
            try:
                conn, addr = self.socket.accept()
                threading.Thread(target=self.handle_connection, args=(conn,)).start()
            except:
                pass
                
    def handle_connection(self, conn):
        """Handle incoming connection"""
        try:
            data = conn.recv(4096).decode()
            message = json.loads(data)
            
            response = self.process_message(message)
            conn.send(json.dumps(response).encode())
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
        finally:
            conn.close()
            
    def process_message(self, message):
        """Process incoming message"""
        cmd = message.get('command')
        
        # Try to import script interface for enhanced functionality
        try:
            from script_interface import interface, handle_connector_command
            from script_wrappers import wrappers
            interface_available = True
        except ImportError:
            interface_available = False
        
        if cmd == 'status':
            status = {
                'status': 'active',
                'connector_id': self.connector_id,
                'directory': str(self.directory),
                'depth': self.depth,
                'scripts': len(self.scripts_in_directory),
                'children': len(self.child_connectors),
                'interface_available': interface_available,
                'enhanced_integration_available': ENHANCED_INTEGRATION_AVAILABLE
            }
            if interface_available:
                status['controllable_scripts'] = [s['name'] for s in interface.list_scripts()]
            if self.enhanced_integration:
                status['enhanced_scripts'] = list(self.enhanced_integration.controllers.keys())
                status['shared_state'] = self.enhanced_integration.shared_state
                status['event_data'] = self.event_data
            return status
        elif cmd == 'scan':
            self.scan_directory()
            return {'status': 'scan_complete', 'scripts': list(self.scripts_in_directory.keys())}
        elif cmd == 'execute':
            script = message.get('script')
            if script in self.scripts_in_directory:
                return self.execute_script(script)
            return {'error': 'Script not found'}
        elif cmd == 'troubleshoot':
            return self.troubleshoot_connections()
        elif cmd == 'script_control' and interface_available:
            # Delegate script control to interface
            return handle_connector_command(message)
        elif cmd == 'wrapper_function' and interface_available:
            # Execute wrapper function
            func_name = message.get('function')
            args = message.get('args', {})
            if hasattr(wrappers, func_name):
                func = getattr(wrappers, func_name)
                return func(**args)
            else:
                return {'error': f'Function {func_name} not found'}
        elif cmd == 'enhanced_control' and self.enhanced_integration:
            # Enhanced integration commands
            return self._process_enhanced_command(message)
        else:
            return {'error': 'Unknown command'}
    
    def _process_enhanced_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process enhanced integration commands"""
        action = message.get('action')
        
        try:
            if action == 'call_function':
                script = message.get('script')
                function = message.get('function')
                args = message.get('args', [])
                kwargs = message.get('kwargs', {})
                
                result = self.enhanced_integration.call_function(
                    script, function, *args, **kwargs
                )
                return {'result': result, 'status': 'success'}
            
            elif action == 'set_parameter':
                script = message.get('script')
                parameter = message.get('parameter')
                value = message.get('value')
                
                self.enhanced_integration.set_parameter(script, parameter, value)
                return {'status': 'success', 'message': f'Parameter {parameter} set'}
            
            elif action == 'get_script_info':
                script = message.get('script')
                info = self.enhanced_integration.get_script_info(script)
                return {'info': info, 'status': 'success'}
            
            elif action == 'broadcast_event':
                # Broadcast event to all scripts
                event_type = message.get('event_type', 'info')
                data = message.get('data', {})
                
                try:
                    event_type_enum = EventType(event_type)
                except ValueError:
                    event_type_enum = EventType.INFO
                
                self.enhanced_integration.event_bus.publish(Event(
                    type=event_type_enum,
                    source=f'hivemind_{self.connector_id}',
                    data=data
                ))
                return {'status': 'success', 'message': 'Event broadcast'}
            
            else:
                return {'error': f'Unknown action: {action}'}
                
        except Exception as e:
            self.logger.error(f"Enhanced command error: {e}")
            return {'error': str(e), 'status': 'error'}
            
    def connect_to_parent(self):
        """Connect to parent connector"""
        try:
            self.parent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.parent_socket.connect(('localhost', self.parent_port))
            
            # Register with parent
            register_msg = {
                'command': 'register',
                'connector_id': self.connector_id,
                'port': self.port,
                'directory': str(self.directory)
            }
            self.parent_socket.send(json.dumps(register_msg).encode())
            self.logger.info(f"Connected to parent on port {self.parent_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to parent: {e}")
            
    def scan_directory(self):
        """Scan directory for Python scripts and subdirectories"""
        self.scripts_in_directory.clear()
        
        try:
            for item in self.directory.iterdir():
                if item.is_file() and item.suffix == '.py' and item.name != 'hivemind_connector.py':
                    self.scripts_in_directory[item.name] = str(item)
                elif item.is_dir() and not self.should_skip_directory(item):
                    # Check for child connector
                    child_connector = item / 'hivemind_connector.py'
                    if child_connector.exists():
                        self.child_connectors[item.name] = str(child_connector)
                        
            self.logger.info(f"Found {len(self.scripts_in_directory)} scripts and {len(self.child_connectors)} child connectors")
            
            # Re-initialize enhanced integration with new scripts
            if ENHANCED_INTEGRATION_AVAILABLE and self.enhanced_integration is None:
                self._initialize_enhanced_integration()
        except Exception as e:
            self.logger.error(f"Error scanning directory: {e}")
            
    def should_skip_directory(self, path):
        """Check if directory should be skipped"""
        skip_dirs = {
            'venv', 'env', '.env', '__pycache__', '.git', 
            'node_modules', '.venv', 'virtualenv', '.tox',
            'build', 'dist', '.pytest_cache', '.mypy_cache'
        }
        return path.name.startswith('.') or path.name.lower() in skip_dirs
        
    def execute_script(self, script_name):
        """Execute a script in the directory"""
        if script_name not in self.scripts_in_directory:
            return {'error': 'Script not found'}
            
        try:
            result = subprocess.run(
                [sys.executable, self.scripts_in_directory[script_name]],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                'status': 'executed',
                'script': script_name,
                'returncode': result.returncode,
                'stdout': result.stdout[-1000:],  # Last 1000 chars
                'stderr': result.stderr[-1000:]
            }
        except Exception as e:
            return {'error': f'Execution failed: {str(e)}'}
            
    def troubleshoot_connections(self):
        """Troubleshoot connector connections"""
        issues = []
        
        # Check parent connection
        if not self.parent_socket:
            issues.append("Not connected to parent")
        
        # Check listening socket
        if not self.socket:
            issues.append("Not listening for connections")
            
        # Check child connectors
        for name, path in self.child_connectors.items():
            if not Path(path).exists():
                issues.append(f"Child connector missing: {name}")
                
        return {
            'connector_id': self.connector_id,
            'issues': issues,
            'healthy': len(issues) == 0
        }
        
    def heartbeat_loop(self):
        """Send heartbeat to parent"""
        while self.running:
            try:
                if self.parent_socket:
                    heartbeat = {
                        'command': 'heartbeat',
                        'connector_id': self.connector_id,
                        'timestamp': time.time()
                    }
                    self.parent_socket.send(json.dumps(heartbeat).encode())
            except:
                # Reconnect if connection lost
                self.connect_to_parent()
                
            time.sleep(30)  # Heartbeat every 30 seconds

if __name__ == "__main__":
    connector = HivemindConnector()
    try:
        connector.start()
    except KeyboardInterrupt:
        connector.running = False
        if connector.enhanced_integration:
            connector.enhanced_integration.stop()
        print("\nConnector stopped")
