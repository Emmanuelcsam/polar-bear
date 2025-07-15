#!/usr/bin/env python3
"""
Hivemind Connector for: modules/real-time-monitoring
Connector ID: 446b9447
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
import shared_config # Import the shared configuration module

# Import all scripts that need to be controlled
import live_fiber_analyzer
import live_monitoring_dashboard
import live_video_processor
import location_tracking_pipeline
import realtime_monitor
import run_calibration
import run_circle_detector
import run_geometry_demo_fixed
import run_geometry_demo
import src.applications.example_application as example_application
import src.applications.realtime_circle_detector as realtime_circle_detector_src
import src.core.integrated_geometry_system as integrated_geometry_system
import src.core.python313_fix as python313_fix
import src.tools.performance_benchmark_tool as performance_benchmark_tool
import src.tools.realtime_calibration_tool as realtime_calibration_tool_src
import src.tools.setup_installer as setup_installer
import src.tools.uv_compatible_setup as uv_compatible_setup

class HivemindConnector:
    def __init__(self):
        self.connector_id = "446b9447"
        self.directory = Path("/home/jarvis/Documents/GitHub/polar-bear/modules/real-time-monitoring")
        self.port = 10038
        self.parent_port = 10004
        self.depth = 2
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {}
        self.scripts_in_directory = {}
        self.running_scripts = {}

        # Dictionary to hold references to all controllable scripts
        self.controllable_scripts = {
            "live_fiber_analyzer": live_fiber_analyzer,
            "live_monitoring_dashboard": live_monitoring_dashboard,
            "live_video_processor": live_video_processor,
            "location_tracking_pipeline": location_tracking_pipeline,
            "realtime_monitor": realtime_monitor,
            "run_calibration": run_calibration,
            "run_circle_detector": run_circle_detector,
            "run_geometry_demo_fixed": run_geometry_demo_fixed,
            "run_geometry_demo": run_geometry_demo,
            "example_application": example_application,
            "realtime_circle_detector_src": realtime_circle_detector_src,
            "integrated_geometry_system": integrated_geometry_system,
            "python313_fix": python313_fix,
            "performance_benchmark_tool": performance_benchmark_tool,
            "realtime_calibration_tool_src": realtime_calibration_tool_src,
            "setup_installer": setup_installer,
            "uv_compatible_setup": uv_compatible_setup,
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"Connector_{self.connector_id}")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def start(self):
        """Start the connector"""
        self.logger.info(f"Starting connector {self.connector_id} on port {self.port}")
        
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
        script_name = message.get('script_name')

        if cmd == 'status':
            return {
                'status': 'active',
                'connector_id': self.connector_id,
                'directory': str(self.directory),
                'depth': self.depth,
                'scripts_loaded': len(self.controllable_scripts),
                'scripts_running': list(self.running_scripts.keys()),
                'children': len(self.child_connectors)
            }
        elif cmd == 'get_all_scripts_status':
            return self._get_all_scripts_status()
        elif cmd == 'get_script_status':
            if script_name:
                return self._get_script_status(script_name)
            return {'error': 'script_name not provided'}
        elif cmd == 'set_script_parameter':
            if script_name and 'key' in message and 'value' in message:
                return self._set_script_parameter(script_name, message['key'], message['value'])
            return {'error': 'script_name, key, or value not provided'}
        elif cmd == 'start_script':
            if script_name:
                return self._start_script(script_name)
            return {'error': 'script_name not provided'}
        elif cmd == 'stop_script':
            if script_name:
                return self._stop_script(script_name)
            return {'error': 'script_name not provided'}
        elif cmd == 'scan':
            self.scan_directory()
            return {'status': 'scan_complete', 'scripts_found': list(self.scripts_in_directory.keys())}
        elif cmd == 'troubleshoot':
            return self.troubleshoot_connections()
        else:
            return {'error': 'Unknown command'}
            
    def _get_all_scripts_status(self):
        """
        Retrieves information from all integrated scripts.
        Returns a dictionary where keys are script names and values are their info.
        """
        all_info = {}
        for name, module in self.controllable_scripts.items():
            try:
                if hasattr(module, 'get_script_info') and callable(module.get_script_info):
                    info = module.get_script_info()
                    all_info[name] = info
                else:
                    all_info[name] = {"status": "not_controllable", "message": "Module does not expose get_script_info"}
            except Exception as e:
                self.logger.error(f"Error getting info from {name}: {e}")
                all_info[name] = {"status": "error", "message": str(e)}
        return all_info

    def _get_script_status(self, script_name: str):
        """
        Retrieves information from a specific integrated script.
        """
        if script_name in self.controllable_scripts:
            module = self.controllable_scripts[script_name]
            try:
                if hasattr(module, 'get_script_info') and callable(module.get_script_info):
                    return module.get_script_info()
                return {"status": "not_controllable", "message": "Module does not expose get_script_info"}
            except Exception as e:
                self.logger.error(f"Error getting info from {script_name}: {e}")
                return {"status": "error", "message": str(e)}
        return {"error": "Script not found"}

    def _set_script_parameter(self, script_name: str, key: str, value: any):
        """
        Sets a parameter for a specific script.
        """
        if script_name in self.controllable_scripts:
            module = self.controllable_scripts[script_name]
            try:
                if hasattr(module, 'set_script_parameter') and callable(module.set_script_parameter):
                    success = module.set_script_parameter(key, value)
                    if success:
                        self.logger.info(f"Successfully set parameter '{key}' to '{value}' for '{script_name}'")
                        return {"status": "success", "script_name": script_name, "key": key, "value": value}
                    else:
                        self.logger.warning(f"Failed to set parameter '{key}' for '{script_name}'. Parameter might not be supported or value is invalid.")
                        return {"status": "failed", "message": "Parameter not supported or invalid value"}
                return {"status": "not_controllable", "message": "Module does not expose set_script_parameter"}
            except Exception as e:
                self.logger.error(f"Error setting parameter for {script_name}: {e}")
                return {"status": "error", "message": str(e)}
        return {"error": "Script not found"}

    def _start_script(self, script_name: str):
        """
        Starts a script in a new thread if it's not already running.
        """
        if script_name not in self.controllable_scripts:
            return {"status": "error", "message": "Script not found"}

        if script_name in self.running_scripts and self.running_scripts[script_name].is_alive():
            return {"status": "info", "message": f"Script {script_name} is already running."}

        module = self.controllable_scripts[script_name]
        if not hasattr(module, 'main') or not callable(module.main):
            return {"status": "error", "message": f"Script {script_name} does not have a callable main function."}

        def script_target():
            try:
                self.logger.info(f"Starting {script_name} in a new thread...")
                module.main() # Call the main function of the script
                self.logger.info(f"Script {script_name} thread finished.")
            except Exception as e:
                self.logger.error(f"Error running script {script_name} in thread: {e}")
            finally:
                # Clean up reference if script finishes on its own
                if script_name in self.running_scripts:
                    del self.running_scripts[script_name]

        thread = threading.Thread(target=script_target, daemon=True)
        thread.start()
        self.running_scripts[script_name] = thread
        self.logger.info(f"Script {script_name} started in background thread.")
        return {"status": "success", "message": f"Script {script_name} started."}

    def _stop_script(self, script_name: str):
        """
        Attempts to stop a running script.
        This relies on the script having a mechanism to stop (e.g., a 'running' flag).
        """
        if script_name not in self.running_scripts or not self.running_scripts[script_name].is_alive():
            return {"status": "info", "message": f"Script {script_name} is not running."}

        module = self.controllable_scripts[script_name]
        stop_successful = False

        # Attempt to stop gracefully by setting a 'running' flag or calling a 'stop' method
        if hasattr(module, 'running_instance') and hasattr(module.running_instance, 'running'):
            module.running_instance.running = False
            self.logger.info(f"Attempted graceful stop of {script_name} via 'running' flag.")
            stop_successful = True
        elif hasattr(module, 'running_instance') and hasattr(module.running_instance, 'stop') and callable(module.running_instance.stop):
            module.running_instance.stop()
            self.logger.info(f"Attempted graceful stop of {script_name} via 'stop()' method.")
            stop_successful = True
        elif hasattr(module, 'stop_script') and callable(module.stop_script):
            module.stop_script()
            self.logger.info(f"Attempted graceful stop of {script_name} via module-level stop_script().")
            stop_successful = True
        else:
            self.logger.warning(f"No graceful stop mechanism found for {script_name}. Thread will continue until completion or external termination.")
            return {"status": "warning", "message": "No graceful stop mechanism found. Manual intervention may be required."}

        # Give it a moment to stop and then check if the thread is still alive
        self.running_scripts[script_name].join(timeout=5) # Wait up to 5 seconds

        if not self.running_scripts[script_name].is_alive():
            del self.running_scripts[script_name]
            self.logger.info(f"Script {script_name} thread successfully terminated.")
            return {"status": "success", "message": f"Script {script_name} stopped."}
        else:
            self.logger.error(f"Script {script_name} thread did not terminate after stop signal.")
            return {"status": "error", "message": f"Script {script_name} did not stop gracefully."}

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
        """Execute a script in the directory (legacy, now calls start_script)"""
        self.logger.warning(f"'execute_script' is deprecated. Use 'start_script' instead.")
        return self._start_script(script_name)
            
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
        print("\nConnector stopped")
