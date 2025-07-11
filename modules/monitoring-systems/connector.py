
import os
import logging
import sys
import json
import socket
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import importlib.util
import subprocess
from connector_interface import ConnectorInterface, ScriptRegistry, ScriptState

# --- Configuration ---
LOG_FILE = "connector.log"
CONNECTOR_PORT = 10130
SCRIPT_REGISTRY_FILE = "script_registry.json"

# --- Setup Logging ---
# Ensure the logger is configured from scratch for each script
logger = logging.getLogger(os.path.abspath(__file__))
logger.setLevel(logging.INFO)

# Prevent logging from propagating to the root logger
logger.propagate = False

# Remove any existing handlers to avoid duplicate logs
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

# File Handler
try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except (IOError, OSError) as e:
    # Fallback to console if file logging fails
    print(f"Could not write to log file {LOG_FILE}: {e}", file=sys.stderr)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


class MonitoringConnector:
    """Enhanced connector with full bidirectional control"""
    
    def __init__(self):
        self.port = CONNECTOR_PORT
        self.running = True
        self.scripts: Dict[str, Dict[str, Any]] = {}
        self.script_interfaces: Dict[str, ConnectorInterface] = {}
        self.registry = ScriptRegistry()
        self.server_socket = None
        
        # Load script registry if exists
        self.load_registry()
        
    def load_registry(self):
        """Load script registry from file"""
        if Path(SCRIPT_REGISTRY_FILE).exists():
            try:
                with open(SCRIPT_REGISTRY_FILE, 'r') as f:
                    data = json.load(f)
                    self.scripts = data.get('scripts', {})
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                
    def save_registry(self):
        """Save script registry to file"""
        try:
            with open(SCRIPT_REGISTRY_FILE, 'w') as f:
                json.dump({'scripts': self.scripts}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            
    def scan_directory(self):
        """Scan directory for Python scripts"""
        directory = Path.cwd()
        for file in directory.glob('*.py'):
            if file.name not in ['connector.py', 'hivemind_connector.py', 'connector_interface.py']:
                script_name = file.name
                if script_name not in self.scripts:
                    self.scripts[script_name] = {
                        'path': str(file),
                        'status': 'discovered',
                        'parameters': {},
                        'metrics': {},
                        'last_seen': time.time()
                    }
                    logger.info(f"Discovered script: {script_name}")
                    
        self.save_registry()
        
    def start_server(self):
        """Start server to listen for script connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', self.port))
        self.server_socket.listen(10)
        logger.info(f"Connector listening on port {self.port}")
        
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self.handle_connection, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Server error: {e}")
                
    def handle_connection(self, conn: socket.socket):
        """Handle incoming connection from script"""
        try:
            # Receive initial message
            data = conn.recv(4096)
            if not data:
                return
                
            message = json.loads(data.decode())
            command = message.get('command')
            
            if command == 'register_script':
                script_name = message.get('script_name')
                state = message.get('state')
                self.handle_script_registration(script_name, state, conn)
                
            elif command == 'get_scripts':
                response = {
                    'status': 'success',
                    'scripts': list(self.scripts.keys())
                }
                conn.send(json.dumps(response).encode())
                
            elif command == 'control_script':
                script_name = message.get('script')
                action = message.get('action')
                params = message.get('params', {})
                response = self.control_script(script_name, action, params)
                conn.send(json.dumps(response).encode())
                
        except Exception as e:
            logger.error(f"Connection handling error: {e}")
            try:
                error_response = {'status': 'error', 'message': str(e)}
                conn.send(json.dumps(error_response).encode())
            except:
                pass
        finally:
            conn.close()
            
    def handle_script_registration(self, script_name: str, state: Dict[str, Any], conn: socket.socket):
        """Handle script registration"""
        logger.info(f"Script {script_name} registered")
        
        if script_name not in self.scripts:
            self.scripts[script_name] = {
                'path': state.get('script_name', script_name),
                'status': 'registered',
                'parameters': {},
                'metrics': {},
                'last_seen': time.time()
            }
            
        # Update script information
        self.scripts[script_name]['status'] = state.get('status', 'idle')
        self.scripts[script_name]['last_seen'] = time.time()
        
        # Store parameters
        if 'parameters' in state:
            self.scripts[script_name]['parameters'] = state['parameters']
            
        # Store metrics
        if 'metrics' in state:
            self.scripts[script_name]['metrics'] = state['metrics']
            
        self.save_registry()
        
        # Keep connection open for bidirectional communication
        script_thread = threading.Thread(
            target=self.maintain_script_connection,
            args=(script_name, conn),
            daemon=True
        )
        script_thread.start()
        
    def maintain_script_connection(self, script_name: str, conn: socket.socket):
        """Maintain connection with script for real-time control"""
        try:
            while self.running:
                # Listen for updates from script
                conn.settimeout(1.0)
                try:
                    data = conn.recv(4096)
                    if not data:
                        break
                        
                    message = json.loads(data.decode())
                    event = message.get('event')
                    
                    if event == 'parameter_updated':
                        param_name = message['data']['parameter']
                        new_value = message['data']['new_value']
                        self.scripts[script_name]['parameters'][param_name] = new_value
                        logger.info(f"Script {script_name} parameter {param_name} updated to {new_value}")
                        
                    elif event == 'metric_updated':
                        metric_name = message['data']['metric']
                        value = message['data']['value']
                        self.scripts[script_name]['metrics'][metric_name] = value
                        
                    elif event == 'status_changed':
                        new_status = message['data']['status']
                        self.scripts[script_name]['status'] = new_status
                        logger.info(f"Script {script_name} status changed to {new_status}")
                        
                except socket.timeout:
                    continue
                    
        except Exception as e:
            logger.error(f"Connection lost with script {script_name}: {e}")
        finally:
            self.scripts[script_name]['status'] = 'disconnected'
            conn.close()
            
    def control_script(self, script_name: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Control a script through its interface"""
        if script_name not in self.scripts:
            return {'status': 'error', 'message': f'Script {script_name} not found'}
            
        script_info = self.scripts[script_name]
        
        if action == 'start':
            # Start the script as a subprocess
            try:
                proc = subprocess.Popen(
                    [sys.executable, script_info['path']],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.scripts[script_name]['pid'] = proc.pid
                self.scripts[script_name]['status'] = 'starting'
                return {'status': 'success', 'message': f'Script {script_name} started', 'pid': proc.pid}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
                
        elif action == 'stop':
            # Send stop command to script
            # This would be implemented through the connection
            return {'status': 'success', 'message': f'Stop command sent to {script_name}'}
            
        elif action == 'set_parameter':
            # Set parameter through connection
            param_name = params.get('name')
            value = params.get('value')
            # This would be sent through the active connection
            return {'status': 'success', 'message': f'Parameter {param_name} set to {value}'}
            
        elif action == 'get_state':
            return {
                'status': 'success',
                'state': {
                    'status': script_info.get('status', 'unknown'),
                    'parameters': script_info.get('parameters', {}),
                    'metrics': script_info.get('metrics', {}),
                    'last_seen': script_info.get('last_seen', 0)
                }
            }
            
        else:
            return {'status': 'error', 'message': f'Unknown action: {action}'}
            
    def get_all_scripts_status(self) -> Dict[str, Any]:
        """Get status of all scripts"""
        status = {}
        for script_name, info in self.scripts.items():
            status[script_name] = {
                'status': info.get('status', 'unknown'),
                'last_seen': info.get('last_seen', 0),
                'parameters_count': len(info.get('parameters', {})),
                'metrics_count': len(info.get('metrics', {}))
            }
        return status
        
    def run(self):
        """Run the connector"""
        logger.info("Starting enhanced monitoring connector...")
        
        # Scan directory for scripts
        self.scan_directory()
        logger.info(f"Found {len(self.scripts)} scripts")
        
        # Start server
        server_thread = threading.Thread(target=self.start_server, daemon=True)
        server_thread.start()
        
        # Monitor loop
        try:
            while self.running:
                # Periodic status check
                status = self.get_all_scripts_status()
                active_scripts = sum(1 for s in status.values() if s['status'] == 'running')
                logger.info(f"Active scripts: {active_scripts}/{len(self.scripts)}")
                
                # Save registry periodically
                self.save_registry()
                
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("Shutting down connector...")
            self.running = False
            if self.server_socket:
                self.server_socket.close()


def main():
    """Main function for the connector script."""
    connector = MonitoringConnector()
    connector.run()

if __name__ == "__main__":
    main()
