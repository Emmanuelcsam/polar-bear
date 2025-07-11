#!/usr/bin/env python3
"""
Polar Bear System - Main Control Script
Comprehensive Python system with connector scripts throughout the project
"""

import os
import sys
import json
import logging
import socket
import threading
import time
import subprocess
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import queue
import hashlib

# System-wide configuration
SYSTEM_CONFIG = {
    "version": "1.0.0",
    "name": "Polar Bear System",
    "author": "Polar Bear Project",
    "description": "Comprehensive connector and control system",
    "log_level": "INFO",
    "log_file": "polar_bear_system.log",
    "config_file": "polar_bear_config.json",
    "connectors": {
        "root": {"port": 9000, "name": "Root Controller"},
        "ruleset": {"port": 9001, "name": "Ruleset Connector"},
        "modules": {"port": 9002, "name": "Modules Connector"},
        "training": {"port": 9003, "name": "Training Connector"},
        "static": {"port": 9004, "name": "Static Connector"},
        "templates": {"port": 9005, "name": "Templates Connector"}
    },
    "directories_to_exclude": ["venv", "env", ".venv", ".env", "__pycache__", "node_modules", ".git"],
    "auto_install_requirements": True,
    "requirements_files": ["requirements.txt", "requirements_web.txt"],
    "communication_timeout": 30,
    "heartbeat_interval": 10
}

class SystemLogger:
    """Comprehensive logging system for terminal and file output"""
    
    def __init__(self, name: str, log_file: str, log_level: str = "INFO"):
        self.name = name
        self.log_file = log_file
        self.log_level = getattr(logging, log_level)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler with formatting
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)

class ConnectorCommunication:
    """Handles inter-connector communication"""
    
    def __init__(self, connector_name: str, port: int, logger: SystemLogger):
        self.connector_name = connector_name
        self.port = port
        self.logger = logger
        self.socket = None
        self.clients = {}
        self.message_queue = queue.Queue()
        self.running = False
        self.server_thread = None
        
    def start_server(self):
        """Start the communication server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('localhost', self.port))
            self.socket.listen(5)
            self.running = True
            
            self.server_thread = threading.Thread(target=self._accept_connections)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.logger.info(f"Communication server started on port {self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
    
    def _accept_connections(self):
        """Accept incoming connections"""
        while self.running:
            try:
                self.socket.settimeout(1.0)
                client_socket, address = self.socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.logger.error(f"Error accepting connection: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address: Tuple[str, int]):
        """Handle client messages"""
        try:
            # Receive initial identification
            data = client_socket.recv(1024).decode('utf-8')
            if data:
                message = json.loads(data)
                client_name = message.get('connector', 'unknown')
                self.clients[client_name] = {
                    'socket': client_socket,
                    'address': address,
                    'last_seen': time.time()
                }
                self.logger.info(f"Connected to {client_name} from {address}")
                
                # Send acknowledgment
                ack = {'status': 'connected', 'connector': self.connector_name}
                client_socket.send(json.dumps(ack).encode('utf-8'))
                
                # Handle messages
                while self.running:
                    data = client_socket.recv(1024).decode('utf-8')
                    if not data:
                        break
                    
                    message = json.loads(data)
                    message['from'] = client_name
                    self.message_queue.put(message)
                    self.clients[client_name]['last_seen'] = time.time()
                    
        except Exception as e:
            self.logger.error(f"Error handling client: {e}")
        finally:
            client_socket.close()
            if client_name in self.clients:
                del self.clients[client_name]
                self.logger.info(f"Disconnected from {client_name}")
    
    def send_message(self, target: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific connector"""
        try:
            # Try to connect if not already connected
            if target not in self.clients:
                port = SYSTEM_CONFIG['connectors'].get(target, {}).get('port')
                if port:
                    self._connect_to_connector(target, port)
            
            if target in self.clients:
                client_socket = self.clients[target]['socket']
                message['timestamp'] = time.time()
                client_socket.send(json.dumps(message).encode('utf-8'))
                return True
            else:
                self.logger.warning(f"Cannot send message to {target}: not connected")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending message to {target}: {e}")
            return False
    
    def _connect_to_connector(self, name: str, port: int):
        """Connect to another connector"""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5.0)
            client_socket.connect(('localhost', port))
            
            # Send identification
            ident = {'connector': self.connector_name, 'type': 'identify'}
            client_socket.send(json.dumps(ident).encode('utf-8'))
            
            # Receive acknowledgment
            data = client_socket.recv(1024).decode('utf-8')
            if data:
                ack = json.loads(data)
                if ack.get('status') == 'connected':
                    self.clients[name] = {
                        'socket': client_socket,
                        'address': ('localhost', port),
                        'last_seen': time.time()
                    }
                    self.logger.info(f"Connected to {name} on port {port}")
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to {name}: {e}")
    
    def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected connectors"""
        for target in list(self.clients.keys()):
            self.send_message(target, message)
    
    def get_message(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get message from queue"""
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the communication server"""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.server_thread:
            self.server_thread.join()

class RequirementsManager:
    """Manages automatic detection and installation of requirements"""
    
    def __init__(self, logger: SystemLogger):
        self.logger = logger
        self.installed_packages = set()
        self._detect_installed_packages()
    
    def _detect_installed_packages(self):
        """Detect currently installed packages"""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                self.installed_packages = {pkg['name'].lower() for pkg in packages}
                self.logger.info(f"Detected {len(self.installed_packages)} installed packages")
        except Exception as e:
            self.logger.error(f"Error detecting installed packages: {e}")
    
    def check_and_install_requirements(self, requirements_file: str) -> bool:
        """Check and install requirements from file"""
        if not os.path.exists(requirements_file):
            self.logger.warning(f"Requirements file not found: {requirements_file}")
            return False
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = f.readlines()
            
            missing_packages = []
            for req in requirements:
                req = req.strip()
                if req and not req.startswith('#'):
                    # Extract package name (handle version specifiers)
                    package_name = req.split('>=')[0].split('==')[0].split('~=')[0].strip().lower()
                    if package_name not in self.installed_packages:
                        missing_packages.append(req)
            
            if missing_packages:
                self.logger.info(f"Missing packages: {', '.join(missing_packages)}")
                if SYSTEM_CONFIG['auto_install_requirements']:
                    self._install_packages(missing_packages)
                else:
                    self.logger.warning("Auto-install disabled. Please install manually.")
                    return False
            else:
                self.logger.info("All requirements satisfied")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking requirements: {e}")
            return False
    
    def _install_packages(self, packages: List[str]):
        """Install missing packages"""
        self.logger.info(f"Installing {len(packages)} packages...")
        for package in packages:
            try:
                self.logger.info(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    self.logger.info(f"Successfully installed {package}")
                else:
                    self.logger.error(f"Failed to install {package}: {result.stderr}")
            except Exception as e:
                self.logger.error(f"Error installing {package}: {e}")

class ConfigurationManager:
    """Manages system configuration through interactive questions"""
    
    def __init__(self, logger: SystemLogger):
        self.logger = logger
        self.config = {}
        self.config_file = SYSTEM_CONFIG['config_file']
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                self.logger.error(f"Error loading configuration: {e}")
                self.config = {}
        else:
            self.logger.info("No existing configuration found")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def interactive_setup(self):
        """Interactive configuration setup"""
        print("\n" + "="*60)
        print("POLAR BEAR SYSTEM - CONFIGURATION SETUP")
        print("="*60)
        
        # Basic settings
        self.config['project_name'] = self._ask_question(
            "Project name",
            default=self.config.get('project_name', 'Polar Bear Project')
        )
        
        self.config['log_level'] = self._ask_question(
            "Log level (DEBUG/INFO/WARNING/ERROR)",
            default=self.config.get('log_level', 'INFO'),
            validator=lambda x: x.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        )
        
        # Connector settings
        print("\nConnector Configuration:")
        self.config['connectors_enabled'] = {}
        
        for connector, info in SYSTEM_CONFIG['connectors'].items():
            enabled = self._ask_question(
                f"Enable {info['name']}? (y/n)",
                default='y',
                validator=lambda x: x.lower() in ['y', 'n']
            )
            self.config['connectors_enabled'][connector] = enabled.lower() == 'y'
        
        # Advanced settings
        if self._ask_question("\nConfigure advanced settings? (y/n)", default='n').lower() == 'y':
            self.config['heartbeat_interval'] = int(self._ask_question(
                "Heartbeat interval (seconds)",
                default=str(self.config.get('heartbeat_interval', 10))
            ))
            
            self.config['communication_timeout'] = int(self._ask_question(
                "Communication timeout (seconds)",
                default=str(self.config.get('communication_timeout', 30))
            ))
            
            self.config['auto_restart'] = self._ask_question(
                "Auto-restart failed connectors? (y/n)",
                default='y'
            ).lower() == 'y'
        
        # Save configuration
        self.save_config()
        print("\nConfiguration saved successfully!")
        
    def _ask_question(self, prompt: str, default: str = "", validator=None) -> str:
        """Ask a configuration question"""
        while True:
            if default:
                answer = input(f"{prompt} [{default}]: ").strip()
                if not answer:
                    answer = default
            else:
                answer = input(f"{prompt}: ").strip()
            
            if validator:
                try:
                    if validator(answer):
                        return answer
                    else:
                        print("Invalid input. Please try again.")
                except:
                    print("Invalid input. Please try again.")
            else:
                return answer

class PolarBearSystem:
    """Main control system for Polar Bear project"""
    
    def __init__(self):
        self.logger = SystemLogger(
            "PolarBearSystem",
            SYSTEM_CONFIG['log_file'],
            SYSTEM_CONFIG['log_level']
        )
        self.logger.info("Initializing Polar Bear System...")
        
        self.config_manager = ConfigurationManager(self.logger)
        self.requirements_manager = RequirementsManager(self.logger)
        self.communication = ConnectorCommunication(
            "root",
            SYSTEM_CONFIG['connectors']['root']['port'],
            self.logger
        )
        
        self.connectors = {}
        self.running = False
        self.monitor_thread = None
        
    def initialize(self):
        """Initialize the system"""
        # Check and install requirements
        for req_file in SYSTEM_CONFIG['requirements_files']:
            if os.path.exists(req_file):
                self.logger.info(f"Checking requirements from {req_file}")
                self.requirements_manager.check_and_install_requirements(req_file)
        
        # Start communication server
        self.communication.start_server()
        
        # Deploy connectors
        self.deploy_connectors()
        
        # Start monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_connectors)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("System initialization complete")
    
    def deploy_connectors(self):
        """Deploy connector scripts to directories"""
        self.logger.info("Deploying connectors...")
        
        config = self.config_manager.config.get('connectors_enabled', {})
        
        for connector_name, info in SYSTEM_CONFIG['connectors'].items():
            if connector_name == 'root':
                continue  # Skip root connector
            
            if config.get(connector_name, True):
                self._create_connector_script(connector_name)
    
    def _create_connector_script(self, connector_name: str):
        """Create a connector script for a specific directory"""
        connector_path = os.path.join(os.getcwd(), connector_name, f"{connector_name}_connector.py")
        
        # Check if directory exists
        if not os.path.exists(os.path.dirname(connector_path)):
            self.logger.warning(f"Directory {connector_name} not found, skipping connector")
            return
        
        # Skip if in virtual environment
        parent_dir = os.path.dirname(connector_path)
        for exclude_dir in SYSTEM_CONFIG['directories_to_exclude']:
            if exclude_dir in parent_dir:
                self.logger.warning(f"Skipping {connector_name} - in excluded directory")
                return
        
        # Create connector script (continued in next part)
        self.logger.info(f"Creating connector for {connector_name}")
        self._write_connector_script(connector_path, connector_name)
    
    def _write_connector_script(self, script_path: str, connector_name: str):
        """Write the connector script content"""
        port = SYSTEM_CONFIG['connectors'][connector_name]['port']
        
        script_content = f'''#!/usr/bin/env python3
"""
{connector_name.title()} Connector - Part of Polar Bear System
Auto-generated connector script
"""

import os
import sys
import json
import socket
import threading
import time
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONNECTOR_CONFIG = {{
    "name": "{connector_name}",
    "port": {port},
    "root_port": {SYSTEM_CONFIG['connectors']['root']['port']},
    "heartbeat_interval": 10
}}

class ConnectorLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)

class {connector_name.title()}Connector:
    def __init__(self):
        self.logger = ConnectorLogger("{connector_name.title()}Connector")
        self.socket = None
        self.root_socket = None
        self.running = False
        self.directory = os.path.dirname(os.path.abspath(__file__))
        
    def start(self):
        """Start the connector"""
        self.logger.info(f"Starting {{CONNECTOR_CONFIG['name']}} connector...")
        
        # Connect to root controller
        self.connect_to_root()
        
        # Start server
        self.start_server()
        
        # Start heartbeat
        heartbeat_thread = threading.Thread(target=self.heartbeat_loop)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        
        # Main loop
        self.running = True
        self.main_loop()
    
    def connect_to_root(self):
        """Connect to root controller"""
        try:
            self.root_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.root_socket.connect(('localhost', CONNECTOR_CONFIG['root_port']))
            
            # Send identification
            ident = {{'connector': CONNECTOR_CONFIG['name'], 'type': 'identify'}}
            self.root_socket.send(json.dumps(ident).encode('utf-8'))
            
            # Start receiver thread
            receiver = threading.Thread(target=self.receive_from_root)
            receiver.daemon = True
            receiver.start()
            
            self.logger.info("Connected to root controller")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to root: {{e}}")
    
    def receive_from_root(self):
        """Receive messages from root controller"""
        while self.running:
            try:
                data = self.root_socket.recv(1024).decode('utf-8')
                if data:
                    message = json.loads(data)
                    self.handle_message(message)
            except Exception as e:
                if self.running:
                    self.logger.error(f"Error receiving from root: {{e}}")
                break
    
    def start_server(self):
        """Start the connector server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('localhost', CONNECTOR_CONFIG['port']))
            self.socket.listen(5)
            
            server_thread = threading.Thread(target=self.accept_connections)
            server_thread.daemon = True
            server_thread.start()
            
            self.logger.info(f"Server started on port {{CONNECTOR_CONFIG['port']}}")
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {{e}}")
    
    def accept_connections(self):
        """Accept incoming connections"""
        while self.running:
            try:
                self.socket.settimeout(1.0)
                client_socket, address = self.socket.accept()
                self.logger.info(f"Connection from {{address}}")
                
                # Handle client in new thread
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket,)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.logger.error(f"Error accepting connection: {{e}}")
    
    def handle_client(self, client_socket):
        """Handle client connection"""
        try:
            while self.running:
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                
                message = json.loads(data)
                response = self.process_message(message)
                
                if response:
                    client_socket.send(json.dumps(response).encode('utf-8'))
                    
        except Exception as e:
            self.logger.error(f"Error handling client: {{e}}")
        finally:
            client_socket.close()
    
    def handle_message(self, message):
        """Handle incoming message"""
        msg_type = message.get('type')
        
        if msg_type == 'scan':
            self.scan_directory()
        elif msg_type == 'status':
            self.send_status()
        elif msg_type == 'execute':
            self.execute_command(message.get('command'))
        else:
            self.logger.warning(f"Unknown message type: {{msg_type}}")
    
    def process_message(self, message):
        """Process message and return response"""
        return {{'status': 'received', 'timestamp': time.time()}}
    
    def scan_directory(self):
        """Scan the connector's directory"""
        self.logger.info("Scanning directory...")
        
        files = []
        for root, dirs, filenames in os.walk(self.directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in [
                'venv', 'env', '.venv', '.env', '__pycache__', '.git'
            ]]
            
            for filename in filenames:
                if filename.endswith('.py'):
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, self.directory)
                    files.append(rel_path)
        
        # Send results to root
        self.send_to_root({{
            'type': 'scan_results',
            'connector': CONNECTOR_CONFIG['name'],
            'files': files,
            'count': len(files)
        }})
    
    def send_status(self):
        """Send status to root controller"""
        status = {{
            'type': 'status_update',
            'connector': CONNECTOR_CONFIG['name'],
            'status': 'running',
            'directory': self.directory,
            'timestamp': time.time()
        }}
        self.send_to_root(status)
    
    def execute_command(self, command):
        """Execute a command"""
        self.logger.info(f"Executing command: {{command}}")
        # Add command execution logic here
    
    def send_to_root(self, message):
        """Send message to root controller"""
        try:
            if self.root_socket:
                self.root_socket.send(json.dumps(message).encode('utf-8'))
        except Exception as e:
            self.logger.error(f"Error sending to root: {{e}}")
    
    def heartbeat_loop(self):
        """Send periodic heartbeat"""
        while self.running:
            self.send_to_root({{
                'type': 'heartbeat',
                'connector': CONNECTOR_CONFIG['name'],
                'timestamp': time.time()
            }})
            time.sleep(CONNECTOR_CONFIG['heartbeat_interval'])
    
    def main_loop(self):
        """Main connector loop"""
        self.logger.info("Connector running. Press Ctrl+C to stop.")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the connector"""
        self.running = False
        
        if self.socket:
            self.socket.close()
        
        if self.root_socket:
            self.send_to_root({{
                'type': 'disconnect',
                'connector': CONNECTOR_CONFIG['name']
            }})
            self.root_socket.close()
        
        self.logger.info("Connector stopped")

if __name__ == "__main__":
    connector = {connector_name.title()}Connector()
    connector.start()
'''
        
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable on Unix-like systems
            if os.name != 'nt':
                os.chmod(script_path, 0o755)
            
            self.logger.info(f"Created connector script: {script_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating connector script: {e}")
    
    def _monitor_connectors(self):
        """Monitor connector health"""
        while self.running:
            try:
                # Check for messages
                message = self.communication.get_message()
                if message:
                    self.handle_message(message)
                
                # Check connector health
                current_time = time.time()
                for name, client in list(self.communication.clients.items()):
                    if current_time - client['last_seen'] > SYSTEM_CONFIG['communication_timeout']:
                        self.logger.warning(f"Connector {name} timeout")
                        # Could implement auto-restart here
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
    
    def handle_message(self, message: Dict[str, Any]):
        """Handle incoming messages from connectors"""
        msg_type = message.get('type')
        from_connector = message.get('from')
        
        self.logger.debug(f"Received {msg_type} from {from_connector}")
        
        if msg_type == 'heartbeat':
            # Heartbeat received, update last seen
            pass
        elif msg_type == 'scan_results':
            files = message.get('files', [])
            self.logger.info(f"{from_connector} found {len(files)} Python files")
        elif msg_type == 'status_update':
            status = message.get('status')
            self.logger.info(f"{from_connector} status: {status}")
        elif msg_type == 'disconnect':
            self.logger.info(f"{from_connector} disconnecting")
    
    def send_command(self, target: str, command: str):
        """Send command to specific connector"""
        message = {
            'type': 'execute',
            'command': command,
            'timestamp': time.time()
        }
        
        if target == 'all':
            self.communication.broadcast_message(message)
        else:
            self.communication.send_message(target, message)
    
    def scan_all_connectors(self):
        """Request all connectors to scan their directories"""
        self.logger.info("Requesting directory scan from all connectors...")
        self.communication.broadcast_message({'type': 'scan'})
    
    def get_system_status(self):
        """Get status from all connectors"""
        self.logger.info("Requesting status from all connectors...")
        self.communication.broadcast_message({'type': 'status'})
    
    def interactive_mode(self):
        """Interactive control mode"""
        print("\n" + "="*60)
        print("POLAR BEAR SYSTEM - INTERACTIVE MODE")
        print("="*60)
        print("\nCommands:")
        print("  status    - Get status from all connectors")
        print("  scan      - Scan all connector directories")
        print("  send      - Send command to specific connector")
        print("  list      - List connected connectors")
        print("  config    - Configure system")
        print("  restart   - Restart a connector")
        print("  exit      - Exit interactive mode")
        print("="*60)
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'exit':
                    break
                elif command == 'status':
                    self.get_system_status()
                elif command == 'scan':
                    self.scan_all_connectors()
                elif command == 'list':
                    print("\nConnected connectors:")
                    for name in self.communication.clients.keys():
                        print(f"  - {name}")
                elif command == 'send':
                    target = input("Target connector (or 'all'): ").strip()
                    cmd = input("Command: ").strip()
                    self.send_command(target, cmd)
                elif command == 'config':
                    self.config_manager.interactive_setup()
                elif command == 'restart':
                    connector = input("Connector to restart: ").strip()
                    self.restart_connector(connector)
                else:
                    print("Unknown command. Type 'exit' to quit.")
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' command to quit.")
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {e}")
    
    def restart_connector(self, connector_name: str):
        """Restart a specific connector"""
        self.logger.info(f"Restarting {connector_name} connector...")
        
        # Send shutdown command
        self.communication.send_message(connector_name, {'type': 'shutdown'})
        
        # Wait a moment
        time.sleep(2)
        
        # Recreate and start connector
        self._create_connector_script(connector_name)
        
        # Could implement actual process restart here
        self.logger.info(f"Connector {connector_name} restart initiated")
    
    def shutdown(self):
        """Shutdown the system"""
        self.logger.info("Shutting down Polar Bear System...")
        
        # Notify all connectors
        self.communication.broadcast_message({'type': 'shutdown'})
        
        # Stop monitoring
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # Stop communication
        self.communication.stop()
        
        self.logger.info("System shutdown complete")

def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("POLAR BEAR SYSTEM v" + SYSTEM_CONFIG['version'])
    print("="*60)
    
    # Create system instance
    system = PolarBearSystem()
    
    # Check for first run
    if not os.path.exists(SYSTEM_CONFIG['config_file']):
        print("\nFirst run detected. Starting configuration setup...")
        system.config_manager.interactive_setup()
    
    try:
        # Initialize system
        system.initialize()
        
        # Enter interactive mode
        system.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        system.logger.error(f"System error: {e}")
        raise
    finally:
        # Cleanup
        system.shutdown()

if __name__ == "__main__":
    main()