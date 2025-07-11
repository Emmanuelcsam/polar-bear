#!/usr/bin/env python3
"""
Polar Bear Hivemind System - Deep Recursive Connector Network
This system creates connector scripts in EVERY subdirectory throughout the project
"""

import os
import sys
import json
import socket
import threading
import time
import subprocess
import logging
import queue
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
import importlib.util
import pkg_resources

class PolarBearHivemind:
    """Main control system for deep recursive connector network"""
    
    def __init__(self, root_path: str = None):
        self.root_path = Path(root_path or os.getcwd())
        self.config_file = self.root_path / "hivemind_config.json"
        self.log_dir = self.root_path / "hivemind_logs"
        self.connectors: Dict[str, dict] = {}
        self.connector_hierarchy: Dict[str, List[str]] = {}
        self.message_queue = queue.Queue()
        self.running = False
        self.port_base = 10000
        self.used_ports: Set[int] = set()
        
        # Directories to skip
        self.skip_dirs = {
            'venv', 'env', '.env', '__pycache__', '.git', 
            'node_modules', '.venv', 'virtualenv', '.tox',
            'build', 'dist', '.pytest_cache', '.mypy_cache',
            'egg-info', '.egg-info'
        }
        
        # Setup logging
        self.setup_logging()
        
        # Load or create configuration
        self.config = self.load_config()
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        self.log_dir.mkdir(exist_ok=True)
        
        # File handler
        log_file = self.log_dir / f"hivemind_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Logger
        self.logger = logging.getLogger('PolarBearHivemind')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def load_config(self) -> dict:
        """Load or create configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        
        # Default configuration
        config = {
            "project_name": "Polar Bear Hivemind",
            "auto_install_requirements": True,
            "scan_interval": 60,
            "max_connector_depth": 10,
            "connector_timeout": 30,
            "enable_troubleshooting": True,
            "log_level": "INFO"
        }
        
        self.save_config(config)
        return config
        
    def save_config(self, config: dict):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    def check_requirements(self):
        """Check and install required packages"""
        self.logger.info("Checking system requirements...")
        
        required_packages = [
            'psutil',
            'colorama',
            'requests'
        ]
        
        optional_packages = [
            'watchdog'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        # Check optional packages
        for package in optional_packages:
            try:
                __import__(package)
                self.logger.info(f"Optional package {package} is available")
            except ImportError:
                self.logger.warning(f"Optional package {package} not available - some features may be limited")
                
        if missing_packages and self.config['auto_install_requirements']:
            self.logger.info(f"Installing missing packages: {missing_packages}")
            for package in missing_packages:
                try:
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install', package, '--upgrade'
                    ])
                    self.logger.info(f"Successfully installed {package}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to install {package}: {e}")
                    
    def get_directory_depth(self, path: Path) -> int:
        """Get the depth of a directory relative to root"""
        try:
            relative = path.relative_to(self.root_path)
            return len(relative.parts)
        except ValueError:
            return 0
            
    def generate_connector_id(self, path: Path) -> str:
        """Generate unique ID for a connector based on its path"""
        relative_path = path.relative_to(self.root_path)
        return hashlib.md5(str(relative_path).encode()).hexdigest()[:8]
        
    def get_available_port(self) -> int:
        """Get an available port for a connector"""
        port = self.port_base
        while port in self.used_ports or self.is_port_in_use(port):
            port += 1
        self.used_ports.add(port)
        return port
        
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return False
            except:
                return True
                
    def should_skip_directory(self, path: Path) -> bool:
        """Check if directory should be skipped"""
        dir_name = path.name
        
        # Skip hidden directories
        if dir_name.startswith('.'):
            return True
            
        # Skip known virtual environment directories
        if dir_name.lower() in self.skip_dirs:
            return True
            
        # Skip if directory name contains 'venv' or 'env'
        if 'venv' in dir_name.lower() or 'env' in dir_name.lower():
            return True
            
        return False
        
    def create_connector_script(self, directory: Path, parent_id: Optional[str] = None) -> Optional[str]:
        """Create a connector script in the specified directory"""
        if self.should_skip_directory(directory):
            return None
            
        connector_file = directory / "hivemind_connector.py"
        connector_id = self.generate_connector_id(directory)
        port = self.get_available_port()
        depth = self.get_directory_depth(directory)
        
        # Determine parent port
        parent_port = 10000  # Root hivemind port
        if parent_id and parent_id in self.connectors:
            parent_port = self.connectors[parent_id]['port']
        
        connector_script = f'''#!/usr/bin/env python3
"""
Hivemind Connector for: {directory.relative_to(self.root_path)}
Connector ID: {connector_id}
Depth Level: {depth}
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

class HivemindConnector:
    def __init__(self):
        self.connector_id = "{connector_id}"
        self.directory = Path("{directory}")
        self.port = {port}
        self.parent_port = {parent_port}
        self.depth = {depth}
        self.running = True
        self.socket = None
        self.parent_socket = None
        self.child_connectors = {{}}
        self.scripts_in_directory = {{}}
        
        # Setup logging
        self.logger = logging.getLogger(f"Connector_{{self.connector_id}}")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def start(self):
        """Start the connector"""
        self.logger.info(f"Starting connector {{self.connector_id}} on port {{self.port}}")
        
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
            self.logger.error(f"Error handling connection: {{e}}")
        finally:
            conn.close()
            
    def process_message(self, message):
        """Process incoming message"""
        cmd = message.get('command')
        
        if cmd == 'status':
            return {{
                'status': 'active',
                'connector_id': self.connector_id,
                'directory': str(self.directory),
                'depth': self.depth,
                'scripts': len(self.scripts_in_directory),
                'children': len(self.child_connectors)
            }}
        elif cmd == 'scan':
            self.scan_directory()
            return {{'status': 'scan_complete', 'scripts': list(self.scripts_in_directory.keys())}}
        elif cmd == 'execute':
            script = message.get('script')
            if script in self.scripts_in_directory:
                return self.execute_script(script)
            return {{'error': 'Script not found'}}
        elif cmd == 'troubleshoot':
            return self.troubleshoot_connections()
        else:
            return {{'error': 'Unknown command'}}
            
    def connect_to_parent(self):
        """Connect to parent connector"""
        try:
            self.parent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.parent_socket.connect(('localhost', self.parent_port))
            
            # Register with parent
            register_msg = {{
                'command': 'register',
                'connector_id': self.connector_id,
                'port': self.port,
                'directory': str(self.directory)
            }}
            self.parent_socket.send(json.dumps(register_msg).encode())
            self.logger.info(f"Connected to parent on port {{self.parent_port}}")
        except Exception as e:
            self.logger.error(f"Failed to connect to parent: {{e}}")
            
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
                        
            self.logger.info(f"Found {{len(self.scripts_in_directory)}} scripts and {{len(self.child_connectors)}} child connectors")
        except Exception as e:
            self.logger.error(f"Error scanning directory: {{e}}")
            
    def should_skip_directory(self, path):
        """Check if directory should be skipped"""
        skip_dirs = {{
            'venv', 'env', '.env', '__pycache__', '.git', 
            'node_modules', '.venv', 'virtualenv', '.tox',
            'build', 'dist', '.pytest_cache', '.mypy_cache'
        }}
        return path.name.startswith('.') or path.name.lower() in skip_dirs
        
    def execute_script(self, script_name):
        """Execute a script in the directory"""
        if script_name not in self.scripts_in_directory:
            return {{'error': 'Script not found'}}
            
        try:
            result = subprocess.run(
                [sys.executable, self.scripts_in_directory[script_name]],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {{
                'status': 'executed',
                'script': script_name,
                'returncode': result.returncode,
                'stdout': result.stdout[-1000:],  # Last 1000 chars
                'stderr': result.stderr[-1000:]
            }}
        except Exception as e:
            return {{'error': f'Execution failed: {{str(e)}}'}}
            
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
                issues.append(f"Child connector missing: {{name}}")
                
        return {{
            'connector_id': self.connector_id,
            'issues': issues,
            'healthy': len(issues) == 0
        }}
        
    def heartbeat_loop(self):
        """Send heartbeat to parent"""
        while self.running:
            try:
                if self.parent_socket:
                    heartbeat = {{
                        'command': 'heartbeat',
                        'connector_id': self.connector_id,
                        'timestamp': time.time()
                    }}
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
        print("\\nConnector stopped")
'''
        
        try:
            with open(connector_file, 'w') as f:
                f.write(connector_script)
            os.chmod(connector_file, 0o755)
            
            # Store connector info
            self.connectors[connector_id] = {
                'id': connector_id,
                'path': str(directory),
                'port': port,
                'depth': depth,
                'parent_id': parent_id,
                'children': []
            }
            
            # Update parent's children list
            if parent_id and parent_id in self.connectors:
                self.connectors[parent_id]['children'].append(connector_id)
                
            self.logger.info(f"Created connector in {directory.relative_to(self.root_path)} (ID: {connector_id}, Port: {port})")
            return connector_id
            
        except Exception as e:
            self.logger.error(f"Failed to create connector in {directory}: {e}")
            return None
            
    def deploy_connectors_recursively(self, directory: Path = None, parent_id: Optional[str] = None, current_depth: int = 0):
        """Recursively deploy connectors to all subdirectories"""
        if directory is None:
            directory = self.root_path
            
        if current_depth > self.config['max_connector_depth']:
            self.logger.warning(f"Max depth reached at {directory}")
            return
            
        # Create connector in current directory (skip root)
        connector_id = None
        if directory != self.root_path:
            connector_id = self.create_connector_script(directory, parent_id)
            if not connector_id:
                return  # Skip this directory and its subdirectories
                
        # Process subdirectories
        try:
            for item in directory.iterdir():
                if item.is_dir() and not self.should_skip_directory(item):
                    self.deploy_connectors_recursively(
                        item, 
                        connector_id or parent_id, 
                        current_depth + 1
                    )
        except PermissionError:
            self.logger.warning(f"Permission denied accessing {directory}")
            
    def start_root_server(self):
        """Start the root server that manages all connectors"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', 10000))
        self.server_socket.listen(10)
        
        self.logger.info("Root server started on port 10000")
        
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self.handle_root_connection, args=(conn,)).start()
            except:
                if self.running:
                    self.logger.error("Error accepting connection")
                    
    def handle_root_connection(self, conn):
        """Handle connection to root server"""
        try:
            data = conn.recv(4096).decode()
            message = json.loads(data)
            
            cmd = message.get('command')
            if cmd == 'register':
                connector_id = message.get('connector_id')
                if connector_id in self.connectors:
                    self.connectors[connector_id]['active'] = True
                    self.connectors[connector_id]['last_seen'] = time.time()
                    self.logger.info(f"Connector {connector_id} registered")
            elif cmd == 'heartbeat':
                connector_id = message.get('connector_id')
                if connector_id in self.connectors:
                    self.connectors[connector_id]['last_seen'] = time.time()
                    
        except Exception as e:
            self.logger.error(f"Error handling root connection: {e}")
        finally:
            conn.close()
            
    def interactive_setup(self):
        """Interactive configuration setup"""
        print("\n=== Polar Bear Hivemind Configuration ===\n")
        
        # Project name
        project_name = input(f"Project name [{self.config['project_name']}]: ").strip()
        if project_name:
            self.config['project_name'] = project_name
            
        # Auto-install requirements
        auto_install = input("Auto-install missing requirements? (y/n) [y]: ").strip().lower()
        self.config['auto_install_requirements'] = auto_install != 'n'
        
        # Max depth
        max_depth = input(f"Maximum directory depth [{self.config['max_connector_depth']}]: ").strip()
        if max_depth.isdigit():
            self.config['max_connector_depth'] = int(max_depth)
            
        # Scan interval
        scan_interval = input(f"Directory scan interval (seconds) [{self.config['scan_interval']}]: ").strip()
        if scan_interval.isdigit():
            self.config['scan_interval'] = int(scan_interval)
            
        self.save_config(self.config)
        print("\nConfiguration saved!")
        
    def run(self):
        """Main run method"""
        self.running = True
        
        # Check requirements
        self.check_requirements()
        
        # Interactive setup if first run
        if not self.config_file.exists():
            self.interactive_setup()
            
        print(f"\n{'='*60}")
        print(f"Starting {self.config['project_name']}")
        print(f"Root directory: {self.root_path}")
        print(f"{'='*60}\n")
        
        # Deploy connectors
        self.logger.info("Deploying connectors throughout the project...")
        self.deploy_connectors_recursively()
        
        self.logger.info(f"Deployed {len(self.connectors)} connectors")
        
        # Start root server
        server_thread = threading.Thread(target=self.start_root_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Start connectors
        self.start_all_connectors()
        
        # Interactive command loop
        self.command_loop()
        
    def start_all_connectors(self):
        """Start all deployed connectors"""
        self.logger.info("Starting all connectors...")
        
        # Start connectors in order of depth (parent before children)
        sorted_connectors = sorted(
            self.connectors.items(), 
            key=lambda x: x[1]['depth']
        )
        
        for connector_id, info in sorted_connectors:
            connector_script = Path(info['path']) / 'hivemind_connector.py'
            if connector_script.exists():
                try:
                    subprocess.Popen(
                        [sys.executable, str(connector_script)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    time.sleep(0.1)  # Small delay between starts
                except Exception as e:
                    self.logger.error(f"Failed to start connector {connector_id}: {e}")
                    
    def command_loop(self):
        """Interactive command interface"""
        print("\nHivemind is running. Available commands:")
        print("  status  - Show system status")
        print("  tree    - Show connector hierarchy")
        print("  scan    - Rescan all directories")
        print("  check   - Check connector health")
        print("  reload  - Reload configuration")
        print("  exit    - Stop hivemind\n")
        
        while self.running:
            try:
                command = input("hivemind> ").strip().lower()
                
                if command == 'exit':
                    self.shutdown()
                    break
                elif command == 'status':
                    self.show_status()
                elif command == 'tree':
                    self.show_hierarchy()
                elif command == 'scan':
                    self.rescan_all()
                elif command == 'check':
                    self.check_connector_health()
                elif command == 'reload':
                    self.config = self.load_config()
                    print("Configuration reloaded")
                elif command:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' to stop hivemind")
                
    def show_status(self):
        """Show system status"""
        active = sum(1 for c in self.connectors.values() if c.get('active', False))
        print(f"\nTotal connectors: {len(self.connectors)}")
        print(f"Active connectors: {active}")
        print(f"Port range: {self.port_base} - {max(self.used_ports) if self.used_ports else self.port_base}")
        
    def show_hierarchy(self, connector_id=None, indent=0):
        """Show connector hierarchy as tree"""
        if connector_id is None:
            print("\nConnector Hierarchy:")
            # Find root level connectors
            for cid, info in self.connectors.items():
                if info['depth'] == 1:
                    self.show_hierarchy(cid, 1)
        else:
            info = self.connectors.get(connector_id)
            if info:
                status = "●" if info.get('active', False) else "○"
                print(f"{'  ' * indent}{status} {Path(info['path']).name}/ (port: {info['port']})")
                for child_id in info['children']:
                    self.show_hierarchy(child_id, indent + 1)
                    
    def check_connector_health(self):
        """Check health of all connectors"""
        print("\nChecking connector health...")
        healthy = 0
        issues = []
        
        for connector_id, info in self.connectors.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect(('localhost', info['port']))
                sock.send(json.dumps({'command': 'status'}).encode())
                response = sock.recv(4096).decode()
                sock.close()
                healthy += 1
            except:
                issues.append(f"{info['path']} (port {info['port']})")
                
        print(f"Healthy connectors: {healthy}/{len(self.connectors)}")
        if issues:
            print("Connectors with issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
                
    def rescan_all(self):
        """Trigger rescan on all connectors"""
        print("Triggering rescan on all connectors...")
        count = 0
        
        for connector_id, info in self.connectors.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect(('localhost', info['port']))
                sock.send(json.dumps({'command': 'scan'}).encode())
                sock.close()
                count += 1
            except:
                pass
                
        print(f"Rescan triggered on {count} connectors")
        
    def shutdown(self):
        """Shutdown the hivemind"""
        print("\nShutting down hivemind...")
        self.running = False
        if hasattr(self, 'server_socket'):
            self.server_socket.close()
        
        # Save final state
        state_file = self.log_dir / f"hivemind_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(state_file, 'w') as f:
            json.dump({
                'connectors': self.connectors,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
            
        print("Hivemind stopped")

if __name__ == "__main__":
    hivemind = PolarBearHivemind()
    try:
        hivemind.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()