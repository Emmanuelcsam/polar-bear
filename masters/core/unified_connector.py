#!/usr/bin/env python3
"""
Unified Connector System - Consolidates all connector functionality
Supports basic, enhanced, hivemind, and mega modes
"""

import os
import sys
import json
import logging
import socket
import threading
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback

class ConnectorMode(Enum):
    """Available connector modes"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    HIVEMIND = "hivemind"
    MEGA = "mega"

@dataclass
class ConnectorConfig:
    """Unified connector configuration"""
    mode: ConnectorMode = ConnectorMode.BASIC
    connector_id: str = "unified_001"
    port: int = 10065
    parent_port: Optional[int] = None
    depth: int = 0
    log_file: str = "unified_connector.log"
    scan_interval: int = 60
    enable_networking: bool = False
    enable_execution: bool = False
    enable_discovery: bool = True
    max_threads: int = 4

class UnifiedConnector:
    """Unified connector with all modes"""
    
    def __init__(self, config: Optional[ConnectorConfig] = None):
        self.config = config or ConnectorConfig()
        self.logger = self._setup_logging()
        self.discovered_scripts = {}
        self.connected_nodes = {}
        self.running = False
        self.threads = []
        
        # Mode-specific initialization
        if self.config.mode in [ConnectorMode.HIVEMIND, ConnectorMode.MEGA]:
            self.config.enable_networking = True
            self.config.enable_execution = True
            
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f"UnifiedConnector_{self.config.connector_id}")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.config.log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def start(self):
        """Start the connector based on configured mode"""
        self.running = True
        self.logger.info(f"Starting Unified Connector in {self.config.mode.value} mode")
        
        if self.config.mode == ConnectorMode.BASIC:
            self._run_basic_mode()
        elif self.config.mode == ConnectorMode.ENHANCED:
            self._run_enhanced_mode()
        elif self.config.mode == ConnectorMode.HIVEMIND:
            self._run_hivemind_mode()
        elif self.config.mode == ConnectorMode.MEGA:
            self._run_mega_mode()
            
    def _run_basic_mode(self):
        """Basic mode - simple file discovery and logging"""
        self.logger.info("Running in BASIC mode - File discovery only")
        
        while self.running:
            try:
                # Scan for Python files
                self._scan_directory()
                
                # Log discovered scripts
                self.logger.info(f"Discovered {len(self.discovered_scripts)} scripts")
                for script, info in self.discovered_scripts.items():
                    self.logger.info(f"  - {script}: {info.get('description', 'No description')}")
                    
                # Wait before next scan
                time.sleep(self.config.scan_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                self.stop()
            except Exception as e:
                self.logger.error(f"Error in basic mode: {e}")
                
    def _run_enhanced_mode(self):
        """Enhanced mode - adds script analysis and categorization"""
        self.logger.info("Running in ENHANCED mode - With script analysis")
        
        # Start discovery thread
        discovery_thread = threading.Thread(target=self._continuous_discovery)
        discovery_thread.start()
        self.threads.append(discovery_thread)
        
        while self.running:
            try:
                # Analyze discovered scripts
                self._analyze_scripts()
                
                # Generate report
                self._generate_report()
                
                time.sleep(30)  # Report every 30 seconds
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                self.stop()
            except Exception as e:
                self.logger.error(f"Error in enhanced mode: {e}")
                
    def _run_hivemind_mode(self):
        """Hivemind mode - networked operation with parent/child relationships"""
        self.logger.info(f"Running in HIVEMIND mode - Network enabled on port {self.config.port}")
        
        # Start network server
        if self.config.enable_networking:
            server_thread = threading.Thread(target=self._start_network_server)
            server_thread.start()
            self.threads.append(server_thread)
            
        # Connect to parent if configured
        if self.config.parent_port:
            self._connect_to_parent()
            
        # Start discovery
        discovery_thread = threading.Thread(target=self._continuous_discovery)
        discovery_thread.start()
        self.threads.append(discovery_thread)
        
        # Start heartbeat
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        heartbeat_thread.start()
        self.threads.append(heartbeat_thread)
        
        # Main loop
        while self.running:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                self.stop()
                
    def _run_mega_mode(self):
        """Mega mode - full feature set with all capabilities"""
        self.logger.info("Running in MEGA mode - All features enabled")
        
        # This would include all features from ultimate_mega_connector
        # For now, we'll run hivemind mode as base
        self._run_hivemind_mode()
        
    def _scan_directory(self, directory: str = "."):
        """Scan directory for Python scripts"""
        try:
            for root, dirs, files in os.walk(directory):
                # Skip hidden directories and __pycache__
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for file in files:
                    if file.endswith('.py'):
                        filepath = os.path.join(root, file)
                        self._register_script(filepath)
                        
        except Exception as e:
            self.logger.error(f"Error scanning directory: {e}")
            
    def _register_script(self, filepath: str):
        """Register a discovered script"""
        try:
            # Get basic info
            stat = os.stat(filepath)
            relative_path = os.path.relpath(filepath)
            
            # Try to extract description from file
            description = self._extract_description(filepath)
            
            self.discovered_scripts[relative_path] = {
                'absolute_path': os.path.abspath(filepath),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'description': description,
                'category': self._categorize_script(relative_path),
                'analyzed': False
            }
            
        except Exception as e:
            self.logger.error(f"Error registering script {filepath}: {e}")
            
    def _extract_description(self, filepath: str) -> str:
        """Extract description from script docstring"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for module docstring
            import ast
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
            
            if docstring:
                # Return first line of docstring
                return docstring.split('\n')[0].strip()
                
        except Exception:
            pass
            
        return "No description available"
        
    def _categorize_script(self, filepath: str) -> str:
        """Categorize script based on path and name"""
        filepath_lower = filepath.lower()
        
        if 'connector' in filepath_lower:
            return 'connector'
        elif 'config' in filepath_lower:
            return 'configuration'
        elif 'module' in filepath_lower:
            return 'module'
        elif 'test' in filepath_lower:
            return 'testing'
        elif 'util' in filepath_lower:
            return 'utility'
        elif 'setup' in filepath_lower:
            return 'setup'
        else:
            return 'general'
            
    def _continuous_discovery(self):
        """Continuously scan for new scripts"""
        while self.running:
            self._scan_directory()
            time.sleep(self.config.scan_interval)
            
    def _analyze_scripts(self):
        """Analyze discovered scripts for deeper understanding"""
        for script_path, info in self.discovered_scripts.items():
            if not info.get('analyzed', False):
                try:
                    # Perform deeper analysis
                    info['imports'] = self._extract_imports(info['absolute_path'])
                    info['functions'] = self._extract_functions(info['absolute_path'])
                    info['classes'] = self._extract_classes(info['absolute_path'])
                    info['analyzed'] = True
                except Exception as e:
                    self.logger.error(f"Error analyzing {script_path}: {e}")
                    
    def _extract_imports(self, filepath: str) -> List[str]:
        """Extract imports from a Python file"""
        imports = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
                        
        except Exception:
            pass
            
        return imports
        
    def _extract_functions(self, filepath: str) -> List[str]:
        """Extract function names from a Python file"""
        functions = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                    
        except Exception:
            pass
            
        return functions
        
    def _extract_classes(self, filepath: str) -> List[str]:
        """Extract class names from a Python file"""
        classes = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    
        except Exception:
            pass
            
        return classes
        
    def _generate_report(self):
        """Generate a report of discovered scripts"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'connector_id': self.config.connector_id,
            'mode': self.config.mode.value,
            'total_scripts': len(self.discovered_scripts),
            'categories': {},
            'scripts': self.discovered_scripts
        }
        
        # Count by category
        for script, info in self.discovered_scripts.items():
            category = info.get('category', 'unknown')
            if category not in report['categories']:
                report['categories'][category] = 0
            report['categories'][category] += 1
            
        # Save report
        report_file = f"connector_report_{self.config.connector_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Report saved to {report_file}")
        
    def _start_network_server(self):
        """Start network server for hivemind communication"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('localhost', self.config.port))
            server_socket.listen(5)
            server_socket.settimeout(1.0)
            
            self.logger.info(f"Network server listening on port {self.config.port}")
            
            while self.running:
                try:
                    client_socket, address = server_socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Server error: {e}")
                    
            server_socket.close()
            
        except Exception as e:
            self.logger.error(f"Failed to start network server: {e}")
            
    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle client connections"""
        try:
            # Receive command
            data = client_socket.recv(1024).decode('utf-8')
            command = json.loads(data)
            
            response = self._process_command(command)
            
            # Send response
            client_socket.send(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.logger.error(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            
    def _process_command(self, command: dict) -> dict:
        """Process network commands"""
        cmd_type = command.get('type', 'unknown')
        
        if cmd_type == 'status':
            return {
                'status': 'ok',
                'connector_id': self.config.connector_id,
                'mode': self.config.mode.value,
                'scripts_discovered': len(self.discovered_scripts),
                'uptime': time.time()
            }
        elif cmd_type == 'list_scripts':
            return {
                'status': 'ok',
                'scripts': list(self.discovered_scripts.keys())
            }
        elif cmd_type == 'execute' and self.config.enable_execution:
            script = command.get('script')
            if script in self.discovered_scripts:
                # Execute script (simplified for safety)
                return {
                    'status': 'ok',
                    'message': f'Execution of {script} not implemented in unified connector'
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Script not found'
                }
        else:
            return {
                'status': 'error',
                'message': f'Unknown command: {cmd_type}'
            }
            
    def _connect_to_parent(self):
        """Connect to parent node in hivemind"""
        try:
            parent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            parent_socket.connect(('localhost', self.config.parent_port))
            
            # Send registration
            registration = {
                'type': 'register',
                'connector_id': self.config.connector_id,
                'port': self.config.port,
                'depth': self.config.depth
            }
            parent_socket.send(json.dumps(registration).encode('utf-8'))
            
            # Receive response
            response = parent_socket.recv(1024).decode('utf-8')
            self.logger.info(f"Registered with parent: {response}")
            
            parent_socket.close()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to parent: {e}")
            
    def _heartbeat_loop(self):
        """Send heartbeats to parent"""
        while self.running:
            if self.config.parent_port:
                try:
                    # Send heartbeat to parent
                    parent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    parent_socket.settimeout(5.0)
                    parent_socket.connect(('localhost', self.config.parent_port))
                    
                    heartbeat = {
                        'type': 'heartbeat',
                        'connector_id': self.config.connector_id,
                        'timestamp': time.time()
                    }
                    parent_socket.send(json.dumps(heartbeat).encode('utf-8'))
                    parent_socket.close()
                    
                except Exception as e:
                    self.logger.warning(f"Heartbeat failed: {e}")
                    
            time.sleep(30)  # Heartbeat every 30 seconds
            
    def stop(self):
        """Stop the connector"""
        self.logger.info("Stopping Unified Connector")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
            
        self.logger.info("Unified Connector stopped")
        
    @classmethod
    def from_legacy_config(cls, legacy_type: str, **kwargs):
        """Create unified connector from legacy configuration"""
        config = ConnectorConfig()
        
        if legacy_type == 'basic':
            config.mode = ConnectorMode.BASIC
        elif legacy_type == 'hivemind':
            config.mode = ConnectorMode.HIVEMIND
            config.connector_id = kwargs.get('connector_id', 'legacy_hivemind')
            config.port = kwargs.get('port', 10065)
            config.parent_port = kwargs.get('parent_port')
            config.depth = kwargs.get('depth', 0)
        elif legacy_type == 'mega':
            config.mode = ConnectorMode.MEGA
            
        return cls(config)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Connector System')
    parser.add_argument('--mode', choices=['basic', 'enhanced', 'hivemind', 'mega'],
                        default='basic', help='Connector mode')
    parser.add_argument('--port', type=int, default=10065,
                        help='Port for network communication')
    parser.add_argument('--parent-port', type=int,
                        help='Parent connector port for hivemind mode')
    parser.add_argument('--id', default='unified_001',
                        help='Connector ID')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ConnectorConfig(
        mode=ConnectorMode(args.mode),
        connector_id=args.id,
        port=args.port,
        parent_port=args.parent_port
    )
    
    # Create and start connector
    connector = UnifiedConnector(config)
    
    try:
        connector.start()
    except KeyboardInterrupt:
        connector.stop()

if __name__ == "__main__":
    main()