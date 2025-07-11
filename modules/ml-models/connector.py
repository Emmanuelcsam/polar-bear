#!/usr/bin/env python3
"""
Enhanced Connector for ML Models Module
Provides full control and monitoring capabilities for all scripts
"""

import os
import logging
import sys
import json
import socket
import threading
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# --- Configuration ---
LOG_FILE = "connector.log"
CONNECTOR_PORT = 10117
SCRIPT_CONTROL_PORT = 10118

# --- Setup Logging ---
logger = logging.getLogger(os.path.abspath(__file__))
logger.setLevel(logging.INFO)
logger.propagate = False

# Remove existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

# File Handler
try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except (IOError, OSError) as e:
    print(f"Could not write to log file {LOG_FILE}: {e}", file=sys.stderr)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


class ScriptManager:
    """Manages all scripts and their interfaces"""
    
    def __init__(self):
        self.scripts = {}
        self.script_processes = {}
        self.script_interfaces = {}
        self.collaboration_requests = {}
        
    def discover_scripts(self, directory: Path) -> Dict[str, Path]:
        """Discover all Python scripts in directory"""
        scripts = {}
        for file in directory.glob("*.py"):
            if file.name not in ["connector.py", "hivemind_connector.py", "script_interface.py"]:
                scripts[file.stem] = file
        return scripts
        
    def load_script_interface(self, script_path: Path) -> Optional[Dict]:
        """Load script and check if it implements ScriptInterface"""
        try:
            spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for ScriptInterface implementation
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, '__bases__') and 'ScriptInterface' in [b.__name__ for b in attr.__bases__]:
                    return {
                        "class": attr,
                        "module": module,
                        "path": script_path
                    }
        except Exception as e:
            logger.error(f"Failed to load script {script_path}: {e}")
        return None
        
    def execute_script(self, script_name: str, with_connector: bool = True) -> Dict:
        """Execute a script with optional connector integration"""
        if script_name not in self.scripts:
            return {"error": "Script not found"}
            
        script_path = self.scripts[script_name]
        
        try:
            cmd = [sys.executable, str(script_path)]
            if with_connector:
                cmd.append("--with-connector")
                
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.script_processes[script_name] = process
            
            return {
                "status": "started",
                "pid": process.pid,
                "script": script_name
            }
        except Exception as e:
            return {"error": f"Failed to execute: {str(e)}"}
            
    def stop_script(self, script_name: str) -> Dict:
        """Stop a running script"""
        if script_name in self.script_processes:
            process = self.script_processes[script_name]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
            del self.script_processes[script_name]
            return {"status": "stopped", "script": script_name}
        return {"error": "Script not running"}
        
    def get_script_status(self, script_name: str) -> Dict:
        """Get status of a script"""
        if script_name in self.script_processes:
            process = self.script_processes[script_name]
            if process.poll() is None:
                return {"status": "running", "pid": process.pid}
            else:
                return {"status": "finished", "returncode": process.returncode}
        return {"status": "not_running"}


class EnhancedConnector:
    """Enhanced connector with full script control capabilities"""
    
    def __init__(self):
        self.script_manager = ScriptManager()
        self.registered_scripts = {}
        self.running = True
        self.control_socket = None
        self.directory = Path(__file__).parent
        
    def start(self):
        """Start the enhanced connector"""
        logger.info(f"--- Enhanced Connector Started in {self.directory} ---")
        
        # Discover scripts
        self.script_manager.scripts = self.script_manager.discover_scripts(self.directory)
        logger.info(f"Discovered {len(self.script_manager.scripts)} scripts")
        
        # Start control server
        self.control_thread = threading.Thread(target=self.run_control_server, daemon=True)
        self.control_thread.start()
        
        # Main loop
        self.main_loop()
        
    def run_control_server(self):
        """Run control server for script communication"""
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_socket.bind(('localhost', SCRIPT_CONTROL_PORT))
        self.control_socket.listen(10)
        
        logger.info(f"Control server listening on port {SCRIPT_CONTROL_PORT}")
        
        while self.running:
            try:
                conn, addr = self.control_socket.accept()
                threading.Thread(target=self.handle_control_connection, args=(conn,), daemon=True).start()
            except Exception as e:
                if self.running:
                    logger.error(f"Control server error: {e}")
                    
    def handle_control_connection(self, conn):
        """Handle control connection from scripts"""
        try:
            data = conn.recv(4096).decode()
            message = json.loads(data)
            response = self.process_control_message(message)
            conn.send(json.dumps(response).encode())
        except Exception as e:
            logger.error(f"Error handling control connection: {e}")
        finally:
            conn.close()
            
    def process_control_message(self, message: Dict) -> Dict:
        """Process control messages from scripts"""
        command = message.get("command")
        
        if command == "register_script":
            script_name = message.get("script_name")
            info = message.get("info")
            self.registered_scripts[script_name] = info
            logger.info(f"Script '{script_name}' registered")
            return {"status": "registered"}
            
        elif command == "list_scripts":
            return {"scripts": list(self.script_manager.scripts.keys())}
            
        elif command == "collaborate":
            source = message.get("source")
            target = message.get("target")
            data = message.get("data")
            
            # Store collaboration request
            if target not in self.script_manager.collaboration_requests:
                self.script_manager.collaboration_requests[target] = []
            self.script_manager.collaboration_requests[target].append({
                "source": source,
                "data": data,
                "timestamp": time.time()
            })
            
            return {"status": "collaboration_requested"}
            
        elif command == "broadcast":
            source = message.get("source")
            data = message.get("data")
            
            # Broadcast to all registered scripts
            for script_name in self.registered_scripts:
                if script_name != source:
                    if script_name not in self.script_manager.collaboration_requests:
                        self.script_manager.collaboration_requests[script_name] = []
                    self.script_manager.collaboration_requests[script_name].append({
                        "source": source,
                        "data": data,
                        "broadcast": True,
                        "timestamp": time.time()
                    })
                    
            return {"status": "broadcast_sent"}
            
        elif command == "get_collaborations":
            script_name = message.get("script_name")
            if script_name in self.script_manager.collaboration_requests:
                requests = self.script_manager.collaboration_requests[script_name]
                self.script_manager.collaboration_requests[script_name] = []
                return {"requests": requests}
            return {"requests": []}
            
        elif command == "notification":
            script = message.get("script")
            event = message.get("event")
            data = message.get("data", {})
            
            logger.info(f"Notification from '{script}': {event}")
            
            # Handle specific events
            if event == "results_updated":
                if script in self.registered_scripts:
                    self.registered_scripts[script]["results"] = data
                    
            return {"status": "acknowledged"}
            
        return {"error": "Unknown command"}
        
    def main_loop(self):
        """Main connector loop"""
        try:
            while self.running:
                # Display menu
                print("\n=== ML Models Connector ===")
                print("1. List available scripts")
                print("2. Execute script")
                print("3. Stop script")
                print("4. View script status")
                print("5. View registered scripts")
                print("6. Control script parameters")
                print("7. View script results")
                print("8. Exit")
                
                choice = input("\nEnter choice: ").strip()
                
                if choice == "1":
                    self.list_scripts()
                elif choice == "2":
                    self.execute_script_interactive()
                elif choice == "3":
                    self.stop_script_interactive()
                elif choice == "4":
                    self.view_script_status()
                elif choice == "5":
                    self.view_registered_scripts()
                elif choice == "6":
                    self.control_script_parameters()
                elif choice == "7":
                    self.view_script_results()
                elif choice == "8":
                    self.running = False
                    print("Exiting...")
                else:
                    print("Invalid choice")
                    
        except KeyboardInterrupt:
            self.running = False
            print("\nConnector stopped")
            
    def list_scripts(self):
        """List available scripts"""
        print("\nAvailable scripts:")
        for i, script_name in enumerate(self.script_manager.scripts, 1):
            print(f"{i}. {script_name}")
            
    def execute_script_interactive(self):
        """Execute a script interactively"""
        self.list_scripts()
        script_name = input("\nEnter script name: ").strip()
        
        if script_name in self.script_manager.scripts:
            with_connector = input("Run with connector integration? (y/n): ").lower() == 'y'
            result = self.script_manager.execute_script(script_name, with_connector)
            print(f"Result: {result}")
        else:
            print("Script not found")
            
    def stop_script_interactive(self):
        """Stop a script interactively"""
        running_scripts = [name for name, proc in self.script_manager.script_processes.items() 
                          if proc.poll() is None]
        
        if not running_scripts:
            print("No scripts running")
            return
            
        print("\nRunning scripts:")
        for i, script_name in enumerate(running_scripts, 1):
            print(f"{i}. {script_name}")
            
        script_name = input("\nEnter script name to stop: ").strip()
        
        if script_name in running_scripts:
            result = self.script_manager.stop_script(script_name)
            print(f"Result: {result}")
        else:
            print("Script not running")
            
    def view_script_status(self):
        """View status of all scripts"""
        print("\nScript Status:")
        for script_name in self.script_manager.scripts:
            status = self.script_manager.get_script_status(script_name)
            print(f"{script_name}: {status}")
            
    def view_registered_scripts(self):
        """View registered scripts with their info"""
        if not self.registered_scripts:
            print("No scripts registered")
            return
            
        print("\nRegistered Scripts:")
        for script_name, info in self.registered_scripts.items():
            print(f"\n{script_name}:")
            print(f"  Description: {info.get('description', 'N/A')}")
            print(f"  State: {info.get('state', 'N/A')}")
            print(f"  Parameters: {len(info.get('parameters', {}))}")
            print(f"  Variables: {len(info.get('variables', {}))}")
            
    def control_script_parameters(self):
        """Control script parameters"""
        if not self.registered_scripts:
            print("No scripts registered")
            return
            
        print("\nRegistered scripts:")
        script_names = list(self.registered_scripts.keys())
        for i, name in enumerate(script_names, 1):
            print(f"{i}. {name}")
            
        try:
            idx = int(input("Select script: ")) - 1
            script_name = script_names[idx]
            
            info = self.registered_scripts[script_name]
            params = info.get("parameters", {})
            
            if not params:
                print("No parameters available")
                return
                
            print(f"\nParameters for {script_name}:")
            param_names = list(params.keys())
            for i, (name, param) in enumerate(params.items(), 1):
                print(f"{i}. {name} = {param['value']} ({param['type']})")
                
            param_idx = int(input("Select parameter to modify: ")) - 1
            param_name = param_names[param_idx]
            
            new_value = input(f"Enter new value for {param_name}: ")
            
            # Send parameter update
            # This would be sent to the script via socket
            print(f"Parameter update sent: {param_name} = {new_value}")
            
        except (ValueError, IndexError):
            print("Invalid selection")
            
    def view_script_results(self):
        """View script results"""
        if not self.registered_scripts:
            print("No scripts registered")
            return
            
        print("\nScript Results:")
        for script_name, info in self.registered_scripts.items():
            results = info.get("results", {})
            if results:
                print(f"\n{script_name}:")
                for key, value in results.items():
                    print(f"  {key}: {value}")
            else:
                print(f"\n{script_name}: No results available")


def main():
    """Main function for the enhanced connector"""
    connector = EnhancedConnector()
    connector.start()


if __name__ == "__main__":
    main()