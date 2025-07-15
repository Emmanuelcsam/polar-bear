
import os
import logging
import sys
import json
import socket
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Import the script interface
try:
    from script_interface import get_script_manager, get_connector_interface
except ImportError:
    print("Warning: script_interface module not found. Running in basic mode.")
    get_script_manager = None
    get_connector_interface = None

# --- Configuration ---
LOG_FILE = "connector.log"
CONFIG_FILE = "config.json"
CONNECTOR_PORT = 12000  # Default port for connector communication

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


class AnalysisConnector:
    """Enhanced connector with full script control capabilities"""
    
    def __init__(self, port: int = CONNECTOR_PORT):
        self.port = port
        self.running = True
        self.socket = None
        self.script_manager = None
        self.connector_interface = None
        self.clients = []
        self.lock = threading.Lock()
        
        # Initialize script management if available
        if get_script_manager:
            self.script_manager = get_script_manager()
            self.connector_interface = get_connector_interface()
            logger.info("Script management interface initialized")
        else:
            logger.warning("Running without script management interface")
    
    def start(self):
        """Start the connector service"""
        logger.info(f"Starting Analysis Connector on port {self.port}")
        
        # Start socket server
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_scripts)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Analysis Connector started successfully")
    
    def _run_server(self):
        """Run the socket server for external connections"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('localhost', self.port))
            self.socket.listen(5)
            
            logger.info(f"Listening for connections on port {self.port}")
            
            while self.running:
                try:
                    client_socket, address = self.socket.accept()
                    logger.info(f"New connection from {address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        logger.error(f"Server error: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle a client connection"""
        with self.lock:
            self.clients.append((client_socket, address))
        
        try:
            while self.running:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                # Process command
                try:
                    command = json.loads(data.decode())
                    response = self._process_command(command)
                    
                    # Send response
                    client_socket.send(json.dumps(response).encode())
                    
                except json.JSONDecodeError:
                    error_response = {"error": "Invalid JSON"}
                    client_socket.send(json.dumps(error_response).encode())
                    
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            # Remove client
            with self.lock:
                self.clients = [(s, a) for s, a in self.clients if s != client_socket]
            client_socket.close()
            logger.info(f"Client {address} disconnected")
    
    def _process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Process a command received from a client"""
        cmd_type = command.get("command")
        
        logger.info(f"Processing command: {cmd_type}")
        
        # Handle connector-specific commands
        if cmd_type == "status":
            return self._get_status()
        
        elif cmd_type == "list_files":
            return self._list_files()
        
        elif cmd_type == "read_config":
            return self._read_config()
        
        elif cmd_type == "update_config":
            config_data = command.get("config", {})
            return self._update_config(config_data)
        
        # Delegate script commands to interface
        elif self.connector_interface:
            return self.connector_interface.handle_command(command)
        
        else:
            return {"error": "Unknown command or script interface not available"}
    
    def _get_status(self) -> Dict[str, Any]:
        """Get connector status"""
        status = {
            "status": "active",
            "port": self.port,
            "clients": len(self.clients),
            "script_manager": self.script_manager is not None,
            "directory": os.getcwd()
        }
        
        if self.script_manager:
            scripts = self.script_manager.get_all_scripts()
            status["scripts"] = {
                "total": len(scripts),
                "running": sum(1 for s in scripts if s.status == "running"),
                "completed": sum(1 for s in scripts if s.status == "completed"),
                "failed": sum(1 for s in scripts if s.status == "failed")
            }
        
        return status
    
    def _list_files(self) -> Dict[str, Any]:
        """List files in the current directory"""
        try:
            files = []
            for item in Path(".").iterdir():
                files.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
            return {"files": files}
        except Exception as e:
            return {"error": f"Failed to list files: {e}"}
    
    def _read_config(self) -> Dict[str, Any]:
        """Read the configuration file"""
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return {"config": config}
        except Exception as e:
            return {"error": f"Failed to read config: {e}"}
    
    def _update_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update the configuration file"""
        try:
            # Read existing config
            existing_config = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    existing_config = json.load(f)
            
            # Merge with new data
            existing_config.update(config_data)
            
            # Save
            with open(CONFIG_FILE, 'w') as f:
                json.dump(existing_config, f, indent=4)
            
            # Reload in script manager if available
            if self.script_manager:
                self.script_manager.load_config()
            
            return {"status": "config_updated"}
        except Exception as e:
            return {"error": f"Failed to update config: {e}"}
    
    def _monitor_scripts(self):
        """Monitor script execution and results"""
        while self.running:
            try:
                if self.script_manager:
                    # Get execution results
                    results = self.script_manager.get_execution_results()
                    
                    # Broadcast results to connected clients
                    if results and self.clients:
                        message = {
                            "type": "execution_results",
                            "results": results
                        }
                        self._broadcast(message)
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
    
    def _broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        message_data = json.dumps(message).encode()
        
        with self.lock:
            for client_socket, address in self.clients[:]:
                try:
                    client_socket.send(message_data)
                except Exception as e:
                    logger.error(f"Failed to send to {address}: {e}")
    
    def stop(self):
        """Stop the connector"""
        logger.info("Stopping Analysis Connector")
        self.running = False
        
        # Close server socket
        if self.socket:
            self.socket.close()
        
        # Close all client connections
        with self.lock:
            for client_socket, _ in self.clients:
                try:
                    client_socket.close()
                except:
                    pass
        
        # Stop script manager
        if self.script_manager:
            self.script_manager.stop()
        
        logger.info("Analysis Connector stopped")


def main():
    """Main function for the connector script."""
    logger.info(f"--- Analysis Connector Initialized in {os.getcwd()} ---")
    logger.info(f"This enhanced connector provides full control over analysis/reporting scripts")
    
    # Parse command line arguments
    port = CONNECTOR_PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    # Create and start connector
    connector = AnalysisConnector(port)
    connector.start()
    
    # Run until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        connector.stop()
        logger.info("Connector shutdown complete")


if __name__ == "__main__":
    main()
