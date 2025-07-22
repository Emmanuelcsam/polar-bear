
import os
import logging
import sys
import json
import threading
import socket
from pathlib import Path
from typing import Dict, Any, Optional

# Import the script interface
try:
    from script_interface import get_interface
except ImportError:
    # Create a basic interface if the module doesn't exist yet
    class BasicInterface:
        def __init__(self):
            self.scripts = {}
    def get_interface():
        return BasicInterface()

# --- Configuration ---
LOG_FILE = "connector.log"
CONNECTOR_PORT = 10089

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


class EnhancedConnector:
    """Enhanced connector with full script control capabilities"""
    
    def __init__(self):
        self.interface = get_interface()
        self.scripts_loaded = False
        self.socket_server = None
        self.running = True
        self.current_dir = Path.cwd()
        
    def load_all_scripts(self):
        """Load all Python scripts in the directory"""
        logger.info("Loading all scripts in directory...")
        
        # Load specific scripts we know about
        scripts_to_load = {
            'intensity_reader': 'intensity_reader.py',
            'random_pixel_generator': 'random_pixel_generator.py'
        }
        
        for script_name, filename in scripts_to_load.items():
            script_path = self.current_dir / filename
            if script_path.exists():
                try:
                    self.interface.load_script(script_name, str(script_path))
                    logger.info(f"Loaded script: {script_name} from {filename}")
                except Exception as e:
                    logger.error(f"Failed to load {script_name}: {e}")
                    
        # Also scan for any other Python files
        for path in self.current_dir.glob("*.py"):
            if path.name not in ['connector.py', 'hivemind_connector.py', 'script_interface.py'] and path.name not in scripts_to_load.values():
                script_name = path.stem
                try:
                    self.interface.load_script(script_name, str(path))
                    logger.info(f"Loaded additional script: {script_name}")
                except Exception as e:
                    logger.error(f"Failed to load {script_name}: {e}")
                    
        self.scripts_loaded = True
        
    def process_control_command(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process control commands for scripts"""
        cmd_type = command_data.get('type')
        
        if cmd_type == 'list_scripts':
            return {
                'status': 'success',
                'scripts': self.interface.list_scripts()
            }
            
        elif cmd_type == 'control_script':
            return self.interface.process_command(command_data)
            
        elif cmd_type == 'execute_intensity_reader':
            # Special handling for intensity reader
            params = command_data.get('params', {})
            path = params.get('path')
            threshold = params.get('threshold')
            
            results = []
            def collect_results(value):
                results.append(value)
                
            try:
                self.interface.execute_function('intensity_reader', 'read_intensity', 
                                              path, threshold, callback=collect_results)
                return {
                    'status': 'success',
                    'results': results
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'error': str(e)
                }
                
        elif cmd_type == 'control_pixel_generator':
            # Special handling for pixel generator
            params = command_data.get('params', {})
            action = params.get('action')
            
            if action == 'get_params':
                # Get current parameters
                try:
                    module = self.interface.script_modules.get('random_pixel_generator')
                    return {
                        'status': 'success',
                        'params': {
                            'min_val': getattr(module, 'min_val', 0),
                            'max_val': getattr(module, 'max_val', 255),
                            'delay': getattr(module, 'delay', 0)
                        }
                    }
                except Exception as e:
                    return {'status': 'error', 'error': str(e)}
                    
            elif action == 'set_params':
                # Set new parameters
                try:
                    new_params = params.get('values', {})
                    for key, value in new_params.items():
                        self.interface.set_variable('random_pixel_generator', key, value)
                    return {'status': 'success'}
                except Exception as e:
                    return {'status': 'error', 'error': str(e)}
                    
        else:
            return {
                'status': 'error',
                'error': f'Unknown command type: {cmd_type}'
            }
            
    def start_socket_server(self):
        """Start socket server for remote control"""
        try:
            self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket_server.bind(('localhost', CONNECTOR_PORT))
            self.socket_server.listen(5)
            logger.info(f"Socket server started on port {CONNECTOR_PORT}")
            
            while self.running:
                try:
                    conn, addr = self.socket_server.accept()
                    threading.Thread(target=self.handle_connection, args=(conn,)).start()
                except Exception as e:
                    if self.running:
                        logger.error(f"Socket server error: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to start socket server: {e}")
            
    def handle_connection(self, conn):
        """Handle incoming socket connection"""
        try:
            data = conn.recv(4096).decode()
            command_data = json.loads(data)
            
            response = self.process_control_command(command_data)
            conn.send(json.dumps(response).encode())
            
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
            error_response = {'status': 'error', 'error': str(e)}
            try:
                conn.send(json.dumps(error_response).encode())
            except:
                pass
        finally:
            conn.close()
            
    def run(self):
        """Run the enhanced connector"""
        logger.info("Starting Enhanced Connector...")
        
        # Load all scripts
        self.load_all_scripts()
        
        # Show loaded scripts
        scripts = self.interface.list_scripts()
        logger.info(f"Loaded {len(scripts)} scripts:")
        for name, info in scripts.items():
            logger.info(f"  - {name}: {info['functions']} functions")
            
        # Start socket server in background
        server_thread = threading.Thread(target=self.start_socket_server)
        server_thread.daemon = True
        server_thread.start()
        
        logger.info("Connector ready. Scripts can now be controlled remotely.")
        logger.info(f"Connect to localhost:{CONNECTOR_PORT} to send commands")
        
        # Keep running
        try:
            while self.running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down connector...")
            self.running = False
            

def main():
    """Main function for the connector script."""
    connector = EnhancedConnector()
    connector.run()

if __name__ == "__main__":
    main()
