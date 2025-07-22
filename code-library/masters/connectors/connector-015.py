
import os
import logging
import sys
import json
import socket
import threading
from pathlib import Path
import time
from typing import Dict, Any, Optional

# Import script interface and wrappers
try:
    from script_interface import interface, handle_connector_command
    from script_wrappers import wrappers
    INTERFACE_AVAILABLE = True
except ImportError:
    INTERFACE_AVAILABLE = False

# Import enhanced integration
try:
    from enhanced_integration import EnhancedConnector as EnhancedIntegration, EventType, Event
    ENHANCED_INTEGRATION_AVAILABLE = True
except ImportError:
    ENHANCED_INTEGRATION_AVAILABLE = False

# --- Configuration ---
LOG_FILE = "connector.log"
CONNECTOR_PORT = 10051  # Different from hivemind connector

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
    """Enhanced connector with script control capabilities"""
    
    def __init__(self):
        self.running = True
        self.socket = None
        self.scripts_available = INTERFACE_AVAILABLE
        self.enhanced_integration = None
        self.event_handlers = {}
        self.pending_results = {}
        
        # Initialize enhanced integration if available
        if ENHANCED_INTEGRATION_AVAILABLE:
            self.enhanced_integration = EnhancedIntegration()
            self._setup_event_handlers()
            self._load_all_scripts()
    
    def _setup_event_handlers(self):
        """Setup event handlers for enhanced integration"""
        if self.enhanced_integration:
            # Subscribe to various events
            self.enhanced_integration.event_bus.subscribe(
                EventType.PROGRESS, self._handle_progress_event, 'connector_server'
            )
            self.enhanced_integration.event_bus.subscribe(
                EventType.STATUS, self._handle_status_event, 'connector_server'
            )
            self.enhanced_integration.event_bus.subscribe(
                EventType.ERROR, self._handle_error_event, 'connector_server'
            )
            self.enhanced_integration.event_bus.subscribe(
                EventType.DATA, self._handle_data_event, 'connector_server'
            )
    
    def _load_all_scripts(self):
        """Load all Python scripts with enhanced integration"""
        if self.enhanced_integration:
            script_dir = Path('.')
            for script_file in script_dir.glob('*.py'):
                if script_file.name not in ['connector.py', 'hivemind_connector.py', 
                                          'enhanced_integration.py', 'setup.py']:
                    try:
                        self.enhanced_integration.load_script(str(script_file))
                        logger.info(f"Loaded script with enhanced integration: {script_file.name}")
                    except Exception as e:
                        logger.error(f"Failed to load {script_file.name}: {e}")
    
    def _handle_progress_event(self, event: Event):
        """Handle progress events from scripts"""
        logger.info(f"Progress from {event.source}: {event.data.get('progress', 0) * 100:.1f}% - {event.data.get('message', '')}")
        # Store for client retrieval if needed
        self.event_handlers[event.source] = event.data
    
    def _handle_status_event(self, event: Event):
        """Handle status events from scripts"""
        logger.info(f"Status from {event.source}: {event.data}")
    
    def _handle_error_event(self, event: Event):
        """Handle error events from scripts"""
        logger.error(f"Error from {event.source}: {event.data}")
    
    def _handle_data_event(self, event: Event):
        """Handle data events from scripts"""
        logger.debug(f"Data from {event.source}: {event.data}")
        
    def start_server(self):
        """Start socket server for remote control"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('localhost', CONNECTOR_PORT))
            self.socket.listen(5)
            
            logger.info(f"Connector server listening on port {CONNECTOR_PORT}")
            
            while self.running:
                try:
                    conn, addr = self.socket.accept()
                    threading.Thread(target=self.handle_client, args=(conn,)).start()
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Server error: {e}")
            
    def handle_client(self, conn):
        """Handle client connection"""
        try:
            data = conn.recv(4096).decode()
            command = json.loads(data)
            
            response = self.process_command(command)
            conn.send(json.dumps(response).encode())
            
        except Exception as e:
            logger.error(f"Client handling error: {e}")
            response = {'error': str(e)}
            conn.send(json.dumps(response).encode())
        finally:
            conn.close()
            
    def process_command(self, command):
        """Process incoming commands"""
        cmd_type = command.get('type')
        
        # Enhanced integration commands
        if cmd_type == 'enhanced_control' and self.enhanced_integration:
            return self._process_enhanced_command(command)
        
        # Legacy script control
        elif cmd_type == 'script_control' and self.scripts_available:
            # Delegate to script interface
            return handle_connector_command(command)
            
        elif cmd_type == 'wrapper_function' and self.scripts_available:
            # Execute wrapper function
            func_name = command.get('function')
            args = command.get('args', {})
            
            if hasattr(wrappers, func_name):
                func = getattr(wrappers, func_name)
                return func(**args)
            else:
                return {'error': f'Function {func_name} not found'}
                
        elif cmd_type == 'status':
            return self.get_status()
            
        elif cmd_type == 'list_files':
            return self.list_files()
            
        else:
            return {'error': 'Unknown command type'}
    
    def _process_enhanced_command(self, command):
        """Process enhanced integration commands"""
        action = command.get('action')
        
        try:
            if action == 'call_function':
                # Call function with full bidirectional support
                script = command.get('script')
                function = command.get('function')
                args = command.get('args', [])
                kwargs = command.get('kwargs', {})
                
                result = self.enhanced_integration.call_function(
                    script, function, *args, **kwargs
                )
                return {'result': result, 'status': 'success'}
            
            elif action == 'set_parameter':
                # Set parameter in script
                script = command.get('script')
                parameter = command.get('parameter')
                value = command.get('value')
                
                self.enhanced_integration.set_parameter(script, parameter, value)
                return {'status': 'success', 'message': f'Parameter {parameter} set to {value}'}
            
            elif action == 'get_script_info':
                # Get information about a script
                script = command.get('script')
                info = self.enhanced_integration.get_script_info(script)
                return {'info': info, 'status': 'success'}
            
            elif action == 'get_all_scripts':
                # Get information about all scripts
                info = self.enhanced_integration.get_all_scripts_info()
                return {'scripts': info, 'status': 'success'}
            
            elif action == 'get_events':
                # Get pending events
                return {'events': self.event_handlers, 'status': 'success'}
            
            elif action == 'publish_event':
                # Publish an event to scripts
                event_type = command.get('event_type', 'info')
                target = command.get('target')
                data = command.get('data', {})
                
                try:
                    event_type_enum = EventType(event_type)
                except ValueError:
                    event_type_enum = EventType.INFO
                
                self.enhanced_integration.event_bus.publish(Event(
                    type=event_type_enum,
                    source='connector_client',
                    target=target,
                    data=data
                ))
                return {'status': 'success', 'message': 'Event published'}
            
            else:
                return {'error': f'Unknown action: {action}'}
                
        except Exception as e:
            logger.error(f"Enhanced command error: {e}")
            return {'error': str(e), 'status': 'error'}
            
    def get_status(self):
        """Get connector status"""
        status = {
            'status': 'active',
            'interface_available': self.scripts_available,
            'enhanced_integration_available': ENHANCED_INTEGRATION_AVAILABLE,
            'directory': os.getcwd(),
            'scripts': interface.list_scripts() if self.scripts_available else []
        }
        
        if self.enhanced_integration:
            status['enhanced_scripts'] = list(self.enhanced_integration.controllers.keys())
            status['shared_state'] = self.enhanced_integration.shared_state
            
        return status
        
    def list_files(self):
        """List files in directory"""
        try:
            files = []
            for item in Path('.').iterdir():
                files.append({
                    'name': item.name,
                    'type': 'file' if item.is_file() else 'directory',
                    'size': item.stat().st_size if item.is_file() else None
                })
            return {'files': files}
        except Exception as e:
            return {'error': str(e)}


def main():
    """Main function for the connector script."""
    logger.info(f"--- Enhanced Connector Script Initialized in {os.getcwd()} ---")
    logger.info(f"This script provides full control over PyTorch production scripts.")
    
    if INTERFACE_AVAILABLE:
        logger.info("Script interface loaded successfully")
        logger.info("Available scripts:")
        for script in interface.list_scripts():
            logger.info(f"  - {script['name']}: {script['description']}")
    else:
        logger.warning("Script interface not available - running in basic mode")
    
    if ENHANCED_INTEGRATION_AVAILABLE:
        logger.info("Enhanced integration system available - bidirectional communication enabled")
    else:
        logger.info("Enhanced integration not available - using legacy mode")
    
    # Start server if requested
    if '--server' in sys.argv:
        connector = EnhancedConnector()
        logger.info("Starting connector server...")
        try:
            connector.start_server()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        finally:
            if connector.enhanced_integration:
                connector.enhanced_integration.stop()
    else:
        # List files in the current directory
        try:
            files = os.listdir()
            if files:
                logger.info("Files in this directory:")
                for file in files:
                    logger.info(f"- {file}")
            else:
                logger.info("No files found in this directory.")
        except OSError as e:
            logger.error(f"Error listing files: {e}")

    logger.info("Connector script finished.")

if __name__ == "__main__":
    main()
