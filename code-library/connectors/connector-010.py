
import socket
import threading
import json
import logging
import sys
import os

class SidecarConnector(threading.Thread):
    """
    A sidecar connector that runs in a separate thread within a main script.
    It listens for commands on a dedicated socket to allow external inspection
    and modification of the script's state.
    """
    def __init__(self, script_name, shared_state, port):
        super().__init__()
        self.daemon = True  # Ensure thread exits when main script exits
        self.script_name = script_name
        self.shared_state = shared_state
        self.port = port
        self.running = True
        self.server_socket = None

        # Basic logging for the sidecar
        self.logger = logging.getLogger(f"Sidecar_{self.script_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(self):
        """The main loop for the sidecar thread."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('localhost', self.port))
            self.server_socket.listen(1)
            self.logger.info(f"Sidecar for '{self.script_name}' listening on port {self.port}")

            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    self.handle_connection(conn)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        self.logger.error(f"Error in accept loop: {e}")
        except Exception as e:
            self.logger.error(f"Sidecar socket could not be created: {e}", exc_info=True)
        finally:
            if self.server_socket:
                self.server_socket.close()
            self.logger.info("Sidecar shut down.")

    def handle_connection(self, conn):
        """Handle an incoming connection from a controller."""
        try:
            with conn:
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    return
                message = json.loads(data)
                response = self.process_message(message)
                conn.sendall(json.dumps(response).encode('utf-8'))
        except json.JSONDecodeError:
            self.logger.error("Received invalid JSON.")
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")

    def process_message(self, message):
        """Process the command received from the controller."""
        command = message.get('command')
        
        if command == 'status':
            return {'status': 'running', 'script': self.script_name, 'params': self.shared_state.get_all_parameters()}
        
        elif command == 'get_param':
            key = message.get('key')
            return {'value': self.shared_state.get_parameter(key)}
            
        elif command == 'set_param':
            key = message.get('key')
            value = message.get('value')
            self.shared_state.set_parameter(key, value)
            return {'status': 'success', 'key': key, 'new_value': value}
            
        elif command == 'stop':
            self.shared_state.set('running', False)
            self.stop()
            return {'status': 'stopping'}
            
        else:
            return {'error': 'Unknown command'}

    def stop(self):
        """Stops the sidecar thread."""
        self.running = False
        # Non-blocking way to interrupt the accept() call
        if self.server_socket:
            try:
                # Create a dummy connection to unblock the accept call
                dummy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                dummy_socket.connect(('localhost', self.port))
                dummy_socket.close()
            except Exception as e:
                self.logger.warning(f"Could not unblock server socket cleanly: {e}")
        self.logger.info("Stop signal received.")

def find_free_port():
    """Finds a free port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
