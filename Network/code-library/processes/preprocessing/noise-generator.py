
import torch
import time
import os
from shared_state import SharedState
from connector import SidecarConnector, find_free_port

def main_loop(state):
    """The main processing loop for the noise generator."""
    while state.get('running', True):
        # Read parameters from shared state on each iteration
        output_file = state.get_parameter('output_file', 'noise.pt')
        size_x = state.get_parameter('size_x', 256)
        size_y = state.get_parameter('size_y', 256)
        
        try:
            noise = torch.randn(1, 3, size_x, size_y)
            torch.save(noise, output_file)
            # print(f"Generated new noise tensor and saved to {output_file}")
        except Exception as e:
            print(f"Error in noise generator loop: {e}")

        time.sleep(state.get_parameter('interval_seconds', 1.0))

if __name__ == "__main__":
    script_name = os.path.basename(__file__)
    shared_state = SharedState()
    
    # --- Default Parameters ---
    shared_state.set_parameter('output_file', 'noise.pt')
    shared_state.set_parameter('size_x', 256)
    shared_state.set_parameter('size_y', 256)
    shared_state.set_parameter('interval_seconds', 1.0)
    
    # --- Sidecar Connector ---
    port = find_free_port()
    sidecar = SidecarConnector(script_name, shared_state, port)
    sidecar.start()
    
    # Write port file for hivemind discovery
    port_file = f".{script_name}.port"
    with open(port_file, "w") as f:
        f.write(str(port))

    try:
        main_loop(shared_state)
    except KeyboardInterrupt:
        print("Script interrupted by user.")
    finally:
        print("Shutting down...")
        shared_state.set('running', False)
        sidecar.stop()
        sidecar.join()
        if os.path.exists(port_file):
            os.remove(port_file)
        print("Shutdown complete.")
