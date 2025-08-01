
import torch
import torchvision.utils as vutils
import time
import os
from shared_state import SharedState
from connector import SidecarConnector, find_free_port

def main_loop(state):
    """The main processing loop for the image saver."""
    while state.get('running', True):
        input_file = state.get_parameter('input_file', 'generated.pt')
        output_file = state.get_parameter('output_file', 'output.png')
        
        try:
            if os.path.exists(input_file):
                img = torch.load(input_file)
                vutils.save_image((img + 1) / 2, output_file)
                # print(f"Saved image to {output_file}")
            else:
                # print(f"Generated file not found: {input_file}")
                pass
        except Exception as e:
            print(f"Error in image saver loop: {e}")

        time.sleep(state.get_parameter('interval_seconds', 2.0))

if __name__ == "__main__":
    script_name = os.path.basename(__file__)
    shared_state = SharedState()
    
    # --- Default Parameters ---
    shared_state.set_parameter('input_file', 'generated.pt')
    shared_state.set_parameter('output_file', 'output.png')
    shared_state.set_parameter('interval_seconds', 2.0)
    
    # --- Sidecar Connector ---
    port = find_free_port()
    sidecar = SidecarConnector(script_name, shared_state, port)
    sidecar.start()
    
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
