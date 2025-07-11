
import torch
import torch.nn as nn
import time
import os
from shared_state import SharedState
from connector import SidecarConnector, find_free_port

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

def main_loop(state):
    """The main processing loop for the neural generator."""
    gen = G()
    # You might want to load a checkpoint here if available
    # gen.load_state_dict(torch.load(state.get_parameter('model_path', 'generator.pth')))
    
    while state.get('running', True):
        noise_file = state.get_parameter('input_file', 'noise.pt')
        output_file = state.get_parameter('output_file', 'generated.pt')
        
        try:
            if os.path.exists(noise_file):
                noise = torch.load(noise_file)
                img = gen(noise)
                torch.save(img, output_file)
                # print(f"Generated new image and saved to {output_file}")
            else:
                # print(f"Noise file not found: {noise_file}")
                pass
        except Exception as e:
            print(f"Error in neural generator loop: {e}")

        time.sleep(state.get_parameter('interval_seconds', 1.0))

if __name__ == "__main__":
    script_name = os.path.basename(__file__)
    shared_state = SharedState()
    
    # --- Default Parameters ---
    shared_state.set_parameter('input_file', 'noise.pt')
    shared_state.set_parameter('output_file', 'generated.pt')
    shared_state.set_parameter('interval_seconds', 1.0)
    
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
