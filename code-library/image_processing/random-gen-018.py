
import torch
from PIL import Image
import torchvision.transforms as T
import time
import os
from shared_state import SharedState
from connector import SidecarConnector, find_free_port

def main_loop(state):
    """The main processing loop for the pixel guide."""
    transform = T.ToTensor()
    
    while state.get('running', True):
        generated_file = state.get_parameter('generated_file', 'generated.pt')
        ref_file = state.get_parameter('ref_file', 'ref.jpg')
        output_file = state.get_parameter('output_file', 'generated.pt')
        mix_factor = state.get_parameter('mix_factor', 0.3)

        try:
            if os.path.exists(generated_file) and os.path.exists(ref_file):
                gen = torch.load(generated_file).squeeze()
                ref = transform(Image.open(ref_file).convert('RGB'))
                
                # Ensure tensors are compatible
                if gen.shape != ref.shape:
                    # A simple resize strategy. More complex alignment might be needed.
                    ref = T.Resize(gen.shape[1:])(ref)

                guided = gen * (1.0 - mix_factor) + ref * mix_factor
                torch.save(guided.unsqueeze(0), output_file)
                # print(f"Applied pixel guidance and saved to {output_file}")
            else:
                # print("Waiting for generated and reference files...")
                pass
        except Exception as e:
            print(f"Error in pixel guide loop: {e}")

        time.sleep(state.get_parameter('interval_seconds', 1.0))

if __name__ == "__main__":
    script_name = os.path.basename(__file__)
    shared_state = SharedState()
    
    # --- Default Parameters ---
    shared_state.set_parameter('generated_file', 'generated.pt')
    shared_state.set_parameter('ref_file', 'ref.jpg')
    shared_state.set_parameter('output_file', 'generated.pt') # Overwrites the generated file
    shared_state.set_parameter('mix_factor', 0.3)
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
