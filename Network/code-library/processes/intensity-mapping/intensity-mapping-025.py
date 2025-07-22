
import torch
import torch.nn.functional as F
import time
import os
from shared_state import SharedState
from connector import SidecarConnector, find_free_port

def main_loop(state):
    """The main processing loop for the texture synthesizer."""
    # Initial setup
    target_features_file = state.get_parameter('target_features_file', 'features.pt')
    if not os.path.exists(target_features_file):
        print(f"Target features file not found: {target_features_file}. Waiting...")
        while not os.path.exists(target_features_file) and state.get('running', True):
            time.sleep(1)

    if not state.get('running', True): return

    ref = torch.load(target_features_file)
    canvas = torch.randn_like(ref, requires_grad=True)

    while state.get('running', True):
        try:
            learning_rate = state.get_parameter('learning_rate', 0.01)
            
            # We need to manually implement the optimization step to use the state
            loss = F.mse_loss(canvas, ref)
            loss.backward()
            
            with torch.no_grad():
                canvas -= canvas.grad * learning_rate
                canvas.grad.zero_()

            state.set_parameter('current_loss', loss.item())
            # print(f"Current Loss: {loss.item()}")

            # Save the synthesized texture periodically
            if state.get_parameter('save_intermediate', True):
                torch.save(canvas.detach(), state.get_parameter('output_file', 'generated.pt'))

        except Exception as e:
            print(f"Error in texture synth loop: {e}")

        time.sleep(state.get_parameter('interval_seconds', 0.1))

if __name__ == "__main__":
    script_name = os.path.basename(__file__)
    shared_state = SharedState()
    
    # --- Default Parameters ---
    shared_state.set_parameter('target_features_file', 'features.pt')
    shared_state.set_parameter('output_file', 'generated.pt')
    shared_state.set_parameter('learning_rate', 0.01)
    shared_state.set_parameter('current_loss', None)
    shared_state.set_parameter('save_intermediate', True)
    shared_state.set_parameter('interval_seconds', 0.1)
    
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
