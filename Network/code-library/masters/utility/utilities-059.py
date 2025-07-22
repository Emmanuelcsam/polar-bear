
import torch
import torch.optim as optim
import torchvision.models as models
import time
import os
from shared_state import SharedState
from connector import SidecarConnector, find_free_port

def main_loop(state):
    """The main processing loop for the style optimizer."""
    try:
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        
        # Initial setup
        target_features_file = state.get_parameter('target_features_file', 'features.pt')
        if not os.path.exists(target_features_file):
            print(f"Target features file not found: {target_features_file}. Waiting...")
            while not os.path.exists(target_features_file) and state.get('running', True):
                time.sleep(1)
        
        if not state.get('running', True): return

        target = torch.load(target_features_file)
        img = torch.randn(1, 3, 256, 256, requires_grad=True)
        opt = optim.Adam([img], lr=state.get_parameter('learning_rate', 0.01))

        while state.get('running', True):
            # Update learning rate from shared state
            for g in opt.param_groups:
                g['lr'] = state.get_parameter('learning_rate', 0.01)

            opt.zero_grad()
            features = vgg(img)
            loss = ((features - target)**2).mean()
            loss.backward()
            opt.step()
            
            state.set_parameter('current_loss', loss.item())
            # print(f"Current Loss: {loss.item()}")

            # Save the optimized image periodically
            if state.get_parameter('save_intermediate', True):
                torch.save(img.detach(), state.get_parameter('output_file', 'generated.pt'))

            time.sleep(state.get_parameter('interval_seconds', 0.1))
    except Exception as e:
        print(f"CRITICAL ERROR in style-optimizer main_loop: {e}")
        state.set('running', False)


if __name__ == "__main__":
    try:
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

        main_loop(shared_state)

    except Exception as e:
        import traceback
        print(f"--- CRITICAL ERROR IN {os.path.basename(__file__)} ---")
        print(str(e))
        print(traceback.format_exc())
        print("----------------------------------------------------")
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        print("Shutting down...")
        shared_state.set('running', False)
        if 'sidecar' in locals() and sidecar.is_alive():
            sidecar.stop()
            sidecar.join()
        if 'port_file' in locals() and os.path.exists(port_file):
            os.remove(port_file)
        print("Shutdown complete.")
