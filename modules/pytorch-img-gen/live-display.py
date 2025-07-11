
import torch
import cv2
import numpy as np
import time
import os
from shared_state import SharedState
from connector import SidecarConnector, find_free_port

def main_loop(state):
    """The main processing loop for the live display."""
    cv2.namedWindow('Generated', cv2.WINDOW_NORMAL)

    while state.get('running', True):
        input_file = state.get_parameter('input_file', 'generated.pt')
        
        try:
            if os.path.exists(input_file):
                img = torch.load(input_file).squeeze().permute(1, 2, 0)
                img = ((img + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
                cv2.imshow('Generated', img)
            else:
                # Create a black screen if no image is found
                black_screen = np.zeros((256, 256, 3), np.uint8)
                cv2.imshow('Generated', black_screen)

        except Exception as e:
            print(f"Error in live display loop: {e}")
            # Display error on screen
            error_screen = np.zeros((256, 256, 3), np.uint8)
            cv2.putText(error_screen, "Error", (100, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Generated', error_screen)

        # Wait for a key press for a short duration.
        # This is crucial to allow the window to update and to check for the 'q' key.
        if cv2.waitKey(100) & 0xFF == ord('q'):
            state.set('running', False) # Signal shutdown
        
        # Also check the state flag in case of remote stop
        if not state.get('running'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    script_name = os.path.basename(__file__)
    shared_state = SharedState()
    
    # --- Default Parameters ---
    shared_state.set_parameter('input_file', 'generated.pt')
    
    # --- Sidecar Connector ---
    port = find_free_port()
    sidecar = SidecarConnector(script_name, shared_state, port)
    sidecar.start()
    
    port_file = f".{script_name}.port"
    with open(port_file, "w") as f:
        f.write(str(port))

    try:
        main_loop(shared_state)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Shutting down...")
        shared_state.set('running', False)
        sidecar.stop()
        sidecar.join()
        if os.path.exists(port_file):
            os.remove(port_file)
        cv2.destroyAllWindows()
        print("Shutdown complete.")
