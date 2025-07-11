
import torch
import torchvision.transforms as T
from PIL import Image
import glob
import time
import os
from shared_state import SharedState
from connector import SidecarConnector, find_free_port

def main_task(state):
    """The main task for the reference processor."""
    input_glob = state.get_parameter('input_glob', 'refs/*.jpg')
    output_dir = state.get_parameter('output_dir', 'refs_processed')
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        files_to_process = glob.glob(input_glob)
        
        if not files_to_process:
            print(f"No files found matching glob: {input_glob}")
            state.set_parameter('status', f'Error: No files found for {input_glob}')
            return

        state.set_parameter('status', 'processing')
        state.set_parameter('total_files', len(files_to_process))
        
        transform = T.Compose([T.Resize(256), T.ToTensor()])

        for i, f_path in enumerate(files_to_process):
            if not state.get('running', True):
                print("Processing stopped by connector.")
                break
            
            state.set_parameter('current_file', f_path)
            state.set_parameter('progress', f"{i+1}/{len(files_to_process)}")
            
            output_filename = os.path.join(output_dir, os.path.basename(f_path).replace('.jpg', '.pt'))
            
            img = transform(Image.open(f_path).convert('RGB'))
            torch.save(img, output_filename)
            print(f"Processed {f_path} -> {output_filename}")

        print("Reference processing complete.")
        state.set_parameter('status', 'completed')

    except Exception as e:
        error_message = f"Error in ref processor: {e}"
        print(error_message)
        state.set_parameter('status', error_message)


if __name__ == "__main__":
    script_name = os.path.basename(__file__)
    shared_state = SharedState()
    
    # --- Default Parameters ---
    shared_state.set_parameter('input_glob', 'refs/*.jpg')
    shared_state.set_parameter('output_dir', 'refs_processed')
    shared_state.set_parameter('status', 'initializing')
    shared_state.set_parameter('progress', '0/0')
    
    # --- Sidecar Connector ---
    port = find_free_port()
    sidecar = SidecarConnector(script_name, shared_state, port)
    sidecar.start()
    
    port_file = f".{script_name}.port"
    with open(port_file, "w") as f:
        f.write(str(port))

    try:
        # Create a dummy refs directory if it doesn't exist
        if not os.path.exists('refs'):
            os.makedirs('refs')
            print("Created 'refs' directory. Please add reference images there.")

        main_task(shared_state)
        
        if shared_state.get_parameter('status') == 'completed':
            print("Task complete. Staying alive for 10 seconds for inspection...")
            time.sleep(10)

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
