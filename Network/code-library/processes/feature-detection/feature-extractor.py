
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as T
import time
import os
from shared_state import SharedState
from connector import SidecarConnector, find_free_port

def main_task(state):
    """The main task for the feature extractor."""
    input_file = state.get_parameter('input_file', 'ref.jpg')
    output_file = state.get_parameter('output_file', 'features.pt')
    
    try:
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            state.set_parameter('status', f'Error: Input file not found: {input_file}')
            return

        print(f"Processing {input_file}...")
        state.set_parameter('status', 'loading_model')
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        
        state.set_parameter('status', 'transforming_image')
        transform = T.Compose([
            T.Resize(256),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(Image.open(input_file).convert('RGB')).unsqueeze(0)
        
        state.set_parameter('status', 'extracting_features')
        with torch.no_grad():
            features = vgg(img)
            
        state.set_parameter('status', 'saving_features')
        torch.save(features, output_file)
        print(f"Features extracted and saved to {output_file}")
        state.set_parameter('status', 'completed')

    except Exception as e:
        error_message = f"Error in feature extractor: {e}"
        print(error_message)
        state.set_parameter('status', error_message)

if __name__ == "__main__":
    script_name = os.path.basename(__file__)
    shared_state = SharedState()
    
    # --- Default Parameters ---
    shared_state.set_parameter('input_file', 'ref.jpg')
    shared_state.set_parameter('output_file', 'features.pt')
    shared_state.set_parameter('status', 'initializing')
    
    # --- Sidecar Connector ---
    port = find_free_port()
    sidecar = SidecarConnector(script_name, shared_state, port)
    sidecar.start()
    
    port_file = f".{script_name}.port"
    with open(port_file, "w") as f:
        f.write(str(port))

    try:
        main_task(shared_state)
        # Keep the script alive for a short time to be managed
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
