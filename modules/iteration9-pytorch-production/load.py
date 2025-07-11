import os
import numpy as np
import cv2
import torch

# Try to import integration capabilities
try:
    from integration_wrapper import *
    inject_integration()
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    # Define minimal stubs for independent running
    def get_parameter(n, d=None): return globals().get(n, d)
    def update_state(k, v): pass
    def publish_event(t, d, target=None): pass
    def report_progress(p, m=''): pass

# --- Configuration ---
IMG_SIZE = get_parameter('IMG_SIZE', 128)
OUTPUT_FILE = get_parameter('OUTPUT_FILE', 'target_data.pt')
INPUT_PATH = get_parameter('INPUT_PATH', None)

# --- Main Execution ---
def load_and_prepare_data():
    """
    Prompts for a directory, loads images, calculates the average
    image, and saves it as a PyTorch tensor.
    """
    publish_event('status', {'status': 'loading_images'})
    
    # Use parameter or prompt
    path = INPUT_PATH
    if path is None:
        path = input("Enter the path to the directory with your images: ")
    
    if not os.path.isdir(path):
        print("Error: Invalid directory path.")
        publish_event('error', {'message': 'Invalid directory path'})
        return

    images = []
    total_files = len(os.listdir(path))
    processed = 0
    
    for filename in os.listdir(path):
        try:
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                processed += 1
                report_progress(processed / total_files, f'Loaded {filename}')
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            publish_event('warning', {'message': f'Could not read {filename}: {e}'})

    if not images:
        print("No images found or loaded. Exiting.")
        publish_event('error', {'message': 'No images found'})
        return

    print(f"Loaded {len(images)} images. Calculating average...")
    publish_event('info', {'message': f'Loaded {len(images)} images'})
    
    # Calculate the average image with NumPy first
    avg_image_np = np.mean(images, axis=0).astype(np.uint8)

    # Convert the final target image to a PyTorch tensor
    # We use .long() because these are class indices (0-255) for the loss function
    target_tensor = torch.from_numpy(avg_image_np).long()
    
    # Store in shared state
    update_state('target', target_tensor)
    update_state('target_image_np', avg_image_np)
    update_state('num_source_images', len(images))

    # Save the target tensor for the training script
    torch.save(target_tensor, OUTPUT_FILE)

    # Display the target image for verification if not in headless mode
    if get_parameter('HEADLESS_MODE', False):
        cv2.imwrite('target_preview.png', avg_image_np)
        print("Target preview saved to 'target_preview.png'")
    else:
        cv2.imshow('Target Image (Press any key to continue)', avg_image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"Target tensor saved to '{OUTPUT_FILE}'")
    publish_event('status', {'status': 'complete', 'output_file': OUTPUT_FILE})

if __name__ == '__main__':
    load_and_prepare_data()

