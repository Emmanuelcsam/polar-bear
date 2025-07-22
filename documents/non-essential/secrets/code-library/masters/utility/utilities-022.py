import torch
import torch.nn as nn
import numpy as np
import cv2
import os

# Try to import integration capabilities
try:
    from integration_wrapper import *
    inject_integration()
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    # Define minimal stubs for independent running
    def get_parameter(n, d=None): return globals().get(n, d)
    def get_state(k, d=None): return d
    def update_state(k, v): pass
    def publish_event(t, d, target=None): pass
    def report_progress(p, m=''): pass

# Define PixelGenerator locally to avoid import issues
class PixelGenerator(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(size, size, 3, 256))

    def get_probs(self):
        return torch.softmax(self.logits, dim=3)

# --- Configuration ---
IMG_SIZE = get_parameter('IMG_SIZE', 128)
MODEL_FILENAME = get_parameter('MODEL_FILENAME', 'generator_model.pth')
OUTPUT_FILENAME = get_parameter('OUTPUT_FILENAME', 'final_generated_image.png')
NUM_SAMPLES = get_parameter('NUM_SAMPLES', 1)
SHOW_DISPLAY = get_parameter('SHOW_DISPLAY', True)

# --- Helper Function ---
def generate_final_image(model, sample_idx=0):
    """Generates one image by sampling from the model's probabilities."""
    print(f"Generating image sample {sample_idx + 1}...")
    report_progress((sample_idx + 0.5) / NUM_SAMPLES, f'Generating sample {sample_idx + 1}')
    
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # No need to track gradients here
        # Get the learned probability distributions
        probs = model.get_probs()
        # Create a categorical distribution to sample from
        dist = torch.distributions.Categorical(probs)
        # Sample one value for each pixel's distribution
        generated_tensor = dist.sample()
        # Convert to a NumPy array suitable for OpenCV (uint8)
        img_np = generated_tensor.byte().cpu().numpy()
    return img_np

# --- Main Execution ---
def main():
    """
    Loads the trained model and uses it to generate and save
    final high-quality images.
    """
    publish_event('status', {'status': 'starting_generation'})
    
    # 1. Try to get model from shared state first
    model = get_state('model', None)
    
    if model is None:
        # Load the model structure and the trained parameters
        try:
            model = PixelGenerator(IMG_SIZE)
            if os.path.exists(MODEL_FILENAME):
                model.load_state_dict(torch.load(MODEL_FILENAME))
                publish_event('info', {'message': f'Loaded model from {MODEL_FILENAME}'})
            else:
                print(f"Error: Model file '{MODEL_FILENAME}' not found.")
                print("Please run scripts 2 and 3 first.")
                publish_event('error', {'message': 'Model file not found'})
                return
        except Exception as e:
            print(f"Error loading model: {e}")
            publish_event('error', {'message': f'Error loading model: {e}'})
            return
    else:
        publish_event('info', {'message': 'Using model from shared state'})
    
    # Store model in shared state
    update_state('model', model)

    # 2. Generate images
    generated_images = []
    for i in range(NUM_SAMPLES):
        final_image = generate_final_image(model, i)
        generated_images.append(final_image)
        
        # Save each image
        if NUM_SAMPLES > 1:
            filename = OUTPUT_FILENAME.replace('.png', f'_{i+1}.png')
        else:
            filename = OUTPUT_FILENAME
        
        cv2.imwrite(filename, final_image)
        print(f"Successfully saved image as '{filename}'")
        
        report_progress((i + 1) / NUM_SAMPLES, f'Saved {filename}')
    
    # Store results in shared state
    update_state('generated_images', generated_images)
    publish_event('status', {'status': 'generation_complete', 'num_images': len(generated_images)})

    # 3. Display the results if requested
    if SHOW_DISPLAY:
        for i, img in enumerate(generated_images):
            window_name = f'Generated Image {i+1}/{NUM_SAMPLES} (PyTorch)'
            cv2.imshow(window_name, img)
        
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

