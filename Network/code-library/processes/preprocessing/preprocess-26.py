import torch
import torch.nn as nn
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
    def update_state(k, v): pass
    def publish_event(t, d, target=None): pass

# --- Configuration ---
IMG_SIZE = get_parameter('IMG_SIZE', 128)
MODEL_FILENAME = get_parameter('MODEL_FILENAME', 'generator_model.pth')

# --- Model Definition ---
class PixelGenerator(nn.Module):
    def __init__(self, size):
        super().__init__()
        # Logits are raw, unnormalized scores. Softmax will convert them
        # to probabilities. We initialize with zeros for uniform probability.
        # Shape: (Height, Width, ColorChannels, ColorValues)
        self.logits = nn.Parameter(torch.zeros(size, size, 3, 256))

    def get_probs(self):
        # Use softmax to convert logits into valid probabilities
        return torch.softmax(self.logits, dim=3)

# --- Main Execution ---
def initialize_generator():
    """
    Initializes the PyTorch generator model and saves its initial state.
    """
    if os.path.exists(MODEL_FILENAME):
        print(f"Model '{MODEL_FILENAME}' already exists. Skipping.")
        publish_event('info', {'message': f'Model {MODEL_FILENAME} already exists'})
        return

    print("Initializing a new PyTorch generator model...")
    publish_event('status', {'status': 'initializing'})
    
    # Create an instance of our generator model
    model = PixelGenerator(IMG_SIZE)
    
    # Store in shared state
    update_state('model', model)
    update_state('model_initialized', True)

    # Save the initial state of the model
    # .state_dict() contains all the learnable parameters (our logits)
    torch.save(model.state_dict(), MODEL_FILENAME)
    print(f"Generator model initialized and saved to '{MODEL_FILENAME}'")

if __name__ == '__main__':
    initialize_generator()

