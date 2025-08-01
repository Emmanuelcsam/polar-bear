import torch
import torch.nn as nn
import torch.optim as optim
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
    def report_progress(p, m=''): pass
    def update_state(k, v): pass
    def publish_event(t, d, target=None): pass

# Define PixelGenerator locally to avoid import issues
class PixelGenerator(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(size, size, 3, 256))

    def get_probs(self):
        return torch.softmax(self.logits, dim=3)

# --- Configuration ---
IMG_SIZE = 128
LEARNING_RATE = 0.1

# --- Main Execution ---
# 1. Setup model, optimizer, and loss function
model = PixelGenerator(IMG_SIZE)
if os.path.exists('generator_model.pth'):
    try:
        model.load_state_dict(torch.load('generator_model.pth'))
        publish_event('info', {'message': 'Loaded existing model'})
    except:
        publish_event('warning', {'message': 'Could not load model, starting fresh'})
        
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# Store in shared state for integration
update_state('model', model)
update_state('optimizer', optimizer)
update_state('loss_fn', loss_fn)

# 2. Load the target data
if os.path.exists('target_data.pt'):
    target = torch.load('target_data.pt') # Shape: (H, W, C)
else:
    # Create random target for testing
    target = torch.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3))
    torch.save(target, 'target_data.pt')
    
update_state('target', target)
print("Starting training... Press 'q' in the window to stop.")

iteration = 0
max_iterations = get_parameter('MAX_ITERATIONS', float('inf')) if INTEGRATION_AVAILABLE else float('inf')

while iteration < max_iterations:
    optimizer.zero_grad() # Reset gradients

    # 3. Calculate Loss
    # We need to reshape the tensors for the loss function:
    # Logits: (H, W, C, 256) -> (H*W*C, 256)
    # Target: (H, W, C) -> (H*W*C)
    logits_flat = model.logits.view(-1, 256)
    target_flat = target.view(-1)
    loss = loss_fn(logits_flat, target_flat)

    # 4. Backpropagation and Optimization
    loss.backward()
    optimizer.step()
    
    # Report progress
    if iteration % 10 == 0:
        report_progress(iteration / max_iterations if max_iterations != float('inf') else 0,
                       f'Loss: {loss.item():.4f}')

    # 5. Visualization (every few steps)
    with torch.no_grad(): # Disable gradient calculation for generation
        probs = model.get_probs()
        # Sample from the probability distribution for each pixel
        dist = torch.distributions.Categorical(probs)
        generated_tensor = dist.sample()
        # Convert to NumPy array for display with OpenCV
        img_np = generated_tensor.byte().cpu().numpy()
        cv2.imshow('Evolving Image (Press "q")', img_np)

    # 6. Check for exit conditions
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Check for external stop signal
    if INTEGRATION_AVAILABLE and get_state('stop_training', False):
        publish_event('info', {'message': 'Training stopped by external signal'})
        break
    
    iteration += 1

# 7. Save the final trained model and clean up
torch.save(model.state_dict(), 'generator_model.pth')
cv2.destroyAllWindows()
print(f"Training stopped. Final loss: {loss.item()}. Model saved.")

# Report completion
publish_event('status', {
    'status': 'completed',
    'iterations': iteration,
    'final_loss': loss.item()
})

