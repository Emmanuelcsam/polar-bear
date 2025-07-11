import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from generative_script_pytorch_2 import PixelGenerator # Import model class

# --- Configuration ---
IMG_SIZE = 128
LEARNING_RATE = 0.1

# --- Main Execution ---
# 1. Setup model, optimizer, and loss function
model = PixelGenerator(IMG_SIZE)
model.load_state_dict(torch.load('generator_model.pth'))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# 2. Load the target data
target = torch.load('target_data.pt') # Shape: (H, W, C)
print("Starting training... Press 'q' in the window to stop.")

while True:
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

    # 5. Visualization (every few steps)
    with torch.no_grad(): # Disable gradient calculation for generation
        probs = model.get_probs()
        # Sample from the probability distribution for each pixel
        dist = torch.distributions.Categorical(probs)
        generated_tensor = dist.sample()
        # Convert to NumPy array for display with OpenCV
        img_np = generated_tensor.byte().cpu().numpy()
        cv2.imshow('Evolving Image (Press "q")', img_np)

    # 6. Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Save the final trained model and clean up
torch.save(model.state_dict(), 'generator_model.pth')
cv2.destroyAllWindows()
print(f"Training stopped. Final loss: {loss.item()}. Model saved.")

