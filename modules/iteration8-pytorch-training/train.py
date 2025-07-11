import numpy as np
import cv2

# --- Configuration ---
IMG_SIZE = 128
LEARNING_RATE = 0.05 # How fast the model learns
BATCH_SIZE = 4096    # How many pixels to update each frame

# --- Helper Function ---
def generate_from_model(model):
    """Generates an image by sampling from the model's probabilities."""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # For each pixel and channel, choose a color value based on its probability
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            for c in range(3):
                probs = model[y, x, c]
                color_val = np.random.choice(256, p=probs)
                img[y, x, c] = color_val
    return img

# --- Main Execution ---
# 1. Load the target image and the generator model
target_data = np.load('target_data.npz')['target']
model = np.load('generator_model.npy')
print("Starting training loop... Press 'q' in the window to stop.")

while True:
    # 2. Generate the current best-guess image
    generated_image = generate_from_model(model)
    cv2.imshow('Evolving Image (Press "q" to stop)', generated_image)

    # 3. Reinforcement: Update a random batch of pixels
    # Choose random pixel coordinates to update
    rand_y = np.random.randint(0, IMG_SIZE, size=BATCH_SIZE)
    rand_x = np.random.randint(0, IMG_SIZE, size=BATCH_SIZE)

    for y, x in zip(rand_y, rand_x):
        for c in range(3): # For each color channel (B, G, R)
            target_val = target_data[y, x, c]
            
            # Increase probability of the target color value
            update = model[y, x, c, target_val] * LEARNING_RATE
            model[y, x, c, target_val] += update
            
            # 4. Normalize the probabilities for the updated pixel
            model[y, x, c] /= np.sum(model[y, x, c])

    # 5. Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Save the final trained model and clean up
np.save('generator_model.npy', model)
cv2.destroyAllWindows()
print("Training stopped. Final model saved.")

