import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Use home directory expansion for better Ubuntu compatibility
INPUT_IMAGE_PATH = os.path.expanduser("~/Pictures/Camera/Photo from 2025-07-07 10-59-11.519387.jpeg")
IMG_SIZE = 64  # Image dimensions (width and height)
LATENT_DIM = 30  # Complexity of the learned features
LEARNING_RATE = 1e-3

# --- MODEL DEFINITION ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(IMG_SIZE * IMG_SIZE * 3, 400)
        self.fc21 = nn.Linear(400, LATENT_DIM)  # for mean
        self.fc22 = nn.Linear(400, LATENT_DIM)  # for log variance
        # Decoder
        self.fc3 = nn.Linear(LATENT_DIM, 400)
        self.fc4 = nn.Linear(400, IMG_SIZE * IMG_SIZE * 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, IMG_SIZE * IMG_SIZE * 3))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- DATA PREPARATION ---
def load_image(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        exit()
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Unable to read image at {path}")
        exit()
    
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).float() / 255.0
    return tensor.view(1, -1), image # Return flattened tensor and original RGB image for display

# --- VISUALIZATION SETUP ---
plt.ion() # Turn on interactive mode
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("Live VAE Training")

# Subplot for Original Image
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Subplot for Reconstructed Image
axes[0, 1].set_title("Reconstructed Image")
axes[0, 1].axis('off')

# Subplot for Generated Image
axes[1, 0].set_title("Generated Image")
axes[1, 0].axis('off')

# Subplot for Loss Curve
axes[1, 1].set_title("Training Loss")
axes[1, 1].set_xlabel("Iteration")
axes[1, 1].set_ylabel("Loss")
loss_history = []
line, = axes[1, 1].plot(loss_history)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- MAIN LOOP ---
def main():
    tensor, original_img_display = load_image(INPUT_IMAGE_PATH)
    print(f"Successfully loaded image from: {INPUT_IMAGE_PATH}")

    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Display original image once
    axes[0, 0].imshow(original_img_display)

    iteration = 0
    print("Starting continuous training... Close the plot window to stop.")
    while plt.fignum_exists(fig.number):
        # --- Train Step ---
        model.train()
        recon_batch, mu, logvar = model(tensor)

        BCE = nn.functional.binary_cross_entropy(recon_batch, tensor, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        # --- Visualization Step ---
        model.eval()
        with torch.no_grad():
            # Update Reconstructed Image
            recon_img = recon_batch.numpy().reshape(IMG_SIZE, IMG_SIZE, 3)
            axes[0, 1].imshow(recon_img)

            # Update Generated Image
            z = torch.randn(1, LATENT_DIM)
            generated_sample = model.decode(z).numpy()
            generated_img = generated_sample.reshape(IMG_SIZE, IMG_SIZE, 3)
            axes[1, 0].imshow(generated_img)

            # Update Loss Plot
            line.set_ydata(loss_history)
            line.set_xdata(range(len(loss_history)))
            axes[1, 1].relim()
            axes[1, 1].autoscale_view()

            # Update titles and draw canvas
            fig.suptitle(f"Live VAE Training - Iteration: {iteration+1}")
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        iteration += 1

    print("Plot window closed. Exiting.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        plt.ioff()
        plt.close()
        print("Cleanup complete.")
