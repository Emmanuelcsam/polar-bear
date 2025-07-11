import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
# !!! CHANGE THIS TO THE PATH OF YOUR IMAGE !!!
# Use home directory expansion for better Ubuntu compatibility
INPUT_IMAGE_PATH = os.path.expanduser("~/Pictures/Camera/Photo from 2025-07-07 10-59-11.519387.jpeg")
IMG_SIZE = 64 # Image dimensions (width and height)
LATENT_DIM = 30 # Complexity of the learned features
EPOCHS = 2000 # How long to train the model

# --- MODEL DEFINITION ---
# A simple autoencoder to learn and generate images.
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(IMG_SIZE * IMG_SIZE * 3, 400)
        self.fc21 = nn.Linear(400, LATENT_DIM) # for mean
        self.fc22 = nn.Linear(400, LATENT_DIM) # for log variance
        # Decoder
        self.fc3 = nn.Linear(LATENT_DIM, 400)
        self.fc4 = nn.Linear(400, IMG_SIZE * IMG_SIZE * 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, IMG_SIZE * IMG_SIZE * 3))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- DATA PREPARATION ---
# Load the image using OpenCV, resize it, and convert to a PyTorch tensor.
try:
    # Check if file exists first
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: File not found at {INPUT_IMAGE_PATH}")
        print("Please update the INPUT_IMAGE_PATH variable in the script.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Home directory: {os.path.expanduser('~')}")
        exit()

    image = cv2.imread(INPUT_IMAGE_PATH)
    if image is None:
        print(f"Error: Unable to read image at {INPUT_IMAGE_PATH}")
        print("The file might be corrupted or in an unsupported format.")
        exit()

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert for correct color display
    tensor = torch.from_numpy(image).float() / 255.0
    tensor = tensor.view(1, -1) # Flatten the tensor
    print(f"Successfully loaded image from: {INPUT_IMAGE_PATH}")
except Exception as e:
    print(f"Error loading image: {e}")
    print("Please update the INPUT_IMAGE_PATH variable in the script.")
    exit()

# --- TRAINING ---
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    recon_batch, mu, logvar = model(tensor)

    # Calculate loss: reconstruction + KL divergence
    BCE = nn.functional.binary_cross_entropy(recon_batch, tensor, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = BCE + KLD

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.2f}')

# --- GENERATION ---
# Infinitely generate and display new images.
model.eval()
print("\nTraining finished. Generating images now. Press 'q' or ESC to quit.")

# Create window with specific properties for better Ubuntu compatibility
cv2.namedWindow('Generated Images', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Generated Images', 512, 512)

try:
    while True:
        with torch.no_grad():
            # Create a random point in the learned feature space
            z = torch.randn(1, LATENT_DIM)
            # Decode it into a new image
            sample = model.decode(z).numpy()

            # Reshape and convert back to an OpenCV image for display
            img_out = sample.reshape(IMG_SIZE, IMG_SIZE, 3)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

            # Make image larger for better viewing
            img_out = cv2.resize(img_out, (512, 512), interpolation=cv2.INTER_NEAREST)

            cv2.imshow('Generated Images', img_out)

            # Better key handling for Ubuntu
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    cv2.destroyAllWindows()
    print("Window closed.")
