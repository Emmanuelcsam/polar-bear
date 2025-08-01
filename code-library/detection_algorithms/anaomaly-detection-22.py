import torch
import torch.nn as nn
import cv2
import numpy as np

# --- CONFIGURATION ---
# !!! SET THESE PATHS TO YOUR IMAGES !!!
NORMAL_IMAGE_PATH = r"C:\path\to\your\normal_image.jpg" # Image to learn from
TEST_IMAGE_PATH   = r"C:\path\to\your\test_image.jpg"   # Image to check for anomalies
ANOMALY_MAP_PATH  = r"anomaly_map.png"                  # Where to save the output map

IMG_SIZE = 64
LATENT_DIM = 30
EPOCHS = 2000
ANOMALY_THRESHOLD_MULTIPLIER = 1.5 # How much higher the error must be to be an anomaly

# --- MODEL DEFINITION (Same as before) ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(IMG_SIZE * IMG_SIZE * 3, 400)
        self.fc21 = nn.Linear(400, LATENT_DIM)
        self.fc22 = nn.Linear(400, LATENT_DIM)
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

# --- UTILITY FUNCTION ---
def process_image(path):
    """Loads, resizes, and converts an image to a tensor."""
    try:
        image = cv2.imread(path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).float() / 255.0
        return tensor.view(1, -1) # Return flattened tensor
    except Exception as e:
        print(f"Error loading image at {path}: {e}")
        exit()

# --- 1. TRAINING ---
print("--- Training on Normal Image ---")
normal_tensor = process_image(NORMAL_IMAGE_PATH)
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    recon, mu, logvar = model(normal_tensor)
    BCE = nn.functional.binary_cross_entropy(recon, normal_tensor, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = BCE + KLD
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.2f}')

print("\n--- Training Complete ---")

# --- 2. ANOMALY DETECTION & MAP GENERATION ---
model.eval()
with torch.no_grad():
    # Establish the "normal" reconstruction error threshold
    recon_normal, _, _ = model(normal_tensor)
    normal_error = nn.functional.binary_cross_entropy(recon_normal, normal_tensor, reduction='sum').item()
    anomaly_threshold = normal_error * ANOMALY_THRESHOLD_MULTIPLIER
    print(f"Normal Error Threshold Established: {anomaly_threshold:.2f}")

    # Analyze the test image
    print(f"\n--- Analyzing Test Image: {TEST_IMAGE_PATH} ---")
    test_tensor = process_image(TEST_IMAGE_PATH)
    recon_test, _, _ = model(test_tensor)
    test_error = nn.functional.binary_cross_entropy(recon_test, test_tensor, reduction='sum').item()
    print(f"Test Image Reconstruction Error: {test_error:.2f}")

    # Generate and save the anomaly map
    diff = torch.abs(test_tensor.view(IMG_SIZE, IMG_SIZE, 3) - recon_test.view(IMG_SIZE, IMG_SIZE, 3))
    error_map = torch.sum(diff, dim=2) # Sum differences across color channels
    error_map_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min())
    anomaly_map_img = (error_map_norm * 255).byte().numpy()
    heatmap = cv2.applyColorMap(anomaly_map_img, cv2.COLORMAP_JET)
    
    cv2.imwrite(ANOMALY_MAP_PATH, heatmap)
    print(f"\nAnomaly map saved to: {ANOMALY_MAP_PATH}")

# --- 3. CLASSIFY ---
if test_error > anomaly_threshold:
    print("\nResult: ANOMALY DETECTED")
else:
    print("\nResult: Image appears normal")
