import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
# !!! SET THESE PATHS AND VALUES !!!
NORMAL_IMAGES_DIR = "/home/user/normal_images" # Directory of images to learn from
ANOMALY_MAP_PATH  = "live_anomaly_map.png"     # Where to save the output map on detection
WEBCAM_ID = 0 # 0 is usually the default webcam, try 1 or 2 if not working

IMG_SIZE = 64
LATENT_DIM = 30
EPOCHS = 50 # Fewer epochs per image, but runs over the whole batch
ANOMALY_THRESHOLD_MULTIPLIER = 1.8 # How much higher the error must be to be an anomaly

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
def load_images_from_directory(dir_path):
    """Loads all images from a directory and converts them to a tensor batch."""
    tensors = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    try:
        for filename in os.listdir(dir_path):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                img_path = os.path.join(dir_path, filename)
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(image).float() / 255.0
                    tensors.append(tensor.view(1, -1))
    except Exception as e:
        print(f"Error loading images from {dir_path}: {e}")
        exit()
    if not tensors:
        print(f"No images found in directory: {dir_path}")
        exit()
    return torch.cat(tensors, 0)

# --- 1. TRAINING ON BATCH OF IMAGES ---
print("--- Training on Normal Images from Directory ---")
normal_tensors = load_images_from_directory(NORMAL_IMAGES_DIR)
print(f"Loaded {len(normal_tensors)} images for training.")
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    for image_tensor in normal_tensors:
        model.train()
        recon, mu, logvar = model(image_tensor)
        BCE = nn.functional.binary_cross_entropy(recon, image_tensor, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{EPOCHS}, Last Image Loss: {loss.item():.2f}')

print("\n--- Training Complete ---")

# --- 2. ESTABLISH ANOMALY THRESHOLD ---
model.eval()
with torch.no_grad():
    total_error = 0
    for image_tensor in normal_tensors:
        recon, _, _ = model(image_tensor)
        total_error += nn.functional.binary_cross_entropy(recon, image_tensor, reduction='sum').item()
    avg_normal_error = total_error / len(normal_tensors)
    anomaly_threshold = avg_normal_error * ANOMALY_THRESHOLD_MULTIPLIER
    print(f"Average Normal Error: {avg_normal_error:.2f}")
    print(f"Anomaly Threshold Established: {anomaly_threshold:.2f}")

# --- 3. LIVE ANOMALY DETECTION ---
print("\n--- Starting Live Feed Analysis (Press 'q' to quit) ---")

# Try multiple video capture backends for better Ubuntu compatibility
backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
cap = None

for backend in backends:
    cap = cv2.VideoCapture(WEBCAM_ID, backend)
    if cap.isOpened():
        print(f"Successfully opened webcam with backend: {backend}")
        break

if not cap or not cap.isOpened():
    print(f"Error: Could not open webcam with ID {WEBCAM_ID}.")
    print("Try running: v4l2-ctl --list-devices")
    print("Or check permissions: sudo usermod -a -G video $USER")
    exit()

# Set camera properties for better performance on Ubuntu
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Process the live frame
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_tensor = (torch.from_numpy(frame_rgb).float() / 255.0).view(1, -1)

    # Get reconstruction and error
    model.eval()
    with torch.no_grad():
        recon_frame, _, _ = model(frame_tensor)
        test_error = nn.functional.binary_cross_entropy(recon_frame, frame_tensor, reduction='sum').item()

    # Generate anomaly map
    diff = torch.abs(frame_tensor.view(IMG_SIZE, IMG_SIZE, 3) - recon_frame.view(IMG_SIZE, IMG_SIZE, 3))
    error_map = torch.sum(diff, dim=2)
    error_map_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
    heatmap_small = (error_map_norm * 255).byte().numpy()
    heatmap_color = cv2.applyColorMap(heatmap_small, cv2.COLORMAP_JET)

    # Overlay heatmap on original frame
    frame_display = cv2.resize(frame, (512, 512))
    heatmap_display = cv2.resize(heatmap_color, (512, 512))
    overlayed_frame = cv2.addWeighted(frame_display, 0.6, heatmap_display, 0.4, 0)

    # Check for anomaly and display info
    is_anomaly = test_error > anomaly_threshold
    status_text = f"Error: {test_error:.2f}"
    color = (0, 0, 255) if is_anomaly else (0, 255, 0)
    cv2.putText(overlayed_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if is_anomaly:
        cv2.putText(overlayed_frame, "ANOMALY DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imwrite(ANOMALY_MAP_PATH, heatmap_display) # Save the map

    cv2.imshow('Live Anomaly Detection', overlayed_frame)

    # Use cv2.waitKey with proper handling for Ubuntu
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' or ESC
        break

cap.release()
cv2.destroyAllWindows()