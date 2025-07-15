import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
import glob

# --- CONFIGURATION ---
DATASET_DIR = "dataset"  # Directory with normal images for training
REFERENCE_DIR = "reference"  # Directory with images to test (includes anomalies)
IMG_SIZE = 64
LATENT_DIM = 32
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 2

# --- VAE MODEL DEFINITION ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, LATENT_DIM)
        self.fc_logvar = nn.Linear(256, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, IMG_SIZE * IMG_SIZE * 3),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, IMG_SIZE * IMG_SIZE * 3))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# --- DATA HANDLING ---
class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_paths = glob.glob(os.path.join(directory, '**', '*.jpg'), recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

# --- TRAINING FUNCTION ---
def train_vae(model, dataloader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss(reduction='sum')
    print("Starting VAE training on normal images...")
    for epoch in range(epochs):
        total_loss = 0
        for (data, _) in dataloader:
            data = data.view(data.size(0), -1)
            recon, mu, logvar = model(data)
            
            recon_loss = loss_function(recon, data)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader.dataset):.4f}")
    print("Training finished.")

# --- ANOMALY DETECTION ---
def get_anomaly_threshold(model, dataloader):
    model.eval()
    errors = []
    loss_function = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for (data, _) in dataloader:
            data = data.view(data.size(0), -1)
            recon, _, _ = model(data)
            error = loss_function(recon, data).mean(axis=1)
            errors.extend(error.numpy())
    
    threshold = np.mean(errors) + 2 * np.std(errors)
    print(f"Calculated anomaly threshold: {threshold:.4f}")
    return threshold

def test_and_visualize(model, threshold, test_dir):
    transform = get_transforms()
    test_files = glob.glob(os.path.join(test_dir, '**', '*.jpg'), recursive=True)
    loss_function = nn.MSELoss(reduction='mean')

    for img_path in test_files:
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            image_flat = image_tensor.view(image_tensor.size(0), -1)
            recon_flat, _, _ = model(image_flat)
            error = loss_function(recon_flat, image_flat).item()

        is_anomaly = error > threshold
        classification = "Anomaly" if is_anomaly else "Normal"
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f'Classification: {classification} (Error: {error:.4f})', color='red' if is_anomaly else 'green')
        
        original_display = np.array(image.resize((256, 256)))
        ax1.imshow(original_display)
        ax1.set_title("Original Image")
        ax1.axis('off')

        recon_display = recon_flat.numpy().reshape(3, IMG_SIZE, IMG_SIZE)
        recon_display = np.transpose(recon_display, (1, 2, 0))
        ax2.imshow(cv2.resize(recon_display, (256, 256)))
        ax2.set_title("Reconstructed by VAE")
        ax2.axis('off')
        
        plt.show()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load training data and train the VAE
    train_dataset = ImageDataset(DATASET_DIR, transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    vae = VAE()
    train_vae(vae, train_loader, epochs=EPOCHS)
    
    # 2. Determine the anomaly detection threshold
    threshold = get_anomaly_threshold(vae, train_loader)
    
    # 3. Test on reference images and visualize results
    print("\nStarting anomaly detection on reference images...")
    print("Close each plot window to proceed to the next image.")
    test_and_visualize(vae, threshold, REFERENCE_DIR)
    
    print("\nAnomaly detection complete.")
