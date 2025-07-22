import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import logging
from datetime import datetime
import os

# Setup logging (verbose with timestamps)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('fiber_log.txt'))

# Custom Dataset (adapt to your .pt files)
class FiberDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.is_train = is_train
        logger.info(f"Loaded {len(self.files)} files from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(file_path)  # Assume .pt has {'image': tensor, 'mask': tensor} for regions
        image = data['image']  # [3, H, W]
        mask = data['mask']    # [H, W] with labels 0=background, 1=core, 2=cladding, 3=ferrule
        return image, mask

# U-Net for Segmentation (Feature Extraction + Region Classification)
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=4):  # 4 classes: bg, core, cladding, ferrule
        super(UNet, self).__init__()
        logger.info("Initializing U-Net for region segmentation")

        # Encoder (feature extraction)
        self.inc = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1))

        # Decoder (classification)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = nn.Conv2d(256, 128, 3, padding=1)  # Skip connection
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = nn.Conv2d(128, 64, 3, padding=1)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.inc(x))
        x2 = F.relu(self.down1(x1))
        x3 = F.relu(self.down2(x2))

        # Decoder with skip connections
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)  # Skip
        x = F.relu(self.conv_up1(x))
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)  # Skip
        x = F.relu(self.conv_up2(x))
        logits = self.outc(x)
        return logits  # [B, n_classes, H, W]

# Autoencoder for Anomaly Detection (Defect Extraction in Regions)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        logger.info("Initializing Autoencoder for defect detection")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Full Model: Segmentation + Anomaly Detection
class FiberModel(nn.Module):
    def __init__(self):
        super(FiberModel, self).__init__()
        self.segmenter = UNet()
        self.anomaly_detector = Autoencoder()

    def forward(self, x):
        # Segment regions
        seg_logits = self.segmenter(x)  # [B, 4, H, W]
        seg_probs = F.softmax(seg_logits, dim=1)
        seg_mask = torch.argmax(seg_probs, dim=1)  # [B, H, W]

        # For each region (e.g., core=1), mask and detect anomalies
        anomalies = torch.zeros_like(x)  # Placeholder for defect map
        for b in range(x.size(0)):
            for region in [1, 2, 3]:  # core, cladding, ferrule
                region_mask = (seg_mask[b] == region).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                masked_input = x[b:b+1] * region_mask.repeat(1, 3, 1, 1)
                reconstructed = self.anomaly_detector(masked_input)
                defect_map = torch.abs(masked_input - reconstructed).mean(dim=1, keepdim=True)  # [1, 1, H, W]
                anomalies[b:b+1] += defect_map * region_mask  # Accumulate defects

        return {
            'seg_logits': seg_logits,
            'seg_mask': seg_mask,
            'defect_map': anomalies.mean(dim=1)  # [B, H, W] anomaly intensity
        }

# Training Loop
def train_model(model, dataloader, optimizer, criterion_seg, criterion_ano, epochs=10, device='cuda'):
    model.to(device)
    model.train()
    logger.info(f"Starting training for {epochs} epochs")

    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device).long()  # Batched

            outputs = model(images)

            # Segmentation loss (cross-entropy for classes)
            seg_loss = criterion_seg(outputs['seg_logits'], masks)

            # Anomaly loss (MSE reconstruction on masked regions; simulate with full image for simplicity)
            reconstructed = model.anomaly_detector(images)
            ano_loss = criterion_ano(reconstructed, images)

            loss = seg_loss + ano_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    logger.info("Training complete")

# Evaluation
def evaluate_model(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    total_iou = 0
    total_mse = 0
    num_batches = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device).long()

            outputs = model(images)

            # IoU for segmentation
            pred_mask = torch.argmax(outputs['seg_logits'], dim=1)
            intersection = (pred_mask == masks).sum().item()
            union = (pred_mask.numel() + masks.numel() - intersection)
            iou = intersection / union if union > 0 else 0
            total_iou += iou

            # MSE for anomalies (reconstruction error)
            reconstructed = model.anomaly_detector(images)
            mse = F.mse_loss(reconstructed, images)
            total_mse += mse.item()

            num_batches += 1

    avg_iou = total_iou / num_batches
    avg_mse = total_mse / num_batches
    logger.info(f"Evaluation - Avg IoU: {avg_iou:.4f}, Avg MSE: {avg_mse:.4f}")
    return avg_iou, avg_mse

# Real-Time Inference (for single image/frame)
def real_time_inference(model, image_tensor, device='cuda'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        start_time = datetime.now()
        outputs = model(image_tensor.unsqueeze(0).to(device))  # Add batch dim
        end_time = datetime.now()
        logger.info(f"Inference time: {(end_time - start_time).total_seconds() * 1000:.2f} ms")
        return outputs['seg_mask'].squeeze(0), outputs['defect_map'].squeeze(0)

# Example Usage
if __name__ == "__main__":
    # Config (no flags)
    config = {'batch_size': 8, 'epochs': 10, 'lr': 0.001, 'data_dir': 'your_dataset/'}

    # Dataset and Loader (batched)
    dataset = FiberDataset(config['data_dir'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Model, Optimizer, Losses
    model = FiberModel()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion_seg = nn.CrossEntropyLoss()  # For region classification
    criterion_ano = nn.MSELoss()  # For defect reconstruction

    # Train and Evaluate
    train_model(model, dataloader, optimizer, criterion_seg, criterion_ano, config['epochs'])
    evaluate_model(model, dataloader)

    # Real-time example
    test_image = torch.randn(3, 256, 256)  # Your input frame
    seg_mask, defect_map = real_time_inference(model, test_image)
    logger.info(f"Real-time output: Seg shape {seg_mask.shape}, Defect shape {defect_map.shape}")
