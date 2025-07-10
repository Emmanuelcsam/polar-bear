#!/usr/bin/env python3
"""
Siamese Neural Networks and Similarity Learning Implementation
Based on the lecture transcript on similarity learning techniques
"""

import sys
import subprocess
import importlib
import os
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_action(message, level="INFO"):
    """Log every action with timestamp and color coding"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    color = Colors.OKGREEN if level == "SUCCESS" else Colors.OKBLUE if level == "INFO" else Colors.WARNING if level == "WARNING" else Colors.FAIL
    print(f"{color}[{timestamp}] [{level}] {message}{Colors.ENDC}")

def check_and_install_package(package_name, import_name=None):
    """Check if package is installed, if not install the latest version"""
    if import_name is None:
        import_name = package_name
    
    try:
        log_action(f"Checking for package: {package_name}")
        importlib.import_module(import_name)
        log_action(f"Package {package_name} is already installed", "SUCCESS")
        return True
    except ImportError:
        log_action(f"Package {package_name} not found. Installing latest version...", "WARNING")
        try:
            # Use --upgrade to ensure we get the latest version
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            log_action(f"Successfully installed latest version of {package_name}", "SUCCESS")
            return True
        except subprocess.CalledProcessError:
            log_action(f"Failed to install {package_name}", "ERROR")
            return False

def setup_environment():
    """Auto-detect and install all required packages"""
    log_action("=== Starting Environment Setup ===", "INFO")
    
    required_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn"),
        ("pillow", "PIL"),
        ("tqdm", "tqdm"),
        ("tensorboard", "tensorboard")
    ]
    
    all_installed = True
    for package, import_name in required_packages:
        if not check_and_install_package(package, import_name):
            all_installed = False
    
    if not all_installed:
        log_action("Some packages failed to install. Please check your environment.", "ERROR")
        sys.exit(1)
    
    log_action("All required packages are installed and up to date!", "SUCCESS")
    return True

# Setup environment before importing
setup_environment()

# Now import all required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from PIL import Image
import random
from tqdm import tqdm
import os
from collections import defaultdict

log_action("All imports successful", "SUCCESS")

class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network Architecture as described in the lecture.
    Processes two images in parallel with shared weights.
    """
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        log_action(f"Initializing Siamese Network with embedding dimension: {embedding_dim}")
        
        # Shared CNN backbone for feature extraction
        self.feature_extractor = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=10, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        
        # Calculate size after convolutions
        self.fc_input_size = 512 * 1 * 1  # Adjust based on input size
        
        # Fully connected layers for embedding
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim)
        )
        
        log_action("Siamese Network architecture created successfully", "SUCCESS")
    
    def forward_single(self, x):
        """Forward pass for a single image"""
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def forward(self, x1, x2):
        """Forward pass for image pair"""
        # Process both images through the same network (shared weights)
        embedding1 = self.forward_single(x1)
        embedding2 = self.forward_single(x2)
        return embedding1, embedding2

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss as described in the lecture.
    Brings positive pairs together and pushes negative pairs apart.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        log_action(f"Initialized Contrastive Loss with margin: {margin}")
    
    def forward(self, embedding1, embedding2, label):
        """
        label: 1 for positive pair (same class), 0 for negative pair
        """
        # Calculate Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)
        
        # Contrastive loss formula from lecture
        positive_loss = label * torch.pow(distance, 2)
        negative_loss = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(positive_loss + negative_loss) / 2
        return loss

class TripletLoss(nn.Module):
    """
    Triplet Loss as described in the lecture.
    Ensures anchor-positive distance < anchor-negative distance.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        log_action(f"Initialized Triplet Loss with margin: {margin}")
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        """
        # Calculate distances
        pos_distance = F.pairwise_distance(anchor, positive, keepdim=True)
        neg_distance = F.pairwise_distance(anchor, negative, keepdim=True)
        
        # Triplet loss formula from lecture
        loss = torch.mean(torch.clamp(pos_distance - neg_distance + self.margin, min=0.0))
        return loss

class SiamesePairDataset(Dataset):
    """
    Dataset for Siamese Network training with pairs
    Creates positive and negative pairs from the base dataset
    """
    def __init__(self, root_dataset, transform=None):
        self.dataset = root_dataset
        self.transform = transform
        self.labels = np.array([item[1] for item in self.dataset])
        self.label_to_indices = defaultdict(list)
        
        log_action("Creating index mapping for efficient pair sampling")
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        
        log_action(f"Dataset initialized with {len(self.dataset)} samples across {len(self.label_to_indices)} classes")
    
    def __getitem__(self, index):
        # Get anchor image
        img1, label1 = self.dataset[index]
        
        # Randomly decide if this should be positive or negative pair
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            # Positive pair - same class
            same_class_indices = self.label_to_indices[label1]
            img2_idx = random.choice(same_class_indices)
            img2, label2 = self.dataset[img2_idx]
            target = 1.0
        else:
            # Negative pair - different class
            different_labels = list(set(self.label_to_indices.keys()) - {label1})
            different_label = random.choice(different_labels)
            img2_idx = random.choice(self.label_to_indices[different_label])
            img2, label2 = self.dataset[img2_idx]
            target = 0.0
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(target, dtype=torch.float32)
    
    def __len__(self):
        return len(self.dataset)

class TripletDataset(Dataset):
    """
    Dataset for Triplet Loss training
    Creates anchor-positive-negative triplets
    """
    def __init__(self, root_dataset, transform=None):
        self.dataset = root_dataset
        self.transform = transform
        self.labels = np.array([item[1] for item in self.dataset])
        self.label_to_indices = defaultdict(list)
        
        log_action("Creating index mapping for triplet sampling")
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        
        log_action(f"Triplet dataset initialized with {len(self.dataset)} samples")
    
    def __getitem__(self, index):
        # Get anchor
        anchor_img, anchor_label = self.dataset[index]
        
        # Get positive (same class as anchor)
        positive_indices = self.label_to_indices[anchor_label]
        positive_idx = random.choice([idx for idx in positive_indices if idx != index])
        positive_img, _ = self.dataset[positive_idx]
        
        # Get negative (different class from anchor)
        negative_label = random.choice(list(set(self.label_to_indices.keys()) - {anchor_label}))
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_img, _ = self.dataset[negative_idx]
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img
    
    def __len__(self):
        return len(self.dataset)

class HardNegativeMiner:
    """
    Hard Negative Mining as described in the lecture
    Selects hard triplets for more effective training
    """
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        log_action("Initialized Hard Negative Miner")
    
    def mine_hard_triplets(self, num_triplets=1000):
        """Find hard triplets where positive and negative distances are similar"""
        log_action("Starting hard negative mining process")
        
        self.model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for data in tqdm(self.dataloader, desc="Computing embeddings"):
                images, batch_labels = data
                images = images.to(self.device)
                
                # Get embeddings
                emb = self.model.forward_single(images)
                embeddings.append(emb.cpu())
                labels.extend(batch_labels.numpy())
        
        embeddings = torch.cat(embeddings)
        labels = np.array(labels)
        
        log_action("Mining hard triplets from computed embeddings")
        hard_triplets = []
        
        # Find hard triplets
        for anchor_idx in tqdm(range(len(embeddings)), desc="Finding hard triplets"):
            anchor_label = labels[anchor_idx]
            anchor_emb = embeddings[anchor_idx]
            
            # Get positive candidates
            positive_mask = labels == anchor_label
            positive_mask[anchor_idx] = False
            positive_indices = np.where(positive_mask)[0]
            
            if len(positive_indices) == 0:
                continue
            
            # Get negative candidates
            negative_mask = labels != anchor_label
            negative_indices = np.where(negative_mask)[0]
            
            # Calculate distances
            pos_distances = torch.norm(embeddings[positive_indices] - anchor_emb, dim=1)
            neg_distances = torch.norm(embeddings[negative_indices] - anchor_emb, dim=1)
            
            # Find hardest positive (farthest from anchor)
            hardest_positive_idx = positive_indices[pos_distances.argmax()]
            
            # Find hardest negative (closest to anchor)
            hardest_negative_idx = negative_indices[neg_distances.argmin()]
            
            hard_triplets.append((anchor_idx, hardest_positive_idx, hardest_negative_idx))
            
            if len(hard_triplets) >= num_triplets:
                break
        
        log_action(f"Found {len(hard_triplets)} hard triplets", "SUCCESS")
        return hard_triplets

def train_contrastive(model, train_loader, epochs=10, learning_rate=0.001):
    """Train Siamese Network with Contrastive Loss"""
    log_action("=== Starting Contrastive Loss Training ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_action(f"Using device: {device}")
    
    model = model.to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (img1, img2, label) in enumerate(progress_bar):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            embedding1, embedding2 = model(img1, img2)
            
            # Calculate loss
            loss = criterion(embedding1, embedding2, label)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = running_loss / len(train_loader)
        log_action(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}", "SUCCESS")
    
    return model

def train_triplet(model, train_loader, epochs=10, learning_rate=0.001):
    """Train Siamese Network with Triplet Loss"""
    log_action("=== Starting Triplet Loss Training ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_action(f"Using device: {device}")
    
    model = model.to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (anchor, positive, negative) in enumerate(progress_bar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            anchor_emb = model.forward_single(anchor)
            positive_emb = model.forward_single(positive)
            negative_emb = model.forward_single(negative)
            
            # Calculate loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = running_loss / len(train_loader)
        log_action(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}", "SUCCESS")
    
    return model

def visualize_embeddings(model, dataloader, device, num_samples=1000):
    """Visualize learned embeddings using t-SNE"""
    log_action("Generating embedding visualization")
    
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for i, (images, batch_labels) in enumerate(dataloader):
            if len(embeddings) * dataloader.batch_size >= num_samples:
                break
            
            images = images.to(device)
            emb = model.forward_single(images)
            embeddings.append(emb.cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    embeddings = np.vstack(embeddings)[:num_samples]
    labels = np.array(labels)[:num_samples]
    
    log_action("Applying t-SNE for dimensionality reduction")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Learned Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('embeddings_visualization.png', dpi=300, bbox_inches='tight')
    log_action("Saved embedding visualization to embeddings_visualization.png", "SUCCESS")
    plt.close()

def perform_clustering(model, dataloader, device, n_clusters=10):
    """Perform clustering on learned embeddings"""
    log_action(f"Performing clustering with {n_clusters} clusters")
    
    model.eval()
    embeddings = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            emb = model.forward_single(images)
            embeddings.append(emb.cpu().numpy())
            true_labels.extend(labels.numpy())
    
    embeddings = np.vstack(embeddings)
    true_labels = np.array(true_labels)
    
    log_action("Running K-Means clustering")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate clustering metrics
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    
    log_action(f"Clustering Results - ARI: {ari:.4f}, NMI: {nmi:.4f}", "SUCCESS")
    
    return cluster_labels, embeddings

def image_retrieval_demo(model, dataloader, device, query_idx=0, top_k=5):
    """Demonstrate image retrieval using learned embeddings"""
    log_action("Performing image retrieval demonstration")
    
    model.eval()
    embeddings = []
    all_images = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images_device = images.to(device)
            emb = model.forward_single(images_device)
            embeddings.append(emb.cpu())
            all_images.append(images)
            
            if len(embeddings) * dataloader.batch_size >= 1000:  # Limit for demo
                break
    
    embeddings = torch.cat(embeddings)
    all_images = torch.cat(all_images)
    
    # Get query embedding
    query_embedding = embeddings[query_idx]
    
    # Calculate distances to all other embeddings
    distances = torch.norm(embeddings - query_embedding.unsqueeze(0), dim=1)
    
    # Get top-k nearest neighbors
    _, indices = torch.topk(distances, k=top_k+1, largest=False)
    indices = indices[1:]  # Remove query itself
    
    # Visualize results
    fig, axes = plt.subplots(1, top_k+1, figsize=(15, 3))
    
    # Show query image
    query_img = all_images[query_idx].permute(1, 2, 0).numpy()
    query_img = (query_img * 0.229 + 0.485)  # Denormalize
    axes[0].imshow(np.clip(query_img, 0, 1))
    axes[0].set_title('Query Image')
    axes[0].axis('off')
    
    # Show retrieved images
    for i, idx in enumerate(indices):
        img = all_images[idx].permute(1, 2, 0).numpy()
        img = (img * 0.229 + 0.485)  # Denormalize
        axes[i+1].imshow(np.clip(img, 0, 1))
        axes[i+1].set_title(f'Rank {i+1}\nDist: {distances[idx]:.3f}')
        axes[i+1].axis('off')
    
    plt.suptitle('Image Retrieval Results')
    plt.tight_layout()
    plt.savefig('image_retrieval_demo.png', dpi=300, bbox_inches='tight')
    log_action("Saved image retrieval demo to image_retrieval_demo.png", "SUCCESS")
    plt.close()

def main():
    """Main execution function"""
    log_action("=== Siamese Neural Networks and Similarity Learning Demo ===", "INFO")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Data preparation
    log_action("Preparing datasets")
    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load CIFAR-10 as example dataset
    log_action("Loading CIFAR-10 dataset")
    base_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )
    base_test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create Siamese datasets
    log_action("Creating Siamese pair dataset for Contrastive Loss")
    siamese_train_dataset = SiamesePairDataset(base_train_dataset, transform=transform)
    siamese_train_loader = DataLoader(siamese_train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    log_action("Creating Triplet dataset for Triplet Loss")
    triplet_train_dataset = TripletDataset(base_train_dataset, transform=transform)
    triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    test_loader = DataLoader(base_test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train with Contrastive Loss
    log_action("Training Model 1 with Contrastive Loss")
    model_contrastive = SiameseNetwork(embedding_dim=128)
    model_contrastive = train_contrastive(model_contrastive, siamese_train_loader, epochs=5)
    
    # Train with Triplet Loss
    log_action("Training Model 2 with Triplet Loss")
    model_triplet = SiameseNetwork(embedding_dim=128)
    model_triplet = train_triplet(model_triplet, triplet_train_loader, epochs=5)
    
    # Demonstrate Hard Negative Mining
    log_action("Demonstrating Hard Negative Mining")
    miner = HardNegativeMiner(model_triplet, test_loader, device)
    hard_triplets = miner.mine_hard_triplets(num_triplets=100)
    log_action(f"Found {len(hard_triplets)} hard triplets for further training")
    
    # Visualizations and applications
    log_action("Generating visualizations and demonstrations")
    
    # Visualize embeddings
    visualize_embeddings(model_triplet, test_loader, device)
    
    # Perform clustering
    cluster_labels, embeddings = perform_clustering(model_triplet, test_loader, device)
    
    # Image retrieval demo
    image_retrieval_demo(model_triplet, test_loader, device)
    
    # Save models
    log_action("Saving trained models")
    torch.save(model_contrastive.state_dict(), 'siamese_contrastive.pth')
    torch.save(model_triplet.state_dict(), 'siamese_triplet.pth')
    log_action("Models saved successfully", "SUCCESS")
    
    log_action("=== Siamese Neural Networks Demo Completed Successfully ===", "SUCCESS")
    log_action("Generated files:", "INFO")
    log_action("  - siamese_contrastive.pth (Contrastive Loss model)", "INFO")
    log_action("  - siamese_triplet.pth (Triplet Loss model)", "INFO")
    log_action("  - embeddings_visualization.png (t-SNE visualization)", "INFO")
    log_action("  - image_retrieval_demo.png (Image retrieval results)", "INFO")

if __name__ == "__main__":
    main()
