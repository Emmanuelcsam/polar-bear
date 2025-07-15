#!/usr/bin/env python3
"""
Training script demonstrating how to use the enhanced fiber anomaly detection
system with Siamese networks and similarity learning.
"""

import os
import numpy as np
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from enhanced_detection import DeepFiberAnalyzer, OmniConfig, FiberDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class FiberTripletDataset(FiberDataset):
    """Dataset that returns triplets for training"""
    
    def __init__(self, image_paths, labels, transform=None):
        super().__init__(image_paths, labels, transform)
        
        # Organize images by label
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
            
        self.labels_set = list(self.label_to_indices.keys())
        
    def __getitem__(self, idx):
        # Get anchor
        anchor_img, anchor_label = super().__getitem__(idx)
        
        # Get positive (same class)
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = np.random.choice(self.label_to_indices[anchor_label])
        positive_img, _ = super().__getitem__(positive_idx)
        
        # Get negative (different class)
        negative_label = np.random.choice([l for l in self.labels_set if l != anchor_label])
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        negative_img, _ = super().__getitem__(negative_idx)
        
        return anchor_img, positive_img, negative_img


def prepare_fiber_data(data_dir):
    """
    Prepare fiber optic data for training.
    Expected directory structure:
    data_dir/
        normal/
            fiber1.png
            fiber2.png
            ...
        scratched/
            fiber1.png
            ...
        contaminated/
            fiber1.png
            ...
    """
    image_paths = []
    labels = []
    
    # Define defect types
    defect_types = ['normal', 'scratched', 'contaminated', 'chipped', 'dirty']
    
    for defect_type in defect_types:
        defect_dir = Path(data_dir) / defect_type
        if defect_dir.exists():
            for img_path in defect_dir.glob('*.png'):
                image_paths.append(str(img_path))
                labels.append(defect_type)
                
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(set(labels))}
    labels = [label_to_int[label] for label in labels]
    
    # Split into train/val sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    return train_paths, val_paths, train_labels, val_labels


def visualize_embeddings(analyzer, data_loader, save_path='embeddings_viz.png'):
    """Visualize learned embeddings using t-SNE"""
    from sklearn.manifold import TSNE
    
    embeddings = []
    labels = []
    
    # Extract embeddings
    analyzer.siamese_net.eval()
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 2:
                images, batch_labels = batch
            else:
                images = batch[0]  # Just use anchors for triplets
                batch_labels = None
                
            images = images.to(analyzer.device)
            emb = analyzer.siamese_net.forward_once(images)
            embeddings.append(emb.cpu().numpy())
            
            if batch_labels is not None:
                labels.extend(batch_labels.numpy())
    
    # Concatenate all embeddings
    embeddings = np.vstack(embeddings)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    if labels:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    
    plt.title('Fiber Embeddings Visualization (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved embedding visualization to {save_path}")


def train_with_hard_mining(analyzer, train_loader, val_loader, epochs=50):
    """Training with hard negative mining as described in the tutorial"""
    from enhanced_detection import TripletLoss
    import torch.nn.functional as F
    
    model = analyzer.siamese_net
    optimizer = torch.optim.Adam(model.parameters(), lr=analyzer.config.learning_rate)
    triplet_loss = TripletLoss(margin=analyzer.config.margin)
    
    # Phase 1: Random sampling for initial epochs
    print("Phase 1: Training with random triplets...")
    for epoch in range(epochs // 2):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            anchor, positive, negative = batch
            anchor = anchor.to(analyzer.device)
            positive = positive.to(analyzer.device)
            negative = negative.to(analyzer.device)
            
            # Forward pass
            anchor_emb, pos_emb, neg_emb = model(anchor, positive, negative)
            
            # Compute loss
            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Phase 2: Hard negative mining
    print("\nPhase 2: Training with hard negatives...")
    for epoch in range(epochs // 2, epochs):
        model.train()
        total_loss = 0
        hard_triplets = 0
        
        for batch in train_loader:
            anchor, positive, negative = batch
            anchor = anchor.to(analyzer.device)
            positive = positive.to(analyzer.device)
            negative = negative.to(analyzer.device)
            
            # Forward pass to find hard cases
            with torch.no_grad():
                anchor_emb, pos_emb, neg_emb = model(anchor, positive, negative)
                pos_dist = F.pairwise_distance(anchor_emb, pos_emb)
                neg_dist = F.pairwise_distance(anchor_emb, neg_emb)
                
                # Identify hard triplets (where pos_dist > neg_dist - margin)
                hard_mask = pos_dist > neg_dist - analyzer.config.margin * 0.5
                
            # Only train on hard triplets
            if hard_mask.any():
                hard_anchors = anchor[hard_mask]
                hard_positives = positive[hard_mask]
                hard_negatives = negative[hard_mask]
                
                # Forward pass on hard triplets
                anchor_emb, pos_emb, neg_emb = model(hard_anchors, hard_positives, hard_negatives)
                
                # Compute loss
                loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                hard_triplets += hard_mask.sum().item()
                
        avg_loss = total_loss / max(1, hard_triplets)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Hard triplets = {hard_triplets}")
        
    return model


def evaluate_model(analyzer, test_loader):
    """Evaluate the trained model"""
    model = analyzer.siamese_net
    model.eval()
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:  # Triplet
                anchor, positive, negative = batch
                anchor = anchor.to(analyzer.device)
                positive = positive.to(analyzer.device)
                negative = negative.to(analyzer.device)
                
                # Get embeddings
                anchor_emb, pos_emb, neg_emb = model(anchor, positive, negative)
                
                # Compute distances
                pos_dist = F.pairwise_distance(anchor_emb, pos_emb)
                neg_dist = F.pairwise_distance(anchor_emb, neg_emb)
                
                # Check if positive is closer than negative
                correct = (pos_dist < neg_dist).sum().item()
                correct_predictions += correct
                total_predictions += anchor.size(0)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Triplet accuracy: {accuracy:.2%}")
    return accuracy


def main():
    """Main training pipeline"""
    # Configuration
    config = OmniConfig(
        use_deep_features=True,
        embedding_dim=128,
        backbone='resnet50',
        batch_size=32,
        learning_rate=0.001,
        margin=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Data directory
    data_dir = input("Enter fiber data directory: ").strip()
    
    # Prepare data
    print("Preparing data...")
    train_paths, val_paths, train_labels, val_labels = prepare_fiber_data(data_dir)
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FiberTripletDataset(train_paths, train_labels, transform)
    val_dataset = FiberDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                          shuffle=False, num_workers=4)
    
    # Initialize analyzer
    analyzer = DeepFiberAnalyzer(config)
    
    # Train with hard negative mining
    print("\nStarting training...")
    trained_model = train_with_hard_mining(analyzer, train_loader, val_loader, epochs=50)
    
    # Visualize embeddings
    print("\nVisualizing embeddings...")
    visualize_embeddings(analyzer, val_loader)
    
    # Evaluate model
    print("\nEvaluating model...")
    val_triplet_dataset = FiberTripletDataset(val_paths, val_labels, val_transform)
    val_triplet_loader = DataLoader(val_triplet_dataset, batch_size=config.batch_size, 
                                  shuffle=False, num_workers=4)
    evaluate_model(analyzer, val_triplet_loader)
    
    # Save model
    save_path = Path('trained_models')
    save_path.mkdir(exist_ok=True)
    torch.save(trained_model.state_dict(), save_path / 'fiber_siamese_model.pth')
    print(f"\nModel saved to {save_path / 'fiber_siamese_model.pth'}")
    
    # Example: Analyze a test fiber
    test_image = input("\nEnter test image path (or press Enter to skip): ").strip()
    if test_image and os.path.exists(test_image):
        print(f"\nAnalyzing {test_image}...")
        output_dir = f"similarity_analysis_{Path(test_image).stem}"
        results = analyzer.analyze_end_face(test_image, output_dir)
        
        if results and 'deep_analysis' in results:
            print(f"\nDeep Analysis Results:")
            print(f"  Learned Similarity: {results['deep_analysis']['learned_similarity']:.3f}")
            print(f"  Cosine Similarity: {results['deep_analysis']['cosine_similarity']:.3f}")
            print(f"  Anomaly Detected: {results['verdict']['is_anomalous']}")
            print(f"  Confidence: {results['verdict']['confidence']:.2%}")


if __name__ == "__main__":
    main()
