#!/usr/bin/env python3
"""
CNN Visualization and Interpretability Script
Based on the lecture transcript on visualization techniques for CNNs
"""

import sys
import subprocess
import importlib
import os
from datetime import datetime

# Auto-installation function
def install_package(package_name, import_name=None):
    """Auto-install packages if not available"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"[{datetime.now()}] ✓ {package_name} is already installed")
        return True
    except ImportError:
        print(f"[{datetime.now()}] ⚠ {package_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            print(f"[{datetime.now()}] ✓ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            print(f"[{datetime.now()}] ✗ Failed to install {package_name}")
            return False

# Check and install required packages
print(f"[{datetime.now()}] Starting CNN Visualization Script")
print(f"[{datetime.now()}] Checking and installing required packages...")

required_packages = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("matplotlib", "matplotlib"),
    ("opencv-python", "cv2"),
    ("scikit-learn", "sklearn"),
    ("pillow", "PIL"),
    ("scipy", "scipy"),
    ("tqdm", "tqdm")
]

for package, import_name in required_packages:
    if not install_package(package, import_name):
        print(f"[{datetime.now()}] ✗ Failed to set up required packages. Exiting.")
        sys.exit(1)

# Import all required modules
print(f"[{datetime.now()}] Importing required modules...")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import cv2
from sklearn.manifold import TSNE
from PIL import Image
import scipy.ndimage as ndimage
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"[{datetime.now()}] ✓ All modules imported successfully")

class CNNVisualizer:
    """Main class for CNN visualization techniques"""
    
    def __init__(self, model=None, device=None):
        print(f"[{datetime.now()}] Initializing CNN Visualizer...")
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"[{datetime.now()}] Using device: {self.device}")
        
        # Load model
        if model is None:
            print(f"[{datetime.now()}] Loading pre-trained AlexNet model...")
            self.model = models.alexnet(pretrained=True)
            print(f"[{datetime.now()}] ✓ AlexNet loaded successfully")
        else:
            self.model = model
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Store activations and gradients
        self.activations = {}
        self.gradients = {}
        self.max_locations = {}  # For unpooling in DeconvNet
        
        print(f"[{datetime.now()}] ✓ CNN Visualizer initialized")
    
    def register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""
        print(f"[{datetime.now()}] Registering hooks for activation and gradient capture...")
        
        def forward_hook(module, input, output, name):
            self.activations[name] = output.detach()
            
        def backward_hook(module, grad_input, grad_output, name):
            self.gradients[name] = grad_output[0].detach()
            
        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
                module.register_forward_hook(lambda m, i, o, n=name: forward_hook(m, i, o, n))
                module.register_backward_hook(lambda m, gi, go, n=name: backward_hook(m, gi, go, n))
                print(f"[{datetime.now()}] Registered hooks for layer: {name}")
        
        print(f"[{datetime.now()}] ✓ Hooks registered successfully")
    
    def visualize_image_patches(self, dataloader, layer_name, filter_idx, num_patches=9):
        """
        Visualize image patches that maximally activate a specific filter
        Based on the image space visualization technique from the lecture
        """
        print(f"[{datetime.now()}] Starting image patch visualization...")
        print(f"[{datetime.now()}] Target layer: {layer_name}, Filter index: {filter_idx}")
        
        self.register_hooks()
        
        # Collect activations for all images
        max_activations = []
        corresponding_patches = []
        
        print(f"[{datetime.now()}] Processing images to find maximal activations...")
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Processing batches")):
                images = images.to(self.device)
                _ = self.model(images)
                
                if layer_name in self.activations:
                    activation = self.activations[layer_name]
                    
                    # Get activation for specific filter
                    filter_activation = activation[:, filter_idx, :, :]
                    
                    # Find maximum activation locations
                    for img_idx in range(images.shape[0]):
                        act_map = filter_activation[img_idx]
                        max_val = act_map.max().item()
                        
                        if max_val > 0:
                            # Find location of maximum
                            max_loc = (act_map == max_val).nonzero()[0]
                            y, x = max_loc[0].item(), max_loc[1].item()
                            
                            # Calculate receptive field in input image
                            # This is simplified - actual calculation depends on network architecture
                            patch_size = 11  # For AlexNet first layer
                            stride = 4
                            
                            y_start = y * stride
                            x_start = x * stride
                            y_end = min(y_start + patch_size, images.shape[2])
                            x_end = min(x_start + patch_size, images.shape[3])
                            
                            patch = images[img_idx, :, y_start:y_end, x_start:x_end]
                            
                            max_activations.append(max_val)
                            corresponding_patches.append(patch.cpu())
        
        # Sort patches by activation strength
        sorted_indices = np.argsort(max_activations)[::-1][:num_patches]
        top_patches = [corresponding_patches[i] for i in sorted_indices]
        
        print(f"[{datetime.now()}] Visualizing top {num_patches} patches...")
        
        # Visualize patches
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(f'Top {num_patches} patches for {layer_name}, filter {filter_idx}')
        
        for idx, (ax, patch) in enumerate(zip(axes.flat, top_patches)):
            if patch.shape[0] == 3:  # RGB image
                patch = patch.permute(1, 2, 0)
                patch = (patch - patch.min()) / (patch.max() - patch.min())
                ax.imshow(patch)
            else:
                ax.imshow(patch.squeeze(), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Patch {idx+1}')
        
        plt.tight_layout()
        plt.savefig(f'image_patches_{layer_name}_{filter_idx}.png')
        print(f"[{datetime.now()}] ✓ Saved visualization to image_patches_{layer_name}_{filter_idx}.png")
        plt.close()
    
    def occlusion_experiment(self, image, target_class, occlusion_size=50, stride=10):
        """
        Perform occlusion experiment to identify important image regions
        Based on the occlusion experiment technique from the lecture
        """
        print(f"[{datetime.now()}] Starting occlusion experiment...")
        print(f"[{datetime.now()}] Occlusion size: {occlusion_size}x{occlusion_size}, Stride: {stride}")
        
        # Prepare image
        if isinstance(image, str):
            print(f"[{datetime.now()}] Loading image from: {image}")
            img = Image.open(image).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            original_img = np.array(img.resize((224, 224)))
        else:
            img_tensor = image.to(self.device)
            original_img = image.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(img_tensor)
            baseline_prob = F.softmax(baseline_output, dim=1)[0, target_class].item()
        
        print(f"[{datetime.now()}] Baseline probability for class {target_class}: {baseline_prob:.4f}")
        
        # Create occlusion map
        height, width = 224, 224
        occlusion_map = np.ones((height, width)) * baseline_prob
        
        print(f"[{datetime.now()}] Performing occlusion sweep...")
        
        # Slide occlusion window
        for y in tqdm(range(0, height - occlusion_size + 1, stride), desc="Y-axis"):
            for x in range(0, width - occlusion_size + 1, stride):
                # Create occluded image
                occluded = img_tensor.clone()
                occluded[:, :, y:y+occlusion_size, x:x+occlusion_size] = 0
                
                # Get prediction for occluded image
                with torch.no_grad():
                    occluded_output = self.model(occluded)
                    occluded_prob = F.softmax(occluded_output, dim=1)[0, target_class].item()
                
                # Update occlusion map
                occlusion_map[y:y+occlusion_size, x:x+occlusion_size] = np.minimum(
                    occlusion_map[y:y+occlusion_size, x:x+occlusion_size],
                    occluded_prob
                )
        
        print(f"[{datetime.now()}] Creating visualization...")
        
        # Visualize results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(original_img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        im = ax2.imshow(occlusion_map, cmap='hot', interpolation='bilinear')
        ax2.set_title(f'Occlusion Sensitivity Map (Class {target_class})')
        ax2.axis('off')
        
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f'occlusion_map_class_{target_class}.png')
        print(f"[{datetime.now()}] ✓ Saved occlusion map to occlusion_map_class_{target_class}.png")
        plt.close()
        
        return occlusion_map
    
    def deconvnet_visualization(self, image, layer_name, filter_idx):
        """
        DeconvNet visualization - project activations back to image space
        Based on the DeconvNet technique from the lecture
        """
        print(f"[{datetime.now()}] Starting DeconvNet visualization...")
        print(f"[{datetime.now()}] Target layer: {layer_name}, Filter: {filter_idx}")
        
        # This is a simplified implementation
        # Full DeconvNet requires implementing unpooling, deconvolution, etc.
        
        class DeconvNet(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.features = []
                self.pooling_indices = []
                
                # Build deconv layers (simplified)
                for module in original_model.features:
                    if isinstance(module, nn.Conv2d):
                        # Create transposed convolution
                        deconv = nn.ConvTranspose2d(
                            module.out_channels,
                            module.in_channels,
                            module.kernel_size,
                            module.stride,
                            module.padding
                        )
                        # Use transposed weights
                        deconv.weight.data = module.weight.data.permute(1, 0, 2, 3)
                        self.features.append(deconv)
                    elif isinstance(module, nn.MaxPool2d):
                        # Store for unpooling
                        self.features.append(nn.MaxUnpool2d(module.kernel_size, module.stride))
            
            def forward(self, x, layer_idx, filter_idx):
                # Simplified forward pass
                # In practice, this would involve proper unpooling with stored indices
                activations = x.clone()
                
                # Zero out all filters except the one we're interested in
                activations[:, :filter_idx] = 0
                activations[:, filter_idx+1:] = 0
                
                # Apply ReLU to keep only positive activations
                activations = F.relu(activations)
                
                # Simplified deconvolution (just upsampling for demo)
                for i in range(layer_idx):
                    activations = F.interpolate(activations, scale_factor=2, mode='nearest')
                    activations = F.relu(activations)
                
                return activations
        
        print(f"[{datetime.now()}] Building DeconvNet architecture...")
        deconv_net = DeconvNet(self.model)
        
        # Forward pass to get activations
        self.register_hooks()
        
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)
        else:
            img_tensor = image.to(self.device)
        
        print(f"[{datetime.now()}] Forward pass through network...")
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        if layer_name in self.activations:
            activation = self.activations[layer_name]
            print(f"[{datetime.now()}] Applying DeconvNet operations...")
            
            # Simple visualization of the activation
            filter_activation = activation[0, filter_idx].cpu().numpy()
            
            # Normalize and resize
            filter_activation = (filter_activation - filter_activation.min()) / (filter_activation.max() - filter_activation.min() + 1e-8)
            filter_activation = cv2.resize(filter_activation, (224, 224))
            
            print(f"[{datetime.now()}] Creating visualization...")
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img_tensor.squeeze().permute(1, 2, 0).cpu())
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(filter_activation, cmap='hot')
            plt.title(f'DeconvNet Visualization\n{layer_name}, Filter {filter_idx}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'deconvnet_{layer_name}_{filter_idx}.png')
            print(f"[{datetime.now()}] ✓ Saved DeconvNet visualization to deconvnet_{layer_name}_{filter_idx}.png")
            plt.close()
    
    def gradient_ascent_visualization(self, target_class, num_iterations=500, lr=0.1):
        """
        Generate synthetic images that maximize class scores using gradient ascent
        Based on the gradient ascent technique from the lecture
        """
        print(f"[{datetime.now()}] Starting gradient ascent visualization...")
        print(f"[{datetime.now()}] Target class: {target_class}, Iterations: {num_iterations}, Learning rate: {lr}")
        
        # Start with random noise
        print(f"[{datetime.now()}] Initializing with random noise image...")
        img = torch.randn(1, 3, 224, 224).to(self.device) * 0.01
        img.requires_grad = True
        
        print(f"[{datetime.now()}] Starting optimization loop...")
        
        for i in tqdm(range(num_iterations), desc="Gradient ascent"):
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(img)
            
            # Get score for target class (before softmax)
            class_score = output[0, target_class]
            
            # L2 regularization
            l2_reg = 0.0001 * torch.sum(img ** 2)
            
            # Total loss (negative because we want to maximize)
            loss = class_score - l2_reg
            
            # Backward pass
            loss.backward()
            
            # Update image
            with torch.no_grad():
                img += lr * img.grad
                
                # Apply regularizations as mentioned in the lecture
                # Gaussian blur
                if i % 10 == 0:
                    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
                    for c in range(3):
                        img_np[:, :, c] = ndimage.gaussian_filter(img_np[:, :, c], sigma=0.5)
                    img = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    img.requires_grad = True
                
                # Clip small values
                img = torch.where(torch.abs(img) < 0.1, torch.zeros_like(img), img)
                
                # Clip gradients
                if img.grad is not None:
                    img.grad = torch.where(torch.abs(img.grad) < 0.01, torch.zeros_like(img.grad), img.grad)
            
            if i % 100 == 0:
                print(f"[{datetime.now()}] Iteration {i}: Class score = {class_score.item():.4f}")
        
        print(f"[{datetime.now()}] Optimization complete. Creating visualization...")
        
        # Normalize for visualization
        img_vis = img.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_vis)
        plt.title(f'Gradient Ascent Visualization\nClass {target_class}')
        plt.axis('off')
        plt.savefig(f'gradient_ascent_class_{target_class}.png')
        print(f"[{datetime.now()}] ✓ Saved gradient ascent visualization to gradient_ascent_class_{target_class}.png")
        plt.close()
        
        return img
    
    def deep_dream(self, image, layer_name, iterations=10, lr=0.01):
        """
        Deep Dream - amplify features detected by the network
        Based on the Deep Dream technique from the lecture
        """
        print(f"[{datetime.now()}] Starting Deep Dream...")
        print(f"[{datetime.now()}] Target layer: {layer_name}, Iterations: {iterations}")
        
        # Load and preprocess image
        if isinstance(image, str):
            print(f"[{datetime.now()}] Loading image from: {image}")
            img = Image.open(image).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)
        else:
            img_tensor = image.to(self.device)
        
        img_tensor.requires_grad = True
        
        print(f"[{datetime.now()}] Starting Deep Dream iterations...")
        
        for i in tqdm(range(iterations), desc="Deep Dream"):
            self.model.zero_grad()
            self.activations.clear()
            
            # Forward pass
            self.register_hooks()
            _ = self.model(img_tensor)
            
            if layer_name in self.activations:
                activation = self.activations[layer_name]
                
                # Set gradients equal to activations (amplify what the network sees)
                activation.backward(activation)
                
                # Update image
                with torch.no_grad():
                    img_tensor += lr * img_tensor.grad
                    img_tensor.grad.zero_()
                    
                    # Clip values to valid range
                    img_tensor = torch.clamp(img_tensor, 0, 1)
        
        print(f"[{datetime.now()}] Deep Dream complete. Creating visualization...")
        
        # Visualize result
        result = img_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        if isinstance(image, str):
            plt.imshow(img)
        else:
            plt.imshow(image.squeeze().permute(1, 2, 0).cpu())
        plt.title('Original Image')
        plt.axis('off')
        
        # Deep Dream result
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title(f'Deep Dream Result\nLayer: {layer_name}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'deep_dream_{layer_name}.png')
        print(f"[{datetime.now()}] ✓ Saved Deep Dream result to deep_dream_{layer_name}.png")
        plt.close()
        
        return img_tensor
    
    def tsne_visualization(self, dataloader, layer_name='classifier.6', num_samples=1000):
        """
        t-SNE visualization of feature space
        Based on the t-SNE visualization technique from the lecture
        """
        print(f"[{datetime.now()}] Starting t-SNE visualization...")
        print(f"[{datetime.now()}] Target layer: {layer_name}, Number of samples: {num_samples}")
        
        features = []
        labels = []
        
        self.register_hooks()
        
        print(f"[{datetime.now()}] Extracting features from data...")
        
        with torch.no_grad():
            sample_count = 0
            for images, targets in tqdm(dataloader, desc="Extracting features"):
                if sample_count >= num_samples:
                    break
                
                images = images.to(self.device)
                _ = self.model(images)
                
                if layer_name in self.activations:
                    feat = self.activations[layer_name].cpu().numpy()
                    feat = feat.reshape(feat.shape[0], -1)  # Flatten
                    features.append(feat)
                    labels.extend(targets.numpy())
                    sample_count += images.shape[0]
        
        features = np.vstack(features)[:num_samples]
        labels = np.array(labels)[:num_samples]
        
        print(f"[{datetime.now()}] Feature shape: {features.shape}")
        print(f"[{datetime.now()}] Running t-SNE dimensionality reduction...")
        
        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_2d = tsne.fit_transform(features)
        
        print(f"[{datetime.now()}] Creating visualization...")
        
        # Visualize
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Class')
        plt.title(f't-SNE Visualization of Features\nLayer: {layer_name}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        plt.tight_layout()
        plt.savefig(f'tsne_visualization_{layer_name}.png', dpi=300)
        print(f"[{datetime.now()}] ✓ Saved t-SNE visualization to tsne_visualization_{layer_name}.png")
        plt.close()
        
        return features_2d, labels

def main():
    """Main function to demonstrate all visualization techniques"""
    print(f"[{datetime.now()}] " + "="*50)
    print(f"[{datetime.now()}] CNN VISUALIZATION AND INTERPRETABILITY DEMO")
    print(f"[{datetime.now()}] " + "="*50)
    
    # Initialize visualizer
    visualizer = CNNVisualizer()
    
    # Load sample data
    print(f"[{datetime.now()}] Loading CIFAR-10 dataset for demonstrations...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Download CIFAR-10 if not available
    try:
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                              download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, 
                                                shuffle=True, num_workers=2)
        print(f"[{datetime.now()}] ✓ Dataset loaded successfully")
    except Exception as e:
        print(f"[{datetime.now()}] ✗ Error loading dataset: {e}")
        return
    
    # Get a sample image
    sample_image, sample_label = dataset[0]
    sample_image = sample_image.unsqueeze(0)
    
    print(f"[{datetime.now()}] " + "-"*50)
    print(f"[{datetime.now()}] Running visualization techniques...")
    print(f"[{datetime.now()}] " + "-"*50)
    
    # 1. Image Patch Visualization
    print(f"\n[{datetime.now()}] 1. IMAGE PATCH VISUALIZATION")
    try:
        visualizer.visualize_image_patches(dataloader, 'features.0', 0, num_patches=9)
    except Exception as e:
        print(f"[{datetime.now()}] ✗ Error in image patch visualization: {e}")
    
    # 2. Occlusion Experiment
    print(f"\n[{datetime.now()}] 2. OCCLUSION EXPERIMENT")
    try:
        visualizer.occlusion_experiment(sample_image, target_class=sample_label, 
                                       occlusion_size=30, stride=10)
    except Exception as e:
        print(f"[{datetime.now()}] ✗ Error in occlusion experiment: {e}")
    
    # 3. DeconvNet Visualization
    print(f"\n[{datetime.now()}] 3. DECONVNET VISUALIZATION")
    try:
        visualizer.deconvnet_visualization(sample_image, 'features.0', 0)
    except Exception as e:
        print(f"[{datetime.now()}] ✗ Error in DeconvNet visualization: {e}")
    
    # 4. Gradient Ascent Visualization
    print(f"\n[{datetime.now()}] 4. GRADIENT ASCENT VISUALIZATION")
    try:
        # Visualize for dog class (5 in CIFAR-10)
        visualizer.gradient_ascent_visualization(target_class=5, num_iterations=200, lr=0.1)
    except Exception as e:
        print(f"[{datetime.now()}] ✗ Error in gradient ascent visualization: {e}")
    
    # 5. Deep Dream
    print(f"\n[{datetime.now()}] 5. DEEP DREAM")
    try:
        visualizer.deep_dream(sample_image, 'features.3', iterations=10, lr=0.01)
    except Exception as e:
        print(f"[{datetime.now()}] ✗ Error in Deep Dream: {e}")
    
    # 6. t-SNE Visualization
    print(f"\n[{datetime.now()}] 6. t-SNE VISUALIZATION")
    try:
        visualizer.tsne_visualization(dataloader, layer_name='classifier.6', num_samples=500)
    except Exception as e:
        print(f"[{datetime.now()}] ✗ Error in t-SNE visualization: {e}")
    
    print(f"\n[{datetime.now()}] " + "="*50)
    print(f"[{datetime.now()}] VISUALIZATION COMPLETE!")
    print(f"[{datetime.now()}] Check the generated PNG files for results")
    print(f"[{datetime.now()}] " + "="*50)

if __name__ == "__main__":
    main()
