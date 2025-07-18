import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

# Ask for image paths
ref_path = input("Enter the path of the reference (main image): ")
feat_path = input("Enter the path of the feature (part contained): ")

print(f"Loading reference image: {ref_path}")
print(f"Loading feature image: {feat_path}")

# Load and normalize images
try:
    ref = plt.imread(ref_path).astype(np.float32)
    feat = plt.imread(feat_path).astype(np.float32)
    
    # Normalize to [0,1] if needed
    if ref.max() > 1:
        ref = ref / 255.0
    if feat.max() > 1:
        feat = feat / 255.0
        
    print(f"Reference image shape: {ref.shape}")
    print(f"Feature image shape: {feat.shape}")
    
except Exception as e:
    print(f"Error loading images: {e}")
    sys.exit(1)

# Convert to grayscale
if len(ref.shape) == 3 and ref.shape[2] >= 3:
    ref_gray = np.dot(ref[..., :3], [0.299, 0.587, 0.114])
else:
    ref_gray = ref if len(ref.shape) == 2 else ref[..., 0]

if len(feat.shape) == 3 and feat.shape[2] >= 3:
    feat_gray = np.dot(feat[..., :3], [0.299, 0.587, 0.114])
else:
    feat_gray = feat if len(feat.shape) == 2 else feat[..., 0]

print(f"Reference grayscale shape: {ref_gray.shape}")
print(f"Feature grayscale shape: {feat_gray.shape}")

# Get dimensions
H, W = ref_gray.shape
h, w = feat_gray.shape

print(f"Reference: {H}x{W}, Feature: {h}x{w}")

if h > H or w > W:
    print("Feature not contained - feature is larger than reference")
    sys.exit(0)

# Prepare tensors
ref_tensor = torch.from_numpy(ref_gray).float().unsqueeze(0).unsqueeze(0)
feat_tensor = torch.from_numpy(feat_gray).float()

print(f"Reference tensor shape: {ref_tensor.shape}")
print(f"Feature tensor shape: {feat_tensor.shape}")

# Base grid for feature
yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
base_grid = torch.from_numpy(np.stack((xx, yy), axis=-1)).float()

# Initial random position
x_pos = torch.tensor(float(np.random.uniform(0, W - w)), requires_grad=True)
y_pos = torch.tensor(float(np.random.uniform(0, H - h)), requires_grad=True)

print(f"Initial position: ({x_pos.item():.2f}, {y_pos.item():.2f})")

optimizer = optim.Adam([x_pos, y_pos], lr=5.0)

best_sim = -float('inf')
best_loss = float('inf')

for iteration in range(300):
    optimizer.zero_grad()
    
    # Create grid for sampling
    offsets = torch.tensor([x_pos, y_pos]).view(1, 1, 2)
    current_points = base_grid + offsets
    
    # Normalize coordinates to [-1, 1]
    norm_x = 2 * current_points[..., 0] / (W - 1) - 1
    norm_y = 2 * current_points[..., 1] / (H - 1) - 1
    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
    
    # Sample from reference image
    crop = F.grid_sample(ref_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze()
    
    # Compute normalized cross-correlation
    crop_flat = crop.flatten()
    feat_flat = feat_tensor.flatten().detach()  # Detach to prevent gradient issues
    
    # Compute means
    crop_mean = crop_flat.mean()
    feat_mean = feat_flat.mean()
    
    # Center the data
    crop_centered = crop_flat - crop_mean
    feat_centered = feat_flat - feat_mean
    
    # Compute correlation
    numerator = torch.sum(crop_centered * feat_centered)
    denom_crop = torch.sqrt(torch.sum(crop_centered**2))
    denom_feat = torch.sqrt(torch.sum(feat_centered**2))
    
    # Avoid division by zero
    correlation = numerator / (denom_crop * denom_feat + 1e-8)
    
    # Loss is negative correlation (we want to maximize correlation)
    loss = -correlation
    
    # Keep track of best similarity
    current_sim = correlation.item()
    if current_sim > best_sim:
        best_sim = current_sim
        best_loss = loss.item()
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Clamp positions to valid ranges
    with torch.no_grad():
        x_pos.clamp_(0, W - w)
        y_pos.clamp_(0, H - h)
    
    if iteration % 50 == 0:
        print(f"Iteration {iteration}: correlation = {current_sim:.4f}, position = ({x_pos.item():.1f}, {y_pos.item():.1f})")

# Final similarity
final_sim = best_sim
print(f"\nFinal similarity: {final_sim:.4f}")

if final_sim > 0.6:  # Lowered threshold for testing
    print("Feature contained")
    # Get final position
    final_x = int(round(x_pos.item()))
    final_y = int(round(y_pos.item()))
    print(f"Found at position: ({final_x}, {final_y})")
    
    # Create visualization
    if len(ref.shape) == 3:
        overlay = np.copy(ref)
        # Create high contrast feature for visualization
        if len(feat.shape) == 3:
            high_contrast_feat = 1 - feat
        else:
            high_contrast_feat = np.stack([1 - feat_gray] * 3, axis=-1)
        overlay[final_y:final_y + h, final_x:final_x + w] = high_contrast_feat
    else:
        overlay = np.copy(ref_gray)
        overlay[final_y:final_y + h, final_x:final_x + w] = 1 - feat_gray
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(ref, cmap='gray' if len(ref.shape) == 2 else None)
    plt.title('Original Reference')
    plt.subplot(1, 2, 2)
    plt.imshow(overlay, cmap='gray' if len(overlay.shape) == 2 else None)
    plt.title(f'Feature Found (similarity: {final_sim:.3f})')
    plt.tight_layout()
    plt.show()
else:
    print("Feature not contained")
    print(f"Best similarity achieved: {final_sim:.4f} (threshold: 0.6)")
