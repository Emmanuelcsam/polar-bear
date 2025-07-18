import numpy as np
import torch import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ask for image paths
ref_path = input("Enter the path of the reference (main image): ")
feat_path = input("Enter the path of the feature (part contained): ")

# Load and normalize images
ref = plt.imread(ref_path).astype(np.float32) / 255 if plt.imread(ref_path).max() > 1 else plt.imread(ref_path)
feat = plt.imread(feat_path).astype(np.float32) / 255 if plt.imread(feat_path).max() > 1 else plt.imread(feat_path)

# Convert to grayscale
ref_gray = np.dot(ref[..., :3], [0.299, 0.587, 0.114])
feat_gray = np.dot(feat[..., :3], [0.299, 0.587, 0.114])

# Get dimensions
H, W = ref_gray.shape
h, w = feat_gray.shape

if h > H or w > W:
    print("Feature not contained")
else:
    # Prepare tensors
    ref_tensor = torch.from_numpy(ref_gray).float().unsqueeze(0).unsqueeze(0)
    feat_tensor = torch.from_numpy(feat_gray).float()

    # Base grid for feature
    yy, xx = np.meshgrid(np.arange(h), np.arange(w))
    base_grid = torch.from_numpy(np.stack((xx, yy), axis=-1)).float()  # h w 2

    # Initial random position
    x_pos = torch.tensor(np.random.uniform(0, high=W - w), requires_grad=True)
    y_pos = torch.tensor(np.random.uniform(0, high=H - h), requires_grad=True)

    optimizer = optim.Adam([x_pos, y_pos], lr=5.0)

    for _ in range(300):  # Learning loop
        optimizer.zero_grad()
        offsets = torch.tensor([x_pos, y_pos]).view(1, 1, 2)
        current_points = base_grid + offsets
        norm_x = 2 * current_points[..., 0] / (W - 1) - 1
        norm_y = 2 * current_points[..., 1] / (H - 1) - 1
        grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
        crop = F.grid_sample(ref_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze()

        # Cosine loss (negative similarity)
        flat_crop = torch.flatten(crop * feat_tensor)
        sim = flat_crop.sum() / (torch.norm(crop) * torch.norm(feat_tensor) + 1e-6)
        loss = -sim
        loss.backward()
        optimizer.step()

    # Final similarity
    final_sim = -loss.item()

    if final_sim > 0.85:  # Adjust threshold as needed
        print("Feature contained")
        # Get final position
        final_x = int(round(x_pos.item()))
        final_y = int(round(y_pos.item()))
        # Create high contrast feature (inverted for contrast)
        high_contrast_feat = 1 - feat
        # Overlay on original
        overlay = np.copy(ref)
        overlay[final_y:final_y + h, final_x:final_x + w] = high_contrast_feat
        plt.imshow(overlay)
        plt.show()
    else:
        print("Feature not contained")
