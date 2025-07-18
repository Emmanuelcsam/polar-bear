import torch
import torch.nn.functional as F
import numpy as np

print("Testing basic gradient computation...")

# Create simple test data
ref = torch.randn(1, 1, 100, 100)  # Reference image
feat = torch.randn(50, 50)  # Feature to find

# Position parameters that need gradients
x_pos = torch.tensor(25.0, requires_grad=True)
y_pos = torch.tensor(25.0, requires_grad=True)

print(f"x_pos requires_grad: {x_pos.requires_grad}")
print(f"y_pos requires_grad: {y_pos.requires_grad}")

# Create grid
h, w = feat.shape
yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
base_grid = torch.from_numpy(np.stack((xx, yy), axis=-1)).float()

print(f"base_grid shape: {base_grid.shape}")

# Apply offset
offsets = torch.stack([x_pos, y_pos]).view(1, 1, 2)
current_points = base_grid + offsets

print(f"current_points requires_grad: {current_points.requires_grad}")

# Normalize to [-1, 1]
H, W = 100, 100
norm_x = 2 * current_points[..., 0] / (W - 1) - 1
norm_y = 2 * current_points[..., 1] / (H - 1) - 1
grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)

print(f"grid requires_grad: {grid.requires_grad}")

# Sample
crop = F.grid_sample(ref, grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze()

print(f"crop requires_grad: {crop.requires_grad}")
print(f"crop shape: {crop.shape}")

# Simple loss
loss = -torch.mean(crop * feat.detach())

print(f"loss requires_grad: {loss.requires_grad}")

try:
    loss.backward()
    print("Gradient computation successful!")
    print(f"x_pos.grad: {x_pos.grad}")
    print(f"y_pos.grad: {y_pos.grad}")
except Exception as e:
    print(f"Error: {e}")
