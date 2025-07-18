import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

def load_image(path):
    """Load image and convert to grayscale numpy array"""
    try:
        # Try loading with PIL first for better format support
        img = Image.open(path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.dot(img_array, [0.299, 0.587, 0.114])
        else:
            gray = img_array
            
        return gray
    except:
        # Fallback to matplotlib
        img = plt.imread(path)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1:
            img = img / 255.0
            
        if len(img.shape) == 3 and img.shape[2] >= 3:
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = img if len(img.shape) == 2 else img[..., 0]
        
        return gray

def grid_search_best_position(ref_gray, feat_gray, step=10):
    """Do a coarse grid search to find the best starting position"""
    H, W = ref_gray.shape
    h, w = feat_gray.shape
    
    best_mse = float('inf')
    best_pos = (0, 0)
    
    for y in range(0, H - h + 1, step):
        for x in range(0, W - w + 1, step):
            crop = ref_gray[y:y+h, x:x+w]
            mse = np.mean((crop - feat_gray)**2)
            if mse < best_mse:
                best_mse = mse
                best_pos = (x, y)
    
    return best_pos, best_mse

# Ask for image paths
ref_path = input("Enter the path of the reference (main image): ")
feat_path = input("Enter the path of the feature (part contained): ")

# Load images
ref_gray = load_image(ref_path)
feat_gray = load_image(feat_path)

# Get dimensions
H, W = ref_gray.shape
h, w = feat_gray.shape

if h > H or w > W:
    print("Feature not contained")
else:
    # First do a grid search to find the best region
    best_grid_pos, best_grid_mse = grid_search_best_position(ref_gray, feat_gray)
    
    # Prepare tensors
    ref_tensor = torch.from_numpy(ref_gray).float().unsqueeze(0).unsqueeze(0)
    feat_tensor = torch.from_numpy(feat_gray).float()

    # Base grid for feature
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    base_grid = torch.from_numpy(np.stack((xx, yy), axis=-1)).float()

    # Start from the best grid position
    x_pos = torch.tensor(float(best_grid_pos[0]), requires_grad=True)
    y_pos = torch.tensor(float(best_grid_pos[1]), requires_grad=True)

    optimizer = optim.Adam([x_pos, y_pos], lr=1.0)

    for _ in range(100):  # Learning loop
        optimizer.zero_grad()
        offsets = torch.tensor([x_pos, y_pos]).view(1, 1, 2)
        current_points = base_grid + offsets
        norm_x = 2 * current_points[..., 0] / (W - 1) - 1
        norm_y = 2 * current_points[..., 1] / (H - 1) - 1
        grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
        crop = F.grid_sample(ref_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze()

        # MSE loss
        mse = torch.mean((crop - feat_tensor)**2)
        loss = mse
        
        try:
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            print(f"Gradient error: {e}")
            break
        
        # Clamp positions
        with torch.no_grad():
            x_pos.clamp_(0, W - w)
            y_pos.clamp_(0, H - h)

    # Final similarity
    final_mse = loss.item()

    if final_mse < 0.1:  # Adjust threshold as needed
        print("Feature contained")
        # Get final position
        final_x = int(round(x_pos.item()))
        final_y = int(round(y_pos.item()))
        # Create high contrast feature (inverted for contrast)
        high_contrast_feat = 1 - feat_gray
        # Overlay on original
        overlay = np.copy(ref_gray)
        overlay[final_y:final_y + h, final_x:final_x + w] = high_contrast_feat
        plt.imshow(overlay, cmap='gray')
        plt.show()
    else:
        print("Feature not contained")
