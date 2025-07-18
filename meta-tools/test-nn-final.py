import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import sys

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

def main():
    # Ask for image paths
    ref_path = input("Enter the path of the reference (main image): ")
    feat_path = input("Enter the path of the feature (part contained): ")

    print(f"Loading reference image: {ref_path}")
    print(f"Loading feature image: {feat_path}")

    # Load images
    try:
        ref_gray = load_image(ref_path)
        feat_gray = load_image(feat_path)
        
        print(f"Reference: shape={ref_gray.shape}, min={ref_gray.min():.3f}, max={ref_gray.max():.3f}")
        print(f"Feature: shape={feat_gray.shape}, min={feat_gray.min():.3f}, max={feat_gray.max():.3f}")
        
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # Get dimensions
    H, W = ref_gray.shape
    h, w = feat_gray.shape

    print(f"Reference: {H}x{W}, Feature: {h}x{w}")

    if h > H or w > W:
        print("Feature not contained - feature is larger than reference")
        return

    # Check if feature has variation
    if feat_gray.std() < 1e-6:
        print("Warning: Feature image appears to be constant or nearly constant")
        print("This may cause issues with similarity computation")

    # Prepare tensors
    ref_tensor = torch.from_numpy(ref_gray).float().unsqueeze(0).unsqueeze(0)
    feat_tensor = torch.from_numpy(feat_gray).float()

    # Base grid for feature
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    base_grid = torch.from_numpy(np.stack((xx, yy), axis=-1)).float()

    # Try multiple random starting positions to avoid local minima
    best_mse = float('inf')
    best_x, best_y = 0, 0
    
    for trial in range(5):  # Try 5 different starting positions
        print(f"\nTrial {trial + 1}/5:")
        
        # Initial random position for this trial
        x_pos = torch.tensor(float(np.random.uniform(0, W - w)), requires_grad=True)
        y_pos = torch.tensor(float(np.random.uniform(0, H - h)), requires_grad=True)

        print(f"Starting position: ({x_pos.item():.2f}, {y_pos.item():.2f})")

        optimizer = optim.Adam([x_pos, y_pos], lr=3.0)

        for iteration in range(150):
            optimizer.zero_grad()
            
            # Create grid for sampling
            offsets = torch.stack([x_pos, y_pos]).view(1, 1, 2)
            current_points = base_grid + offsets
            
            # Normalize coordinates to [-1, 1]
            norm_x = 2 * current_points[..., 0] / (W - 1) - 1
            norm_y = 2 * current_points[..., 1] / (H - 1) - 1
            grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
            
            # Sample from reference image
            crop = F.grid_sample(ref_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze()
            
            # Compute mean squared error (MSE) loss
            mse = torch.mean((crop - feat_tensor)**2)
            loss = mse
            
            try:
                # Backward pass
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"Error in gradient computation at iteration {iteration}: {e}")
                break
            
            # Clamp positions to valid ranges
            with torch.no_grad():
                x_pos.clamp_(0, W - w)
                y_pos.clamp_(0, H - h)
        
        final_mse = mse.item()
        print(f"Final MSE: {final_mse:.6f}, Final position: ({x_pos.item():.1f}, {y_pos.item():.1f})")
        
        if final_mse < best_mse:
            best_mse = final_mse
            best_x = x_pos.item()
            best_y = y_pos.item()

    print(f"\nBest result across all trials:")
    print(f"Best MSE: {best_mse:.6f}")
    print(f"Best position: ({best_x:.1f}, {best_y:.1f})")

    # Final evaluation
    final_sim = np.exp(-best_mse * 10)  # Convert MSE to similarity score
    print(f"Final similarity score: {final_sim:.4f}")

    # Lower threshold for MSE-based detection
    if best_mse < 0.05 or final_sim > 0.6:
        print("Feature contained")
        final_x = int(round(best_x))
        final_y = int(round(best_y))
        print(f"Found at position: ({final_x}, {final_y})")
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(ref_gray, cmap='gray')
        plt.title('Reference Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(feat_gray, cmap='gray')
        plt.title('Feature to Find')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        overlay = np.copy(ref_gray)
        # Highlight found region with a rectangle
        overlay[final_y:final_y+2, final_x:final_x+w] = 1.0  # Top border
        overlay[final_y+h-2:final_y+h, final_x:final_x+w] = 1.0  # Bottom border
        overlay[final_y:final_y+h, final_x:final_x+2] = 1.0  # Left border
        overlay[final_y:final_y+h, final_x+w-2:final_x+w] = 1.0  # Right border
        
        plt.imshow(overlay, cmap='gray')
        plt.title(f'Found Location (MSE: {best_mse:.4f})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Feature not contained")
        print(f"Best MSE: {best_mse:.6f} (threshold: 0.05)")

if __name__ == "__main__":
    main()
