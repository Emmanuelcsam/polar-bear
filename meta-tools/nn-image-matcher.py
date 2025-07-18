import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from PIL import Image

class CorrelationMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)  # 3 correlations to 1 output
        self.noise_scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, ref_img, feat_img, position):
        # Generate adaptive random numbers (less random as we learn)
        rand_ref = torch.randn_like(ref_img) * torch.abs(self.noise_scale)
        rand_feat = torch.randn_like(feat_img) * torch.abs(self.noise_scale)
        
        # Calculate correlations
        corr1 = torch.mean(rand_ref * ref_img)  # random vs reference
        corr2 = torch.mean(rand_feat * feat_img)  # random vs feature
        corr3 = torch.mean(ref_img * feat_img)  # reference vs feature
        
        # Learn from correlations
        correlations = torch.stack([corr1, corr2, corr3])
        match_score = torch.sigmoid(self.fc(correlations))
        
        # Constrain random numbers based on correlations
        self.noise_scale.data *= 0.99  # Reduce randomness
        
        return match_score

def load_image_safe(file_path):
    """Load an image with error handling and multiple fallback methods"""
    # Normalize path separators for Windows
    file_path = os.path.normpath(file_path)
    
    print(f"Attempting to load: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes")
    
    if file_size == 0:
        print(f"Error: File is empty - {file_path}")
        return None
    
    # Method 1: Try OpenCV (original method)
    print("Method 1: Trying OpenCV...")
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        print(f"Successfully loaded with OpenCV, shape: {img.shape}")
        return img
    
    # Method 2: Try OpenCV with color mode first
    print("Method 2: Trying OpenCV color mode...")
    img_color = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img_color is not None:
        print("Successfully loaded as color image with OpenCV, converting to grayscale...")
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        print(f"Converted to grayscale, shape: {img.shape}")
        return img
    
    # Method 3: Try PIL/Pillow
    print("Method 3: Trying PIL/Pillow...")
    try:
        with Image.open(file_path) as pil_img:
            # Convert to grayscale if needed
            if pil_img.mode != 'L':
                pil_img = pil_img.convert('L')
            
            # Convert PIL image to numpy array
            img = np.array(pil_img)
            print(f"Successfully loaded with PIL, shape: {img.shape}")
            return img
    except Exception as e:
        print(f"PIL failed: {e}")
    
    # Method 4: Try PIL with different modes
    print("Method 4: Trying PIL with RGB conversion...")
    try:
        with Image.open(file_path) as pil_img:
            # First convert to RGB, then to grayscale
            if pil_img.mode in ['RGBA', 'P', 'CMYK']:
                pil_img = pil_img.convert('RGB')
            pil_img = pil_img.convert('L')
            
            # Convert PIL image to numpy array
            img = np.array(pil_img)
            print(f"Successfully loaded with PIL (RGB conversion), shape: {img.shape}")
            return img
    except Exception as e:
        print(f"PIL with RGB conversion failed: {e}")
    
    print("All methods failed to load the image.")
    print("The image file may be:")
    print("  - Corrupted")
    print("  - In an unsupported format")
    print("  - Protected or locked")
    print("  - Not a valid image file")
    
    return None

# Print OpenCV build info for debugging
print("OpenCV version:", cv2.__version__)
print("Supported image formats:", ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'])
print("=" * 60)

# Get image paths from user
ref_path = input("Enter path of reference (main) image: ").strip().strip('"').strip("'")
feat_path = input("Enter path of feature (part contained) image: ").strip().strip('"').strip("'")

print("=" * 60)

# Load images with error handling
print("Loading reference image...")
ref_img = load_image_safe(ref_path)
if ref_img is None:
    print("\n" + "="*60)
    print("TROUBLESHOOTING TIPS:")
    print("- Verify the image file is not corrupted")
    print("- Try opening the image in an image viewer")
    print("- Try converting the image to a different format")
    print("- Make sure the file path is correct and accessible")
    print("="*60)
    exit(1)

print("\nLoading feature image...")
feat_img = load_image_safe(feat_path)
if feat_img is None:
    print("\n" + "="*60)
    print("TROUBLESHOOTING TIPS:")
    print("- Verify the image file is not corrupted")
    print("- Try opening the image in an image viewer")
    print("- Try converting the image to a different format")
    print("- Make sure the file path is correct and accessible")
    print("="*60)
    exit(1)

# Check if feature image is smaller than reference image
if feat_img.shape[0] > ref_img.shape[0] or feat_img.shape[1] > ref_img.shape[1]:
    print("Error: Feature image must be smaller than reference image")
    exit(1)

# Convert to tensors and normalize
ref_tensor = torch.tensor(ref_img / 255.0, dtype=torch.float32)
feat_tensor = torch.tensor(feat_img / 255.0, dtype=torch.float32)

# Initialize model and optimizer
model = CorrelationMatcher()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Sliding window search with learning
h, w = feat_img.shape
best_score = 0
best_pos = (0, 0)

for y in range(ref_img.shape[0] - h + 1):
    for x in range(ref_img.shape[1] - w + 1):
        # Extract window from reference
        window = ref_tensor[y:y+h, x:x+w]
        
        # Forward pass with learning
        optimizer.zero_grad()
        score = model(window, feat_tensor, (x, y))
        
        # Learn from correlation patterns
        loss = -score if torch.mean(window * feat_tensor) > 0.7 else score
        loss.backward()
        optimizer.step()
        
        # Track best match
        if score.item() > best_score:
            best_score = score.item()
            best_pos = (x, y)

# Display results
print(f"\nNeural Network Results:")
print(f"Best NN score: {best_score:.4f}")
print(f"Best NN position: {best_pos}")

# Fallback to traditional template matching for comparison
print(f"\nFallback: Traditional Template Matching...")
result_tm = cv2.matchTemplate(ref_img, feat_img, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_tm)
print(f"Template matching score: {max_val:.4f}")
print(f"Template matching position: {max_loc}")

# Use the better result
use_nn = best_score > 0.5 and best_score > max_val
final_score = best_score if use_nn else max_val
final_pos = best_pos if use_nn else max_loc
method_used = "Neural Network" if use_nn else "Template Matching"

print(f"\nFinal Results (using {method_used}):")
print(f"Score: {final_score:.4f}")
print(f"Position: {final_pos}")

if final_score > 0.5:  # Lower threshold for better detection
    print("\n✓ Feature contained!")
    
    # Create overlay
    result = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    overlay = result.copy()
    
    # Draw high contrast rectangle
    x, y = final_pos
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Add score text
    cv2.putText(overlay, f"{method_used}: {final_score:.3f}", 
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add transparent overlay
    cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
    
    # Save and show result
    output_path = "detection_result.png"
    cv2.imwrite(output_path, result)
    print(f"Result saved to: {output_path}")
    
    try:
        print("Displaying result window (press any key to close)...")
        cv2.imshow("Feature Found - Overlay Map", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Could not display window (headless environment): {e}")
        print("Check the saved image file instead.")
else:
    print("\n✗ Feature not contained")
    print(f"Score {final_score:.4f} is below threshold 0.5")