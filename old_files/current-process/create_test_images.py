import numpy as np
import cv2
import os

# Create a simple test image that simulates a fiber optic cross-section
def create_test_fiber_image(width=512, height=512):
    """Create a synthetic fiber optic cross-section for testing"""
    
    # Create blank image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add background noise
    noise = np.random.normal(20, 5, (height, width, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Create fiber core (bright circle in center)
    center_x, center_y = width // 2, height // 2
    core_radius = min(width, height) // 8
    
    # Core
    cv2.circle(img, (center_x, center_y), core_radius, (200, 200, 200), -1)
    
    # Cladding (dimmer ring around core)
    cladding_radius = core_radius * 3
    cv2.circle(img, (center_x, center_y), cladding_radius, (100, 100, 100), 15)
    
    # Add some simulated defects
    # Scratch defect
    cv2.line(img, (center_x - 50, center_y - 30), (center_x + 50, center_y + 30), (0, 0, 255), 2)
    
    # Pit defects (small dark spots)
    cv2.circle(img, (center_x - 20, center_y + 40), 3, (0, 0, 0), -1)
    cv2.circle(img, (center_x + 30, center_y - 50), 4, (0, 0, 0), -1)
    
    # Contamination (irregular bright spots)
    cv2.circle(img, (center_x + 60, center_y + 20), 8, (255, 255, 255), -1)
    
    return img

def main():
    # Create test images directory
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create multiple test images with different characteristics
    test_images = [
        ("fiber_good.png", create_test_fiber_image()),
        ("fiber_defective.png", create_test_fiber_image()),
    ]
    
    for filename, img in test_images:
        filepath = os.path.join(test_dir, filename)
        cv2.imwrite(filepath, img)
        print(f"Created test image: {filepath}")

if __name__ == "__main__":
    main()
