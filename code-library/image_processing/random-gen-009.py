from PIL import Image
import numpy as np

def create_test_image():
    """Create a test image with known patterns for testing"""
    
    # Create 100x100 image
    size = (100, 100)
    pixels = np.zeros((size[1], size[0]), dtype=np.uint8)
    
    # Add some patterns
    # 1. Gradient from top to bottom
    for y in range(size[1]):
        pixels[y, :20] = int(255 * y / size[1])
    
    # 2. Repeated pattern
    for x in range(20, 40):
        pixels[:, x] = 128
    
    # 3. Checkerboard pattern
    for y in range(0, size[1], 10):
        for x in range(40, 60, 10):
            pixels[y:y+5, x:x+5] = 200
    
    # 4. Random noise area
    noise = np.random.randint(0, 256, (size[1], 20))
    pixels[:, 60:80] = noise
    
    # 5. Some specific values for correlation testing
    pixels[10:20, 80:90] = 42  # A specific value
    pixels[30:35, 85:95] = 137  # Another specific value
    
    # Create and save image
    img = Image.fromarray(pixels, mode='L')
    img.save('test_pattern.jpg')
    
    print("[TEST_IMAGE] Created test_pattern.jpg")
    print("[TEST_IMAGE] Patterns included:")
    print("  - Gradient (columns 0-20)")
    print("  - Uniform area with value 128 (columns 20-40)")
    print("  - Checkerboard (columns 40-60)")
    print("  - Random noise (columns 60-80)")
    print("  - Specific test values 42 and 137 (columns 80-100)")
    
    return 'test_pattern.jpg'

if __name__ == "__main__":
    create_test_image()