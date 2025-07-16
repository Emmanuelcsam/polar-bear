"""Test script for the unified image processor"""
import cv2
import numpy as np
from unified_image_processor import UnifiedImageProcessor, process_image_grayscale, process_image_blur_grayscale, process_image_canny_grayscale, process_image_circles


def create_test_image():
    """Create a test image with some shapes"""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    # Draw some circles
    cv2.circle(img, (75, 75), 30, (0, 0, 255), -1)
    cv2.circle(img, (225, 75), 40, (0, 255, 0), -1)
    cv2.circle(img, (150, 200), 50, (255, 0, 0), -1)
    # Draw some lines
    cv2.line(img, (0, 150), (300, 150), (0, 0, 0), 2)
    cv2.line(img, (150, 0), (150, 300), (0, 0, 0), 2)
    return img


def test_all_functions():
    """Test all processing functions"""
    print("Testing Unified Image Processor...")
    
    # Create test image
    test_img = create_test_image()
    cv2.imwrite('test_input.png', test_img)
    print("Created test image: test_input.png")
    
    processor = UnifiedImageProcessor()
    
    # Test 1: Grayscale conversion (mimics multiple original scripts)
    print("\n1. Testing grayscale conversion...")
    gray_result = process_image_grayscale(test_img)
    processor.save_as_image(gray_result, 'test_grayscale.png')
    processor.save_to_csv(gray_result, 'test_grayscale.csv')
    processor.save_to_numpy(gray_result, 'test_grayscale.npy')
    
    # Test 2: Gaussian blur + grayscale (mimics save_image_script (1).py)
    print("\n2. Testing Gaussian blur + grayscale...")
    blur_result = process_image_blur_grayscale(test_img, kernel_size=9, sigma=2)
    processor.save_as_image(blur_result, 'test_blur.png')
    
    # Test 3: Canny edge detection (mimics save_image_script.py)
    print("\n3. Testing Canny edge detection...")
    canny_result = process_image_canny_grayscale(test_img)
    processor.save_as_image(canny_result, 'test_canny.png')
    
    # Test 4: Circle detection (mimics save_result.py)
    print("\n4. Testing circle detection...")
    circle_result = process_image_circles(test_img, kernel_size=5, sigma=1)
    processor.save_as_image(circle_result, 'test_circles.png')
    
    # Test 5: Custom operation combinations
    print("\n5. Testing custom operation combinations...")
    
    # Just blur
    blur_only = processor.process_image(test_img, ['gaussian_blur'], kernel_size=11)
    processor.save_as_image(blur_only, 'test_blur_only.png')
    
    # Grayscale then edge detection
    edges = processor.process_image(test_img, ['grayscale', 'canny_edge'])
    processor.save_as_image(edges, 'test_grayscale_edges.png')
    
    # Test CSV with coordinates
    print("\n6. Testing CSV export with coordinates...")
    processor.save_to_csv(gray_result, 'test_with_coordinates.csv', include_coordinates=True)
    
    print("\nAll tests completed! Check the generated files.")


if __name__ == "__main__":
    test_all_functions()