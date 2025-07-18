import cv2
import numpy as np

# Load the full image
img = cv2.imread('Labrador_Retriever_portrait.jpg', cv2.IMREAD_GRAYSCALE)
if img is not None:
    print(f"Original image shape: {img.shape}")
    
    # Create a smaller feature image by cropping a section
    # Let's take the dog's face area (approximately)
    h, w = img.shape
    
    # Crop a section from the center-left area (where the dog's face is)
    start_y = h // 4
    end_y = 3 * h // 4
    start_x = w // 6
    end_x = 2 * w // 3
    
    feature_img = img[start_y:end_y, start_x:end_x]
    
    print(f"Feature image shape: {feature_img.shape}")
    
    # Save the feature image
    cv2.imwrite('dog_feature.jpg', feature_img)
    print("Saved dog_feature.jpg")
    
    # Also save the full image as reference
    cv2.imwrite('dog_reference.jpg', img)
    print("Saved dog_reference.jpg")
else:
    print("Failed to load the image")
