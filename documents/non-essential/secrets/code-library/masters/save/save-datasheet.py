"""Unified Image Processing Script - Combines all processing operations from individual scripts"""
import cv2
import numpy as np
from typing import Optional, Tuple, Union
import pandas as pd


class UnifiedImageProcessor:
    """Unified processor combining all image processing operations"""
    
    def __init__(self):
        self.operations = {
            'grayscale': self.convert_to_grayscale,
            'gaussian_blur': self.apply_gaussian_blur,
            'canny_edge': self.apply_canny_edge,
            'circle_detection': self.detect_circles,
        }
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
        """Apply Gaussian blur to the image"""
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def apply_canny_edge(self, image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        """Apply Canny edge detection"""
        gray = self.convert_to_grayscale(image)
        return cv2.Canny(gray, low_threshold, high_threshold)
    
    def detect_circles(self, image: np.ndarray, min_dist: int = 50, param1: int = 50, param2: int = 30) -> np.ndarray:
        """Detect circles using Hough Circle Transform"""
        # Ensure we have a color image for drawing
        if len(image.shape) == 2:
            display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display = image.copy()
        
        # Convert to grayscale for circle detection
        gray = self.convert_to_grayscale(display)
        
        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, min_dist, 
                                   param1=param1, param2=param2)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw circle outline
                cv2.circle(display, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw center point
                cv2.circle(display, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        return display
    
    def process_image(self, image: np.ndarray, operations: list = None, **kwargs) -> np.ndarray:
        """
        Process image with specified operations
        
        Args:
            image: Input image
            operations: List of operations to apply. If None, applies grayscale only
            **kwargs: Additional parameters for specific operations
        
        Returns:
            Processed image
        """
        try:
            result = image.copy()
            
            if operations is None:
                operations = ['grayscale']
            
            for op in operations:
                if op in self.operations:
                    if op == 'gaussian_blur':
                        result = self.operations[op](result, 
                                                    kwargs.get('kernel_size', 5),
                                                    kwargs.get('sigma', 0))
                    elif op == 'canny_edge':
                        result = self.operations[op](result,
                                                    kwargs.get('low_threshold', 50),
                                                    kwargs.get('high_threshold', 150))
                    elif op == 'circle_detection':
                        result = self.operations[op](result,
                                                    kwargs.get('min_dist', 50),
                                                    kwargs.get('param1', 50),
                                                    kwargs.get('param2', 30))
                    else:
                        result = self.operations[op](result)
                else:
                    print(f"Unknown operation: {op}")
            
            return result
            
        except Exception as e:
            print(f"Error in processing: {e}")
            return image
    
    def save_to_csv(self, data: np.ndarray, filename: str, include_coordinates: bool = False):
        """Save data to CSV file"""
        if len(data.shape) == 2:
            if include_coordinates:
                # Create DataFrame with coordinates
                rows, cols = data.shape
                df_data = []
                for i in range(rows):
                    for j in range(cols):
                        df_data.append({'x': j, 'y': i, 'value': data[i, j]})
                df = pd.DataFrame(df_data)
            else:
                # Simple 2D array to CSV
                df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("Warning: Can only save 2D arrays to CSV")
    
    def save_to_numpy(self, data: np.ndarray, filename: str):
        """Save data to numpy file"""
        np.save(filename, data)
        print(f"Data saved to {filename}")
    
    def save_as_image(self, data: np.ndarray, filename: str):
        """Save data as image file"""
        cv2.imwrite(filename, data)
        print(f"Image saved to {filename}")


# Convenience functions for backward compatibility
def process_image_grayscale(image: np.ndarray) -> np.ndarray:
    """Process image with grayscale conversion (compatible with original scripts)"""
    processor = UnifiedImageProcessor()
    return processor.process_image(image, ['grayscale'])


def process_image_blur_grayscale(image: np.ndarray, kernel_size: float = 5, sigma: float = 0) -> np.ndarray:
    """Process image with Gaussian blur and grayscale (compatible with save_image_script (1).py)"""
    processor = UnifiedImageProcessor()
    return processor.process_image(image, ['grayscale', 'gaussian_blur'], 
                                   kernel_size=kernel_size, sigma=sigma)


def process_image_canny_grayscale(image: np.ndarray) -> np.ndarray:
    """Process image with Canny edge detection and grayscale (compatible with save_image_script.py)"""
    processor = UnifiedImageProcessor()
    return processor.process_image(image, ['canny_edge'])


def process_image_circles(image: np.ndarray, kernel_size: float = 5, sigma: float = 0) -> np.ndarray:
    """Process image with circle detection (compatible with save_result.py)"""
    processor = UnifiedImageProcessor()
    return processor.process_image(image, ['grayscale', 'gaussian_blur', 'circle_detection'],
                                   kernel_size=kernel_size, sigma=sigma)


# Main execution example
if __name__ == "__main__":
    # Example usage
    print("Unified Image Processor")
    print("Available operations: grayscale, gaussian_blur, canny_edge, circle_detection")
    print("\nExample usage:")
    print("  processor = UnifiedImageProcessor()")
    print("  result = processor.process_image(image, ['grayscale', 'gaussian_blur'])")
    print("  processor.save_to_csv(result, 'output.csv')")
    print("  processor.save_to_numpy(result, 'output.npy')")
    print("  processor.save_as_image(result, 'output.png')")