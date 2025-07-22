#!/usr/bin/env python3

import cv2
import numpy as np
import os
import argparse
import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ImageIO:
    """Unified image I/O and conversion module."""
    
    @staticmethod
    def to_grayscale(image: np.ndarray, method: str = 'opencv') -> np.ndarray:
        """
        Convert image to grayscale using various methods.
        
        Args:
            image: Input image
            method: Conversion method ('opencv', 'luminance', 'min', 'max', 'average')
        
        Returns:
            Grayscale image
        """
        if len(image.shape) == 2:
            return image
        elif len(image.shape) == 3:
            if method == 'opencv':
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif method == 'luminance':
                # ITU-R BT.709 luma coefficients
                b, g, r = cv2.split(image)
                return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.uint8)
            elif method == 'min':
                return np.min(image, axis=2).astype(np.uint8)
            elif method == 'max':
                return np.max(image, axis=2).astype(np.uint8)
            elif method == 'average':
                return np.mean(image, axis=2).astype(np.uint8)
            else:
                raise ValueError(f"Unknown grayscale method: {method}")
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")
    
    @staticmethod
    def to_hsv(image: np.ndarray) -> np.ndarray:
        """Convert image from BGR to HSV color space."""
        if len(image.shape) != 3:
            raise ValueError("Image must be in color (3 channels) for HSV conversion")
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    @staticmethod
    def to_lab(image: np.ndarray) -> np.ndarray:
        """Convert image from BGR to LAB color space."""
        if len(image.shape) != 3:
            raise ValueError("Image must be in color (3 channels) for LAB conversion")
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    @staticmethod
    def to_ycrcb(image: np.ndarray) -> np.ndarray:
        """Convert image from BGR to YCrCb color space."""
        if len(image.shape) != 3:
            raise ValueError("Image must be in color (3 channels) for YCrCb conversion")
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    @staticmethod
    def apply_colormap(image: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Apply a colormap to a grayscale image."""
        gray = ImageIO.to_grayscale(image)
        return cv2.applyColorMap(gray, colormap)
    
    @staticmethod
    def apply_canny_edge_detection(image: np.ndarray, 
                                   low_threshold: int = 50, 
                                   high_threshold: int = 150) -> np.ndarray:
        """Apply Canny edge detection to image."""
        gray = ImageIO.to_grayscale(image)
        return cv2.Canny(gray, low_threshold, high_threshold)
    
    @staticmethod
    def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to grayscale image."""
        gray = ImageIO.to_grayscale(image)
        return cv2.equalizeHist(gray)
    
    @staticmethod
    def read_image(file_path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
        """
        Read image from file.
        
        Args:
            file_path: Path to image file
            flags: OpenCV imread flags (default: cv2.IMREAD_COLOR)
                   Use cv2.IMREAD_UNCHANGED to preserve alpha channel
                   Use cv2.IMREAD_GRAYSCALE to read as grayscale
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        image = cv2.imread(file_path, flags)
        if image is None:
            raise ValueError(f"Failed to read image: {file_path}")
        
        return image
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """Save image to file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            return cv2.imwrite(output_path, image)
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
    
    @staticmethod
    def process_image(image: np.ndarray, 
                      conversion: str = 'grayscale',
                      **kwargs) -> np.ndarray:
        """
        Process image with specified conversion.
        
        Args:
            image: Input image
            conversion: Type of conversion ('grayscale', 'hsv', 'lab', 'canny', 'histogram_eq')
            **kwargs: Additional parameters for specific conversions
        
        Returns:
            Processed image
        """
        conversions = {
            'grayscale': lambda img: ImageIO.to_grayscale(img, kwargs.get('grayscale_method', 'opencv')),
            'hsv': ImageIO.to_hsv,
            'lab': ImageIO.to_lab,
            'ycrcb': ImageIO.to_ycrcb,
            'canny': lambda img: ImageIO.apply_canny_edge_detection(
                img, 
                kwargs.get('low_threshold', 50), 
                kwargs.get('high_threshold', 150)
            ),
            'histogram_eq': ImageIO.apply_histogram_equalization,
            'colormap': lambda img: ImageIO.apply_colormap(img, kwargs.get('colormap', cv2.COLORMAP_JET))
        }
        
        if conversion not in conversions:
            raise ValueError(f"Unknown conversion: {conversion}. "
                           f"Available: {list(conversions.keys())}")
        
        return conversions[conversion](image)


def grayscale_convert(input_path: str, output_path: str) -> None:
    """Convert image to grayscale (compatibility function)."""
    try:
        image = ImageIO.read_image(input_path)
        gray_image = ImageIO.to_grayscale(image)
        if ImageIO.save_image(gray_image, output_path):
            logger.info(f"Successfully converted {input_path} to grayscale")
        else:
            logger.error(f"Failed to save grayscale image to {output_path}")
    except Exception as e:
        logger.error(f"Error converting image: {e}")
        raise


def process_image(image: np.ndarray) -> np.ndarray:
    """Legacy function for backward compatibility."""
    return ImageIO.to_grayscale(image)


def main():
    """CLI interface for image processing."""
    parser = argparse.ArgumentParser(description='Unified Image I/O and Processing Tool')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('-c', '--conversion', default='grayscale',
                        choices=['grayscale', 'hsv', 'lab', 'ycrcb', 'canny', 'histogram_eq', 'colormap'],
                        help='Type of conversion to apply')
    parser.add_argument('--grayscale-method', default='opencv',
                        choices=['opencv', 'luminance', 'min', 'max', 'average'],
                        help='Method for grayscale conversion')
    parser.add_argument('--low-threshold', type=int, default=50,
                        help='Low threshold for Canny edge detection')
    parser.add_argument('--high-threshold', type=int, default=150,
                        help='High threshold for Canny edge detection')
    parser.add_argument('--colormap', type=str, default='jet',
                        choices=['jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'bone'],
                        help='Colormap to apply')
    parser.add_argument('--preserve-alpha', action='store_true',
                        help='Preserve alpha channel when reading images')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    try:
        # Read image
        read_flags = cv2.IMREAD_UNCHANGED if args.preserve_alpha else cv2.IMREAD_COLOR
        image = ImageIO.read_image(args.input, read_flags)
        
        # Map colormap names to OpenCV constants
        colormap_dict = {
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'cool': cv2.COLORMAP_COOL,
            'spring': cv2.COLORMAP_SPRING,
            'summer': cv2.COLORMAP_SUMMER,
            'autumn': cv2.COLORMAP_AUTUMN,
            'winter': cv2.COLORMAP_WINTER,
            'bone': cv2.COLORMAP_BONE
        }
        
        # Process image
        processed = ImageIO.process_image(
            image, 
            args.conversion,
            grayscale_method=args.grayscale_method,
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            colormap=colormap_dict.get(args.colormap, cv2.COLORMAP_JET)
        )
        
        # Save result
        if ImageIO.save_image(processed, args.output):
            print(f"Successfully processed {args.input} -> {args.output}")
        else:
            print(f"Failed to save output to {args.output}")
            return 1
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())