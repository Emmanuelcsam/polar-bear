"""
DO2MR (Difference-of-Min-Max Ranking) and LEI (Linear Enhancement Inspector) detectors
Based on Mei et al., "Automated Inspection..." Sensors 18 (2018)
"""

import cv2
import numpy as np
from typing import Tuple, Dict

class DO2MR_Detector:
    """
    Difference-of-Min-Max Ranking filter for region defects (dust, pits, contamination)
    """
    def __init__(self, window_size: int = 5, gamma: float = 1.5):
        self.window_size = window_size
        self.gamma = gamma
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply DO2MR detection
        
        Args:
            image: Grayscale image
            
        Returns:
            Binary mask of detected defects
        """
        # Gaussian pre-blur (Eq. 1-2 from paper)
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # Local extrema surfaces (Eq. 4)
        kernel = np.ones((self.window_size, self.window_size), np.uint8)
        Imax = cv2.dilate(blurred, kernel)
        Imin = cv2.erode(blurred, kernel)
        
        # Residual map
        Ir = Imax - Imin
        
        # Statistical threshold (Eq. 5-6)
        mu = np.mean(Ir)
        sigma = np.std(Ir)
        threshold = mu + self.gamma * sigma
        
        # Binary mask
        mask = (Ir > threshold).astype(np.uint8) * 255
        
        # Morphological opening to remove small noise
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        return mask

class LEI_Detector:
    """
    Linear Enhancement Inspector for scratch detection
    """
    def __init__(self, angle_step: int = 15, line_length: int = 21):
        self.angle_step = angle_step
        self.line_length = line_length
        self.angles = list(range(0, 180, angle_step))
    
    def _create_line_kernel(self, angle: float, length: int) -> np.ndarray:
        """Create a line-shaped kernel at given angle"""
        angle_rad = np.deg2rad(angle)
        center = length // 2
        kernel = np.zeros((length, length))
        
        # Draw line through center
        for i in range(length):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1
                
        # Normalize
        kernel = kernel / (np.sum(kernel) + 1e-8)
        return kernel
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Apply LEI detection for scratches
        
        Args:
            image: Grayscale image
            
        Returns:
            Binary mask of detected scratches
        """
        # Histogram equalization (Eq. 7-9)
        equalized = cv2.equalizeHist(image)
        
        # Initialize scratch mask
        scratch_mask = np.zeros_like(image)
        
        # For each angle
        for angle in self.angles:
            # Create line kernel
            kernel = self._create_line_kernel(angle, self.line_length)
            
            # Create perpendicular kernel for background
            kernel_bg = self._create_line_kernel(angle + 90, self.line_length // 2)
            
            # Apply filters (Eq. 10: scratch strength)
            f_red = cv2.filter2D(equalized, -1, kernel)
            f_gray = cv2.filter2D(equalized, -1, kernel_bg)
            
            # Scratch strength
            s_theta = 2 * f_red - f_gray
            
            # Threshold per angle
            mu = np.mean(s_theta)
            sigma = np.std(s_theta)
            threshold = mu + 2 * sigma
            
            # Update mask
            angle_mask = (s_theta > threshold).astype(np.uint8)
            scratch_mask = cv2.bitwise_or(scratch_mask, angle_mask)
        
        # Clean up result
        scratch_mask = scratch_mask.astype(np.uint8) * 255
        
        # Connect broken segments
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, kernel_close)
        
        return scratch_mask

class UnifiedDefectDetector:
    """
    Combines DO2MR and LEI for comprehensive defect detection
    """
    def __init__(self):
        self.do2mr = DO2MR_Detector()
        self.lei = LEI_Detector()
    
    def detect_all(self, image: np.ndarray, zones: Dict[str, np.ndarray] = None) -> Dict:
        """
        Detect all types of defects
        
        Args:
            image: Input image (BGR or grayscale)
            zones: Optional zone masks (core, cladding, ferrule)
            
        Returns:
            Dictionary with defect masks and metadata
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply both detectors
        region_defects = self.do2mr.detect(gray)
        scratch_defects = self.lei.detect(gray)
        
        # Combine masks
        all_defects = cv2.bitwise_or(region_defects, scratch_defects)
        
        # Find contours and characterize defects
        contours, _ = cv2.findContours(all_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defect_list = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 10:  # Skip tiny noise
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Determine type based on aspect ratio
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            defect_type = "SCRATCH" if aspect_ratio >= 3.0 else "PIT/DIG"
            
            # Determine zone if provided
            zone = "UNKNOWN"
            if zones:
                cx, cy = x + w//2, y + h//2
                if zones.get('core') is not None and zones['core'][cy, cx] > 0:
                    zone = "CORE"
                elif zones.get('cladding') is not None and zones['cladding'][cy, cx] > 0:
                    zone = "CLADDING"
                elif zones.get('ferrule') is not None and zones['ferrule'][cy, cx] > 0:
                    zone = "FERRULE"
            
            defect_list.append({
                'id': f"DEFECT_{i:03d}",
                'type': defect_type,
                'bbox': [int(x), int(y), int(w), int(h)],
                'area': int(area),
                'aspect_ratio': float(aspect_ratio),
                'zone': zone,
                'centroid': [int(x + w/2), int(y + h/2)]
            })
        
        return {
            'region_mask': region_defects,
            'scratch_mask': scratch_defects,
            'combined_mask': all_defects,
            'defects': defect_list,
            'count': len(defect_list)
        }

# Example usage
if __name__ == "__main__":
    # Test on a sample image
    detector = UnifiedDefectDetector()
    
    # Load test image (replace with actual path)
    # image = cv2.imread("test_fiber.png")
    # results = detector.detect_all(image)
    # print(f"Detected {results['count']} defects")