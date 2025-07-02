# tests/test_defect_detection.py
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.defect_detection import detect_defects

class TestDefectDetection(unittest.TestCase):

    def setUp(self):
        """Set up a dummy image and config for testing."""
        self.image = np.zeros((512, 512, 3), dtype=np.uint8)
        # Draw a white circle to simulate a fiber
        cv2.circle(self.image, (256, 256), 200, (255, 255, 255), -1)
        # Add a small black rectangle to simulate a defect
        cv2.rectangle(self.image, (200, 200), (220, 220), (0, 0, 0), -1)
        
        self.masks = {
            "core": np.zeros((512, 512), dtype=np.uint8),
            "cladding": np.zeros((512, 512), dtype=np.uint8),
            "ferrule": np.zeros((512, 512), dtype=np.uint8),
        }
        cv2.circle(self.masks["cladding"], (256, 256), 200, 255, -1)

        self.config = {
            'blur_kernel_size': [5, 5],
            'canny_thresholds': [50, 150],
            'min_defect_area_px': 10,
            'scratch_aspect_ratio_threshold': 3.0,
            'anomaly_threshold_sigma': 1.0
        }

    def test_detect_defects(self):
        """Test the detect_defects function."""
        defect_mask, defect_regions = detect_defects(self.image, self.masks, self.config)
        
        self.assertIsNotNone(defect_mask)
        self.assertIsInstance(defect_regions, list)
        self.assertGreater(len(defect_regions), 0)

if __name__ == '__main__':
    unittest.main()
