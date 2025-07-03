# tests/test_zone_segmentation.py
import unittest
import numpy as np
import cv2
import yaml
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.zone_segmentation import segment_zones

class TestZoneSegmentation(unittest.TestCase):

    def setUp(self):
        """Set up a dummy image and config for testing."""
        self.image = np.zeros((512, 512, 3), dtype=np.uint8)
        # Draw a white circle to simulate a fiber
        cv2.circle(self.image, (256, 256), 200, (255, 255, 255), -1)
        
        self.config = {
            'blur_kernel_size': [5, 5],
            'canny_thresholds': [50, 150],
            'zone_definitions': {
                'core': 10,
                'cladding': 125,
            }
        }

    def test_segment_zones(self):
        """Test the segment_zones function."""
        results = segment_zones(self.image, self.config)
        
        self.assertIn("masks", results)
        self.assertIn("metrics", results)
        
        self.assertIn("core", results["masks"])
        self.assertIn("cladding", results["masks"])
        
        self.assertIn("center", results["metrics"])
        self.assertIn("cladding_radius_px", results["metrics"])
        self.assertIn("core_radius_px", results["metrics"])

    def test_segment_zones_no_circles(self):
        """Test segment_zones with an image that has no circles."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        results = segment_zones(image, self.config)
        self.assertEqual(results["masks"], {})
        self.assertEqual(results["metrics"], {})

if __name__ == '__main__':
    # Need to import cv2 for the test setup
    import cv2
    unittest.main()
