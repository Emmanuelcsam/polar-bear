# tests/test_feature_extraction.py
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_extraction import extract_features

class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        """Set up dummy data for testing."""
        self.image = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.rectangle(self.image, (200, 200), (220, 220), (128, 128, 128), -1)
        
        contour = np.array([[[200, 200]], [[200, 220]], [[220, 220]], [[220, 200]]])
        self.defect_regions = [{"contour": contour, "area": 400, "centroid": (210, 210)}]
        
        self.zone_masks = {
            "core": np.zeros((512, 512), dtype=np.uint8),
            "cladding": np.ones((512, 512), dtype=np.uint8) * 255,
            "ferrule": np.zeros((512, 512), dtype=np.uint8),
        }
        self.metrics = {}

    def test_extract_features(self):
        """Test the extract_features function."""
        features_list = extract_features(self.image, self.defect_regions, self.zone_masks, self.metrics)
        
        self.assertIsInstance(features_list, list)
        self.assertEqual(len(features_list), 1)
        
        features = features_list[0]
        self.assertIn("area_px", features)
        self.assertIn("aspect_ratio", features)
        self.assertIn("mean_intensity", features)
        self.assertIn("contrast", features)

if __name__ == '__main__':
    unittest.main()
