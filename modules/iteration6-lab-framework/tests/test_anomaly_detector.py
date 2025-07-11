import unittest
import sys
import numpy as np
import cv2
import tempfile
import os
import shutil
sys.path.append('..')

class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        os.makedirs('data', exist_ok=True)
        
        # Create test image with anomaly
        self.test_img = np.ones((32, 32), dtype=np.uint8) * 128
        self.test_img[10:15, 10:15] = 255  # Bright square anomaly
        
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir='.')
        cv2.imwrite(self.temp_file.name, self.test_img)
        
    def tearDown(self):
        os.unlink(self.temp_file.name)
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_detect(self):
        from modules.anomaly_detector import detect
        from core.datastore import get
        
        detect(self.temp_file.name)
        
        # Check if anomalies were stored
        anom_key = f"anom:{self.temp_file.name}"
        deviations = get(anom_key)
        
        self.assertIsNotNone(deviations)
        self.assertEqual(len(deviations), 2)  # row and col indices
        self.assertGreater(len(deviations[0]), 0)  # Should find some anomalies

if __name__ == '__main__':
    unittest.main()