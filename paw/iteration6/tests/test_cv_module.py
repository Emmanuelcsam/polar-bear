import unittest
import sys
import numpy as np
import cv2
import tempfile
import os
sys.path.append('..')
from modules.cv_module import load_gray, hist, anomalies, RES

class TestCVModule(unittest.TestCase):
    def setUp(self):
        self.test_img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(self.temp_file.name, self.test_img)
    
    def tearDown(self):
        os.unlink(self.temp_file.name)
    
    def test_load_gray(self):
        img = load_gray(self.temp_file.name)
        self.assertIsNotNone(img)
        self.assertEqual(len(img.shape), 2)  # Grayscale
    
    def test_hist(self):
        h = hist(self.test_img)
        self.assertEqual(len(h), 256)
        self.assertEqual(h.sum(), self.test_img.size)
    
    def test_anomalies(self):
        img = np.ones((10, 10), dtype=np.uint8) * 128
        img[5, 5] = 255  # Outlier
        mask = anomalies(img, z=2)
        self.assertTrue(mask[5, 5])
        self.assertFalse(mask[0, 0])
    
    def test_res_constant(self):
        self.assertEqual(RES, (32, 32))

if __name__ == '__main__':
    unittest.main()