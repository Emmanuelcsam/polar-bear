import unittest
import sys
import numpy as np
import os
import tempfile
import shutil
sys.path.append('..')

class TestPatternRecognizer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        os.makedirs('data', exist_ok=True)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_cluster_no_data(self):
        from modules.pattern_recognizer import cluster
        cluster(k=3)  # Should not crash
    
    def test_cluster_with_data(self):
        from modules.pattern_recognizer import cluster
        from core.datastore import put, get
        
        # Create some different histograms
        hist1 = np.array([100] + [0] * 255)  # Dark image
        hist2 = np.array([0] * 255 + [100])  # Bright image
        hist3 = np.array([0] * 128 + [100] + [0] * 127)  # Middle gray
        
        put("hist:img1", hist1)
        put("hist:img2", hist2)
        put("hist:img3", hist3)
        
        cluster(k=3)
        
        # Check if categories were assigned
        cat1 = get("cat:hist:img1")
        cat2 = get("cat:hist:img2")
        cat3 = get("cat:hist:img3")
        
        self.assertIsNotNone(cat1)
        self.assertIsNotNone(cat2)
        self.assertIsNotNone(cat3)
        self.assertIn(cat1, [0, 1, 2])
        self.assertIn(cat2, [0, 1, 2])
        self.assertIn(cat3, [0, 1, 2])

if __name__ == '__main__':
    unittest.main()