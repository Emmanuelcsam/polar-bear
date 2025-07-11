import unittest
import sys
import numpy as np
import os
import tempfile
import shutil
sys.path.append('..')

class TestIntensityReader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        os.makedirs('data', exist_ok=True)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_learn_no_histograms(self):
        from modules.intensity_reader import learn
        from core.datastore import get
        learn()  # Should not crash
        dist = get("dist")
        self.assertIsNone(dist)
    
    def test_learn_with_histograms(self):
        from modules.intensity_reader import learn
        from core.datastore import put, get
        # Add some histograms
        hist1 = np.array([10] * 256)
        hist2 = np.array([20] * 256)
        put("hist:img1", hist1)
        put("hist:img2", hist2)
        
        learn()
        dist = get("dist")
        self.assertIsNotNone(dist)
        self.assertEqual(len(dist), 256)
        self.assertAlmostEqual(dist.sum(), 1.0)
        self.assertTrue(np.all(dist == dist[0]))  # All values should be equal
    
    def test_next_guided(self):
        from modules.intensity_reader import next_guided
        from core.datastore import put
        # Create distribution
        dist = np.ones(256) / 256
        put("dist", dist)
        
        img = next_guided()
        self.assertEqual(img.shape, (32, 32))
        self.assertEqual(img.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()