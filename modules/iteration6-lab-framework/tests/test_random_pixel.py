import unittest
import sys
import numpy as np
import os
import tempfile
import shutil
sys.path.append('..')

class TestRandomPixel(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        os.makedirs('data', exist_ok=True)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_gen(self):
        from modules.random_pixel import gen, SIZE
        img = gen()
        self.assertEqual(img.shape, (SIZE, SIZE))
        self.assertEqual(img.dtype, np.uint8)
        self.assertTrue(np.all(img >= 0))
        self.assertTrue(np.all(img <= 255))
        # Check if file was created
        files = os.listdir('data')
        png_files = [f for f in files if f.endswith('.png') and f.startswith('rand_')]
        self.assertGreater(len(png_files), 0)
    
    def test_guided_without_dist(self):
        from modules.random_pixel import guided, SIZE
        img = guided()
        self.assertEqual(img.shape, (SIZE, SIZE))
        self.assertEqual(img.dtype, np.uint8)
    
    def test_guided_with_dist(self):
        from modules.random_pixel import guided, SIZE
        from core.datastore import put
        # Create a simple distribution
        dist = np.ones(256) / 256
        put("dist", dist)
        img = guided()
        self.assertEqual(img.shape, (SIZE, SIZE))
        self.assertEqual(img.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()