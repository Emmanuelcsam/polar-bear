import unittest
import sys
import os
import tempfile
import shutil
import cv2
import numpy as np
sys.path.append('..')

class TestBatchProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        os.makedirs('data', exist_ok=True)
        os.makedirs('test_images', exist_ok=True)
        
        # Create test images
        for i in range(3):
            img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            cv2.imwrite(f'test_images/test_{i}.png', img)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_run_valid_folder(self):
        from modules.batch_processor import run
        from core.datastore import scan, get
        
        run('test_images')
        
        # Check if histograms were created
        hists = scan("hist:")
        self.assertEqual(len(hists), 3)
        
        # Check if distribution was learned
        dist = get("dist")
        self.assertIsNotNone(dist)
        self.assertEqual(len(dist), 256)
    
    def test_run_invalid_folder(self):
        from modules.batch_processor import run
        
        with self.assertRaises(SystemExit):
            run('nonexistent_folder')

if __name__ == '__main__':
    unittest.main()