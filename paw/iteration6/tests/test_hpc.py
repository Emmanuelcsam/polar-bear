import unittest
import sys
import os
import tempfile
import shutil
import glob
sys.path.append('..')

class TestHPC(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        os.makedirs('data', exist_ok=True)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_run(self):
        from modules.hpc import run
        from core.config import CORES
        
        # Run with small number
        run(total=10)
        
        # Check if images were created
        images = glob.glob('data/rand_*.png')
        self.assertGreater(len(images), 0)
        self.assertLessEqual(len(images), 10)
    
    def test_worker_function(self):
        from modules.hpc import _worker
        
        # Test worker directly
        _worker(2)
        
        # Check if images were created
        images = glob.glob('data/rand_*.png')
        self.assertEqual(len(images), 2)

if __name__ == '__main__':
    unittest.main()