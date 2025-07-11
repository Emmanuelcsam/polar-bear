import unittest
import sys
import os
import tempfile
import shutil
import time
import cv2
import numpy as np
import threading
sys.path.append('..')

class TestRealtimeProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        os.makedirs('data', exist_ok=True)
        os.makedirs('watch_dir', exist_ok=True)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_watch(self):
        from modules.realtime_processor import watch
        from core.datastore import scan
        
        # Start watching
        watch('watch_dir', poll=0.1)
        
        # Give it time to start
        time.sleep(0.2)
        
        # Add an image
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        cv2.imwrite('watch_dir/new_img.png', img)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check if histogram was created
        hists = scan("hist:")
        found = False
        for key, _ in hists:
            if 'new_img.png' in key:
                found = True
                break
        self.assertTrue(found)

if __name__ == '__main__':
    unittest.main()