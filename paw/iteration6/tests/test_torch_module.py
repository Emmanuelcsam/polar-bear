import unittest
import sys
import torch
import os
import tempfile
import shutil
from PIL import Image
import numpy as np
sys.path.append('..')

class TestTorchModule(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        os.makedirs('data', exist_ok=True)
        os.makedirs('test_images', exist_ok=True)
        
        # Create test images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
            img.save(f'test_images/test_{i}.png')
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_model_structure(self):
        from modules.torch_module import model, AE
        self.assertIsInstance(model, AE)
        self.assertTrue(hasattr(model, 'enc'))
        self.assertTrue(hasattr(model, 'dec'))
    
    def test_train_folder(self):
        from modules.torch_module import train_folder
        from core.datastore import get
        
        train_folder('test_images', epochs=1)
        
        # Check if model was saved
        saved_state = get("ae")
        self.assertIsNotNone(saved_state)
    
    def test_generate(self):
        from modules.torch_module import generate
        from core.datastore import get
        
        imgs = generate(n=2)
        
        self.assertEqual(imgs.shape, (2, 1, 32, 32))
        self.assertEqual(imgs.dtype, np.uint8)
        
        # Check if generated images were saved
        saved_imgs = get("gen")
        self.assertIsNotNone(saved_imgs)
        self.assertEqual(saved_imgs.shape, imgs.shape)

if __name__ == '__main__':
    unittest.main()