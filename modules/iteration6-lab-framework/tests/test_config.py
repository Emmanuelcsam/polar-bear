import unittest
import sys
import os
import pathlib
sys.path.append('..')
from core.config import BASE, DATA, DEVICE, CORES, PRINT_PREFIX

class TestConfig(unittest.TestCase):
    def test_base_path_is_absolute(self):
        self.assertTrue(BASE.is_absolute())
        self.assertIsInstance(BASE, pathlib.Path)
    
    def test_data_directory_exists(self):
        self.assertTrue(DATA.exists())
        self.assertTrue(DATA.is_dir())
    
    def test_device_setting(self):
        self.assertIn(DEVICE, ["cuda", "cpu"])
        if os.getenv("USE_CPU"):
            self.assertEqual(DEVICE, "cpu")
    
    def test_cores_positive(self):
        self.assertGreater(CORES, 0)
        self.assertIsInstance(CORES, int)
    
    def test_print_prefix(self):
        self.assertEqual(PRINT_PREFIX, "[config]")

if __name__ == '__main__':
    unittest.main()