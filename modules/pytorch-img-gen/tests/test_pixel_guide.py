import unittest
import importlib


class TestPixel_Guide(unittest.TestCase):


    def setUp(self):
        """Load the module for testing."""
        try:
            self.module = importlib.import_module('pixel-guide')
        except ImportError as e:
            self.fail(f"Failed to import module 'pixel-guide': {e}")

    def test_module_is_loadable(self):
        """Test that the module object was loaded successfully."""
        self.assertIsNotNone(self.module, "Module should not be None")

if __name__ == '__main__':
    unittest.main()
