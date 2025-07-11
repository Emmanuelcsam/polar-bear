import unittest
import importlib


class TestRef_Processor(unittest.TestCase):


    def setUp(self):
        """Load the module for testing."""
        try:
            self.module = importlib.import_module('ref-processor')
        except ImportError as e:
            self.fail(f"Failed to import module 'ref-processor': {e}")

    def test_module_is_loadable(self):
        """Test that the module object was loaded successfully."""
        self.assertIsNotNone(self.module, "Module should not be None")

if __name__ == '__main__':
    unittest.main()
