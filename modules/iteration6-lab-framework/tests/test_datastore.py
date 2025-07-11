import unittest
import sys
import os
import tempfile
import shutil
sys.path.append('..')

class TestDatastore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        os.makedirs('data', exist_ok=True)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_put_get(self):
        from core.datastore import put, get
        test_data = {"key": "value", "num": 42}
        put("test_key", test_data)
        retrieved = get("test_key")
        self.assertEqual(retrieved, test_data)
    
    def test_get_default(self):
        from core.datastore import get
        result = get("nonexistent", default="default_value")
        self.assertEqual(result, "default_value")
    
    def test_scan(self):
        from core.datastore import put, scan
        put("prefix:1", "value1")
        put("prefix:2", "value2")
        put("other:3", "value3")
        results = scan("prefix:")
        self.assertEqual(len(results), 2)
        keys = [k for k, v in results]
        self.assertIn("prefix:1", keys)
        self.assertIn("prefix:2", keys)

if __name__ == '__main__':
    unittest.main()