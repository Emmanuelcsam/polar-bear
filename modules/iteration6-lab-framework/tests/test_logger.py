import unittest
import sys
from io import StringIO
from unittest.mock import patch
import time
sys.path.append('..')
from core.logger import log

class TestLogger(unittest.TestCase):
    def test_log_basic(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            log("test", "hello")
            output = fake_out.getvalue()
            self.assertIn("[test]", output)
            self.assertIn("hello", output)
            self.assertIn(":", output)  # Time separator
    
    def test_log_multiple_args(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            log("multi", "arg1", "arg2", 123)
            output = fake_out.getvalue()
            self.assertIn("arg1", output)
            self.assertIn("arg2", output)
            self.assertIn("123", output)

if __name__ == '__main__':
    unittest.main()