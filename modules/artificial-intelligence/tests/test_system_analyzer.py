

import unittest
import ast
from pathlib import Path
import os
import sys
import textwrap

# Add the parent directory to the path to allow importing the system_analyzer
sys.path.insert(0, str(Path(__file__).parent.parent))

from system_analyzer import CodeAnalyzer

class TestSystemAnalyzer(unittest.TestCase):
    """
    Tests the CodeAnalyzer class from the system_analyzer.py script.
    """

    def setUp(self):
        """Set up a dummy python file for testing."""
        self.test_file_content = textwrap.dedent('''
            import os
            import sys
            from typing import List

            class MyClass:
                """A test class."""
                def method_one(self, arg1: int):
                    """A test method."""
                    pass

            def my_function(arg1: str, arg2: List[str]) -> None:
                """A test function."""
                print("Hello")
        ''')
        self.test_file_path = Path("test_analyzer_temp_file.py")
        with open(self.test_file_path, "w") as f:
            f.write(self.test_file_content)

    def tearDown(self):
        """Remove the dummy python file."""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_code_analyzer(self):

        """
        Test that the CodeAnalyzer correctly extracts functions, classes, and imports.
        """
        analyzer = CodeAnalyzer(self.test_file_path)
        metadata = analyzer.analyze()

        # Test imports
        self.assertIn("os", metadata["imports"])
        self.assertIn("sys", metadata["imports"])
        self.assertIn("typing", metadata["imports"])

        # Test function detection
        self.assertEqual(len(metadata["functions"]), 1)
        self.assertEqual(metadata["functions"][0]["name"], "my_function")
        self.assertEqual(metadata["functions"][0]["args"], ["arg1", "arg2"])
        self.assertEqual(metadata["functions"][0]["docstring"], "A test function.")

        # Test class detection
        self.assertIn("MyClass", metadata["classes"])
        class_info = metadata["classes"]["MyClass"]
        self.assertEqual(class_info["name"], "MyClass")
        self.assertEqual(class_info["docstring"], "A test class.")

        # Test method detection
        self.assertEqual(len(class_info["methods"]), 1)
        method_info = class_info["methods"][0]
        self.assertEqual(method_info["name"], "method_one")
        self.assertEqual(method_info["args"], ["self", "arg1"])
        self.assertEqual(method_info["docstring"], "A test method.")

if __name__ == '__main__':
    unittest.main()

