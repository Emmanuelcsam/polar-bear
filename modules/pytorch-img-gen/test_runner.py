import os
import sys
import unittest
import logging
import ast
from pathlib import Path

# --- Constants ---
LOG_FILE_NAME = "connector.log"
CURRENT_DIR = Path(__file__).parent.resolve()
TEST_DIR = CURRENT_DIR / "tests"
WHITELISTED_FILES = [
    "feature-extractor.py",
    "image-saver.py",
    "live-display.py",
    "neural-generator.py",
    "noise-generator.py",
    "pixel-guide.py",
    "ref-processor.py",
    "style-optimizer.py",
    "texture-synth.py",
]

# --- Setup Logging ---
# A simple logger for the test runner itself. More comprehensive logging is in connector.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_NAME, mode='a'), # Append to the main log file
        logging.StreamHandler(sys.stdout)
    ]
)

class FunctionVisitor(ast.NodeVisitor):
    """AST visitor to find all function definitions in a file."""
    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        # We can add more complex logic here to ignore private/protected methods if needed
        if not node.name.startswith('_'):
            self.functions.append(node.name)
        self.generic_visit(node)

def generate_test_stubs():
    """
    Generates placeholder test files for all whitelisted modules
    if they don't already exist.
    """
    logging.info("Starting test stub generation...")
    TEST_DIR.mkdir(exist_ok=True)
    
    # Create an __init__.py in the tests directory to make it a package
    (TEST_DIR / "__init__.py").touch()

    for file_name in WHITELISTED_FILES:
        module_name = file_name.replace('.py', '')
        class_name = f"Test{module_name.replace('-', '_').title()}"
        # Sanitize test file name
        test_file_name = f"test_{module_name.replace('-', '_')}.py"
        test_file_path = TEST_DIR / test_file_name

        if test_file_path.exists():
            logging.info(f"Test file '{test_file_name}' already exists. Skipping.")
            continue

        source_file_path = CURRENT_DIR / file_name
        if not source_file_path.exists():
            logging.warning(f"Source file '{file_name}' not found. Cannot generate tests.")
            continue

        logging.info(f"Generating test stub for '{module_name}' -> '{test_file_name}'")

        try:
            with open(source_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content, filename=file_name)
            visitor = FunctionVisitor()
            visitor.visit(tree)
            functions_to_test = visitor.functions
        except Exception as e:
            logging.error(f"Could not parse '{file_name}' to generate tests: {e}")
            continue

        # Build the test file content line by line to avoid syntax errors
        lines = [
            "import unittest",
            "import importlib",
            "\n",
            f"class {class_name}(unittest.TestCase):",
            "\n",
            "    def setUp(self):",
            '        """Load the module for testing."""',
            "        try:",
            f"            self.module = importlib.import_module('{module_name}')",
            "        except ImportError as e:",
            f"            self.fail(f\"Failed to import module '{module_name}': {{e}}\")",
            ""
        ]

        if not functions_to_test:
            lines.extend([
                "    def test_module_is_loadable(self):",
                '        """Test that the module object was loaded successfully."""',
                '        self.assertIsNotNone(self.module, "Module should not be None")',
                ""
            ])

        for func_name in functions_to_test:
            lines.extend([
                f"    def test_{func_name}(self):",
                f'        """Test case for {func_name}()"""',
                f"        self.assertTrue(hasattr(self.module, '{func_name}'), \"Function {func_name} not found in module\")",
                f"        # TODO: Implement a real test for {func_name}",
                f'        self.fail("Test not implemented for {func_name}")', 
                ""
            ])
        
        lines.extend([
            "if __name__ == '__main__':",
            "    unittest.main()",
            ""
        ])

        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        logging.info(f"Successfully created '{test_file_name}'.")



def run_tests():
    """Discovers and runs all tests in the 'tests' directory."""
    logging.info("Discovering and running tests...")
    print("\n" + "="*70)
    print(" " * 25 + "RUNNING TEST SUITE")
    print("="*70)
    
    # Add parent directory to path to allow test files to import the modules
    sys.path.insert(0, str(CURRENT_DIR))

    loader = unittest.TestLoader()
    suite = loader.discover(str(TEST_DIR), pattern="test_*.py")
    
    if suite.countTestCases() == 0:
        logging.warning("No tests were found in the 'tests' directory.")
        print("No tests found. Generate stubs first if needed.")
        return

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Restore sys.path
    sys.path.pop(0)
    
    print("="*70)
    logging.info(f"Test run complete. Success: {result.wasSuccessful()}")
    return result.wasSuccessful()


if __name__ == "__main__":
    # This allows running the test runner directly
    print("Test Runner Utility")
    print("1. Generate test stubs for missing test files")
    print("2. Run all tests")
    print("3. Generate stubs AND run all tests")
    
    choice = input("Enter your choice: ")
    
    if choice == '1':
        generate_test_stubs()
    elif choice == '2':
        run_tests()
    elif choice == '3':
        generate_test_stubs()
        run_tests()
    else:
        print("Invalid choice.")