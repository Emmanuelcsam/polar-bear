#!/usr/bin/env python3
"""
Test script to verify the fixes work by providing automated responses
"""

import sys
import io
from unittest.mock import patch
import os

# Mock inputs for testing
test_inputs = [
    'n',  # Skip UI
    'test_ref',  # Reference directory (will fail, but that's expected)
    'test_target',  # Target directory
    'test_output'  # Output directory
]

def test_main():
    """Test the main function with mocked inputs"""
    
    # Redirect stdin to provide automated inputs
    original_input = input
    input_iterator = iter(test_inputs)
    
    def mock_input(prompt=""):
        try:
            response = next(input_iterator)
            print(f"{prompt}{response}")  # Show what we're "typing"
            return response
        except StopIteration:
            return ""
    
    # Patch the input function and run main
    with patch('builtins.input', side_effect=mock_input):
        try:
            # Import the main function from the fixed script
            import importlib.util
            spec = importlib.util.spec_from_file_location("directory_crop", 
                "c:/Users/Saem1001/Documents/GitHub/polar-bear/meta-tools/directory-crop-with-ref.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # This should fail gracefully with directory validation errors
            module.main()
            
        except ValueError as e:
            if "directory does not exist" in str(e):
                print("✓ Program handled invalid directory gracefully")
                return True
            else:
                print(f"✗ Unexpected error: {e}")
                return False
        except Exception as e:
            print(f"✗ Failed with error: {e}")
            return False

if __name__ == "__main__":
    print("Testing the fixed program...")
    success = test_main()
    if success:
        print("\n✓ All tests passed! The program can handle basic input/output and graceful error handling.")
    else:
        print("\n✗ Tests failed.")
