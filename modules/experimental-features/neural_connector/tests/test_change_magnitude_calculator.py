
import unittest
import os
import cv2
import numpy as np
import json
from pathlib import Path
import shutil

# Since the script is outside the package, we need to load it dynamically.
# This is similar to what the ModuleLoader does.
import importlib.util

# Path to the script to be tested
script_path = Path(__file__).parent.parent.parent / "change-magnitude-calculator.py"

# Load the script as a module
spec = importlib.util.spec_from_file_location("change_magnitude_calculator", script_path)
change_magnitude_calculator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(change_magnitude_calculator)


class TestChangeMagnitudeCalculator(unittest.TestCase):
    """
    Unit tests for the change-magnitude-calculator.py script.
    """

    def setUp(self):
        """
        Set up a temporary environment for testing.
        This method is called before each test function.
        """
        self.test_dir = Path("temp_test_data")
        self.output_dir = self.test_dir / "output"
        
        # Create directories
        self.test_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Create a dummy image for testing
        self.image_path = self.test_dir / "test_gradient.png"
        self.create_test_image(self.image_path)

    def tearDown(self):
        """
        Clean up the temporary environment after testing.
        This method is called after each test function.
        """
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def create_test_image(self, path: Path):
        """Creates a simple 100x100 image with a gradient from left to right."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            img[:, i] = int(i * 2.55) # Gradient from 0 to 255
        cv2.imwrite(str(path), img)

    def test_analyze_image_creates_output_files(self):
        """
        Test if analyze_image function successfully creates all expected output files.
        """
        # Run the function from the imported script
        change_magnitude_calculator.analyze_image(
            image_path=str(self.image_path),
            output_dir=str(self.output_dir)
        )

        # --- Assertions ---
        # 1. Check if all three output files were created
        base_filename = self.image_path.stem
        json_path = self.output_dir / f"{base_filename}_pixel_values.json"
        map_path = self.output_dir / f"{base_filename}_change_map.png"
        hist_path = self.output_dir / f"{base_filename}_change_histogram.png"

        self.assertTrue(json_path.exists(), "Pixel values JSON file was not created.")
        self.assertTrue(map_path.exists(), "Change map image was not created.")
        self.assertTrue(hist_path.exists(), "Change histogram image was not created.")

        # 2. Check if the created files are not empty
        self.assertGreater(json_path.stat().st_size, 0, "JSON file is empty.")
        self.assertGreater(map_path.stat().st_size, 0, "Change map image is empty.")
        self.assertGreater(hist_path.stat().st_size, 0, "Histogram image is empty.")

    def test_pixel_values_json_content(self):
        """
        Test the content of the generated pixel values JSON file.
        """
        change_magnitude_calculator.analyze_image(
            image_path=str(self.image_path),
            output_dir=str(self.output_dir)
        )
        
        json_path = self.output_dir / f"{self.image_path.stem}_pixel_values.json"
        
        # Check if the JSON is valid
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                self.fail("Could not decode the generated JSON file.")

        # Check for expected structure and content
        self.assertIsInstance(data, list, "JSON root should be a list.")
        self.assertEqual(len(data), 100 * 100, "JSON should contain an entry for every pixel.")
        
        # Check the first pixel's data
        first_pixel = data[0]
        self.assertIn("coordinates", first_pixel)
        self.assertIn("bgr_value", first_pixel)
        self.assertEqual(first_pixel["coordinates"], {"x": 0, "y": 0})
        self.assertEqual(first_pixel["bgr_value"], [0, 0, 0]) # First pixel is black

        # Check the last pixel's data
        last_pixel = data[-1]
        self.assertEqual(last_pixel["coordinates"], {"x": 99, "y": 99})
        # The gradient goes up to 99 * 2.55 = 252.45, which is 252 as an int
        self.assertEqual(last_pixel["bgr_value"], [252, 252, 252])


if __name__ == '__main__':
    unittest.main()
