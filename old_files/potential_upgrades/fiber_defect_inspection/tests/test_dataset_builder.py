# tests/test_dataset_builder.py
import unittest
import pandas as pd
from pathlib import Path
import os
import shutil

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset_builder import build_datasets

class TestDatasetBuilder(unittest.TestCase):

    def setUp(self):
        """Set up a dummy DataFrame and config for testing."""
        data = {
            'image_name': ['img1.png', 'img1.png', 'img2.png'],
            'defect_id': [1, 2, 1],
            'cluster': [0, 1, 0],
            'zone': ['core', 'cladding', 'core'],
            'area_px': [100, 50, 120]
        }
        self.clustered_df = pd.DataFrame(data)
        
        self.config = {
            'output_folder': 'tests/test_output',
            'defect_library_folder': 'tests/test_output/defect_library'
        }
        
        # Create test output directories
        os.makedirs(self.config['output_folder'], exist_ok=True)
        os.makedirs(self.config['defect_library_folder'], exist_ok=True)

    def test_build_datasets(self):
        """Test the build_datasets function."""
        build_datasets(self.clustered_df, self.config)
        
        # Check if the output files were created
        self.assertTrue(Path(self.config['output_folder'], 'image_level_dataset.csv').exists())
        self.assertTrue(Path(self.config['output_folder'], 'region_level_dataset.csv').exists())
        self.assertTrue(Path(self.config['defect_library_folder'], 'defect_library_index.csv').exists())

    def tearDown(self):
        """Clean up the test output files."""
        shutil.rmtree(self.config['output_folder'])

if __name__ == '__main__':
    import shutil
    unittest.main()
