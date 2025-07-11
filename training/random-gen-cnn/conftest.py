"""
Test configuration and fixtures for the image categorization system
"""
import pytest
import tempfile
import os
import shutil
import numpy as np
from PIL import Image
import json
import pickle


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_image_rgb():
    """Create a sample RGB image as numpy array"""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_pixel_db():
    """Create a sample pixel database"""
    return {
        'red': [np.array([255, 0, 0]), np.array([250, 5, 5]), np.array([245, 10, 10])],
        'blue': [np.array([0, 0, 255]), np.array([5, 5, 250]), np.array([10, 10, 245])],
        'green': [np.array([0, 255, 0]), np.array([5, 250, 5]), np.array([10, 245, 10])]
    }


@pytest.fixture
def sample_weights():
    """Create sample weights dictionary"""
    return {
        'red': 1.0,
        'blue': 1.2,
        'green': 0.8
    }


@pytest.fixture
def sample_results():
    """Create sample analysis results"""
    return {
        'image1.jpg': {
            'category': 'red',
            'confidence': 0.85,
            'scores': {'red': 0.8, 'blue': 0.15, 'green': 0.05},
            'timestamp': '2024-01-01T12:00:00'
        },
        'image2.jpg': {
            'category': 'blue',
            'confidence': 0.92,
            'scores': {'red': 0.05, 'blue': 0.9, 'green': 0.05},
            'timestamp': '2024-01-01T12:01:00'
        },
        'image3.jpg': {
            'category': 'red',
            'confidence': 0.78,
            'scores': {'red': 0.75, 'blue': 0.2, 'green': 0.05},
            'timestamp': '2024-01-01T12:02:00'
        }
    }


@pytest.fixture
def create_test_image():
    """Factory function to create test images"""
    def _create_image(color, size=(50, 50), format='RGB'):
        """Create a test image with specified color"""
        if isinstance(color, str):
            img = Image.new(format, size, color=color)
        else:
            img = Image.new(format, size, color=tuple(color))
        return img
    return _create_image


@pytest.fixture
def create_test_directory_structure():
    """Factory function to create test directory structures"""
    def _create_structure(base_dir, structure):
        """
        Create directory structure from dict
        structure = {
            'dir1': {
                'subdir1': ['file1.jpg', 'file2.txt'],
                'subdir2': ['file3.png']
            },
            'dir2': ['file4.jpg']
        }
        """
        for name, content in structure.items():
            if isinstance(content, dict):
                # It's a directory with subdirectories
                dir_path = os.path.join(base_dir, name)
                os.makedirs(dir_path, exist_ok=True)
                _create_structure(dir_path, content)
            elif isinstance(content, list):
                # It's a directory with files
                dir_path = os.path.join(base_dir, name)
                os.makedirs(dir_path, exist_ok=True)
                for filename in content:
                    file_path = os.path.join(dir_path, filename)
                    # Create empty file or basic image
                    if filename.endswith(('.jpg', '.png', '.jpeg')):
                        img = Image.new('RGB', (10, 10), color='white')
                        img.save(file_path)
                    else:
                        open(file_path, 'w').close()
    return _create_structure


@pytest.fixture
def save_test_data():
    """Factory function to save test data to files"""
    def _save_data(data, filepath, format='pickle'):
        """Save data to file in specified format"""
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        return filepath
    return _save_data


@pytest.fixture
def mock_analyze_image():
    """Mock function for image analysis"""
    def _mock_analyze(img_path, pixel_db, weights, comparisons=100):
        """Mock analyze_image function that returns predictable results"""
        # Simple mock: return first category with high confidence
        categories = list(pixel_db.keys())
        if categories:
            best_category = categories[0]
            scores = {cat: 0.1 for cat in categories}
            scores[best_category] = 0.8
            confidence = 0.8
            return best_category, scores, confidence
        else:
            return 'unknown', {}, 0.0
    return _mock_analyze


# Test data constants
SAMPLE_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
SAMPLE_NON_IMAGE_EXTENSIONS = ['.txt', '.pdf', '.doc', '.mp4', '.exe']

# Test configuration
TEST_CONFIG = {
    'sample_size': 10,
    'comparisons': 50,
    'confidence_threshold': 0.5,
    'learning_rate': 0.1,
    'std_threshold': 2.0
}
