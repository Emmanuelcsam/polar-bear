"""
Comprehensive unit tests for the image categorization system
"""
import pytest
import numpy as np
import os
import json
import pickle
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

# Import all refactored modules
import auto_installer_refactored as auto_installer
import pixel_sampler_refactored as pixel_sampler
import correlation_analyzer_refactored as correlation_analyzer
import batch_processor_refactored as batch_processor
import self_reviewer_refactored as self_reviewer


class TestAutoInstaller:
    """Test cases for auto_installer_refactored module"""
    
    def test_check_library_installed_success(self):
        """Test successful library check"""
        # Test with a library that should be installed (os)
        assert auto_installer.check_library_installed('os') == True
    
    def test_check_library_installed_failure(self):
        """Test failed library check"""
        # Test with a non-existent library
        assert auto_installer.check_library_installed('nonexistent_library_12345') == False
    
    def test_check_library_with_dash(self):
        """Test library name with dash conversion"""
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = True
            auto_installer.check_library_installed('test-library')
            mock_import.assert_called_with('test_library')
    
    @patch('subprocess.check_call')
    def test_install_library_success(self, mock_subprocess):
        """Test successful library installation"""
        mock_subprocess.return_value = None
        result = auto_installer.install_library('test_library')
        assert result == True
        mock_subprocess.assert_called_once()
    
    @patch('subprocess.check_call')
    def test_install_library_failure(self, mock_subprocess):
        """Test failed library installation"""
        mock_subprocess.side_effect = Exception("Install failed")
        result = auto_installer.install_library('test_library')
        assert result == False
    
    @patch('auto_installer_refactored.check_library_installed')
    @patch('auto_installer_refactored.install_library')
    def test_auto_install_dependencies(self, mock_install, mock_check):
        """Test auto-install dependencies function"""
        # Setup mocks
        mock_check.side_effect = [True, False, False]  # First lib installed, others not
        mock_install.side_effect = [True, False]  # Second installs successfully, third fails
        
        libraries = ['lib1', 'lib2', 'lib3']
        results = auto_installer.auto_install_dependencies(libraries)
        
        assert results['lib1'] == 'already_installed'
        assert results['lib2'] == 'installed'
        assert results['lib3'] == 'failed'


class TestPixelSampler:
    """Test cases for pixel_sampler_refactored module"""
    
    def test_is_image_file(self):
        """Test image file detection"""
        assert pixel_sampler.is_image_file('test.jpg') == True
        assert pixel_sampler.is_image_file('test.PNG') == True
        assert pixel_sampler.is_image_file('test.jpeg') == True
        assert pixel_sampler.is_image_file('test.txt') == False
        assert pixel_sampler.is_image_file('test.py') == False
    
    def test_load_image_success(self):
        """Test successful image loading"""
        # Create a temporary image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            result = pixel_sampler.load_image(tmp_path)
            assert result is not None
            assert result.shape == (100, 100, 3)
            assert result.dtype == np.uint8
        finally:
            os.unlink(tmp_path)
    
    def test_load_image_failure(self):
        """Test failed image loading"""
        result = pixel_sampler.load_image('nonexistent_file.jpg')
        assert result is None
    
    def test_sample_pixels_from_image(self):
        """Test pixel sampling from image"""
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Sample pixels
        pixels = pixel_sampler.sample_pixels_from_image(test_image, 10)
        
        assert len(pixels) == 10
        for pixel in pixels:
            assert pixel.shape == (3,)
            assert pixel.dtype == np.uint8
    
    def test_build_pixel_database_nonexistent_dir(self):
        """Test pixel database building with nonexistent directory"""
        with pytest.raises(ValueError, match="Reference directory does not exist"):
            pixel_sampler.build_pixel_database('/nonexistent/path')
    
    def test_build_pixel_database_success(self):
        """Test successful pixel database building"""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create category directories
            cat1_dir = os.path.join(tmp_dir, 'category1')
            cat2_dir = os.path.join(tmp_dir, 'category2')
            os.makedirs(cat1_dir)
            os.makedirs(cat2_dir)
            
            # Create test images
            img1 = Image.new('RGB', (50, 50), color='red')
            img2 = Image.new('RGB', (50, 50), color='blue')
            img1.save(os.path.join(cat1_dir, 'test1.jpg'))
            img2.save(os.path.join(cat2_dir, 'test2.jpg'))
            
            # Build database
            pixel_db = pixel_sampler.build_pixel_database(tmp_dir, sample_size=5)
            
            assert len(pixel_db) == 2
            assert 'category1' in pixel_db
            assert 'category2' in pixel_db
            assert len(pixel_db['category1']) == 5
            assert len(pixel_db['category2']) == 5
    
    def test_save_load_pixel_database(self):
        """Test saving and loading pixel database"""
        # Create test database
        test_db = {
            'cat1': [np.array([255, 0, 0]), np.array([0, 255, 0])],
            'cat2': [np.array([0, 0, 255]), np.array([255, 255, 0])]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test saving
            result = pixel_sampler.save_pixel_database(test_db, tmp_path)
            assert result == True
            
            # Test loading
            loaded_db = pixel_sampler.load_pixel_database(tmp_path)
            assert loaded_db is not None
            assert len(loaded_db) == 2
            assert 'cat1' in loaded_db
            assert 'cat2' in loaded_db
            
        finally:
            os.unlink(tmp_path)
    
    def test_get_database_stats(self):
        """Test database statistics calculation"""
        test_db = {
            'cat1': [np.array([255, 0, 0]), np.array([0, 255, 0])],
            'cat2': [np.array([0, 0, 255])]
        }
        
        stats = pixel_sampler.get_database_stats(test_db)
        
        assert stats['categories'] == 2
        assert stats['total_pixels'] == 3
        assert stats['pixels_per_category']['cat1'] == 2
        assert stats['pixels_per_category']['cat2'] == 1


class TestCorrelationAnalyzer:
    """Test cases for correlation_analyzer_refactored module"""
    
    def test_calculate_pixel_similarity(self):
        """Test pixel similarity calculation"""
        pixel1 = np.array([255, 0, 0])  # Red
        pixel2 = np.array([255, 0, 0])  # Red (identical)
        pixel3 = np.array([0, 255, 0])  # Green (different)
        
        # Identical pixels should have high similarity
        sim1 = correlation_analyzer.calculate_pixel_similarity(pixel1, pixel2)
        assert sim1 == 1.0
        
        # Different pixels should have lower similarity
        sim2 = correlation_analyzer.calculate_pixel_similarity(pixel1, pixel3)
        assert 0 < sim2 < 1.0
    
    def test_load_save_weights(self):
        """Test weights loading and saving"""
        test_weights = {'cat1': 1.5, 'cat2': 0.8, 'cat3': 1.2}
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test saving
            result = correlation_analyzer.save_weights(test_weights, tmp_path)
            assert result == True
            
            # Test loading
            loaded_weights = correlation_analyzer.load_weights(tmp_path)
            assert loaded_weights == test_weights
            
        finally:
            os.unlink(tmp_path)
    
    def test_analyze_image_nonexistent_file(self):
        """Test image analysis with nonexistent file"""
        pixel_db = {'cat1': [np.array([255, 0, 0])]}
        weights = {'cat1': 1.0}
        
        with pytest.raises(ValueError, match="Image file does not exist"):
            correlation_analyzer.analyze_image('nonexistent.jpg', pixel_db, weights)
    
    def test_analyze_image_success(self):
        """Test successful image analysis"""
        # Create temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Create test database
            pixel_db = {
                'red': [np.array([255, 0, 0]), np.array([250, 5, 5])],
                'blue': [np.array([0, 0, 255]), np.array([5, 5, 250])]
            }
            weights = {'red': 1.0, 'blue': 1.0}
            
            category, scores, confidence = correlation_analyzer.analyze_image(
                tmp_path, pixel_db, weights, comparisons=10
            )
            
            # Should classify as red since image is red
            assert category == 'red'
            assert 'red' in scores
            assert 'blue' in scores
            assert 0 <= confidence <= 1
            assert scores['red'] > scores['blue']
            
        finally:
            os.unlink(tmp_path)
    
    def test_update_weights_from_feedback(self):
        """Test weight updates from feedback"""
        initial_weights = {'cat1': 1.0, 'cat2': 1.0, 'cat3': 1.0}
        
        # Test correct prediction (should boost correct category, reduce predicted)
        updated_weights = correlation_analyzer.update_weights_from_feedback(
            initial_weights, 'cat1', 'cat2', learning_rate=0.1
        )
        
        assert updated_weights['cat2'] > initial_weights['cat2']  # Correct category boosted
        assert updated_weights['cat1'] < initial_weights['cat1']  # Predicted category reduced
        assert updated_weights['cat3'] == initial_weights['cat3']  # Unchanged
    
    def test_get_image_files_from_directory(self):
        """Test getting image files from directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            open(os.path.join(tmp_dir, 'image1.jpg'), 'w').close()
            open(os.path.join(tmp_dir, 'image2.png'), 'w').close()
            open(os.path.join(tmp_dir, 'document.txt'), 'w').close()
            open(os.path.join(tmp_dir, 'image3.JPEG'), 'w').close()
            
            image_files = correlation_analyzer.get_image_files_from_directory(tmp_dir)
            
            assert len(image_files) == 3
            assert any('image1.jpg' in f for f in image_files)
            assert any('image2.png' in f for f in image_files)
            assert any('image3.JPEG' in f for f in image_files)
            assert not any('document.txt' in f for f in image_files)


class TestBatchProcessor:
    """Test cases for batch_processor_refactored module"""
    
    def test_get_image_files(self):
        """Test getting image files from directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            open(os.path.join(tmp_dir, 'image1.jpg'), 'w').close()
            open(os.path.join(tmp_dir, 'image2.png'), 'w').close()
            open(os.path.join(tmp_dir, 'not_image.txt'), 'w').close()
            
            image_files = batch_processor.get_image_files(tmp_dir)
            
            assert len(image_files) == 2
            assert 'image1.jpg' in image_files
            assert 'image2.png' in image_files
            assert 'not_image.txt' not in image_files
    
    def test_get_image_files_nonexistent_dir(self):
        """Test getting image files from nonexistent directory"""
        result = batch_processor.get_image_files('/nonexistent/path')
        assert result == []
    
    def test_save_load_results(self):
        """Test saving and loading results"""
        test_results = {
            'image1.jpg': {
                'category': 'cat1',
                'confidence': 0.85,
                'scores': {'cat1': 0.8, 'cat2': 0.2}
            },
            'image2.jpg': {
                'category': 'cat2',
                'confidence': 0.92,
                'scores': {'cat1': 0.1, 'cat2': 0.9}
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test saving
            output_file = batch_processor.save_results(test_results, tmp_path)
            assert output_file == tmp_path
            
            # Test loading
            loaded_results = batch_processor.load_results(tmp_path)
            assert loaded_results == test_results
            
        finally:
            os.unlink(tmp_path)
    
    def test_get_category_distribution(self):
        """Test category distribution calculation"""
        test_results = {
            'img1.jpg': {'category': 'cat1'},
            'img2.jpg': {'category': 'cat1'},
            'img3.jpg': {'category': 'cat2'},
            'img4.jpg': {'category': 'cat1'},
            'img5.jpg': {'category': 'cat3'}
        }
        
        distribution = batch_processor.get_category_distribution(test_results)
        
        assert distribution['cat1'] == 3
        assert distribution['cat2'] == 1
        assert distribution['cat3'] == 1
    
    @patch('batch_processor_refactored.ca.analyze_image')
    def test_process_batch_success(self, mock_analyze):
        """Test successful batch processing"""
        # Setup mock
        mock_analyze.return_value = ('cat1', {'cat1': 0.8, 'cat2': 0.2}, 0.8)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test images
            open(os.path.join(tmp_dir, 'test1.jpg'), 'w').close()
            open(os.path.join(tmp_dir, 'test2.jpg'), 'w').close()
            
            pixel_db = {'cat1': [np.array([255, 0, 0])]}
            weights = {'cat1': 1.0}
            
            results = batch_processor.process_batch(tmp_dir, pixel_db, weights)
            
            assert len(results) == 2
            assert 'test1.jpg' in results
            assert 'test2.jpg' in results
            assert results['test1.jpg']['category'] == 'cat1'
            assert results['test2.jpg']['category'] == 'cat1'
    
    def test_process_batch_nonexistent_dir(self):
        """Test batch processing with nonexistent directory"""
        with pytest.raises(ValueError, match="Batch directory does not exist"):
            batch_processor.process_batch('/nonexistent/path', {}, {})
    
    def test_process_batch_no_images(self):
        """Test batch processing with no images"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create non-image file
            open(os.path.join(tmp_dir, 'document.txt'), 'w').close()
            
            with pytest.raises(ValueError, match="No image files found"):
                batch_processor.process_batch(tmp_dir, {}, {})


class TestSelfReviewer:
    """Test cases for self_reviewer_refactored module"""
    
    def test_load_results_success(self):
        """Test successful results loading"""
        test_results = {
            'image1.jpg': {
                'category': 'cat1',
                'confidence': 0.85
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(test_results, tmp)
            tmp_path = tmp.name
        
        try:
            loaded_results = self_reviewer.load_results(tmp_path)
            assert loaded_results == test_results
        finally:
            os.unlink(tmp_path)
    
    def test_load_results_failure(self):
        """Test failed results loading"""
        with pytest.raises(ValueError, match="Error loading results file"):
            self_reviewer.load_results('/nonexistent/file.json')
    
    def test_group_by_category(self):
        """Test grouping results by category"""
        test_results = {
            'img1.jpg': {'category': 'cat1', 'confidence': 0.8},
            'img2.jpg': {'category': 'cat1', 'confidence': 0.9},
            'img3.jpg': {'category': 'cat2', 'confidence': 0.7},
            'img4.jpg': {'category': 'cat1', 'confidence': 0.6}
        }
        
        grouped = self_reviewer.group_by_category(test_results)
        
        assert len(grouped) == 2
        assert len(grouped['cat1']) == 3
        assert len(grouped['cat2']) == 1
    
    def test_find_confidence_inconsistencies(self):
        """Test finding confidence inconsistencies"""
        images = [
            ('img1.jpg', {'confidence': 0.9}),
            ('img2.jpg', {'confidence': 0.8}),
            ('img3.jpg', {'confidence': 0.4}),  # Should be flagged
            ('img4.jpg', {'confidence': 0.85})
        ]
        
        inconsistencies = self_reviewer.find_confidence_inconsistencies(images, threshold=0.3)
        
        # Should find inconsistencies between high and low confidence images
        assert len(inconsistencies) > 0
        # Check that img3 is involved in inconsistencies
        assert any('img3.jpg' in (inc[0], inc[1]) for inc in inconsistencies)
    
    def test_find_statistical_outliers(self):
        """Test finding statistical outliers"""
        images = [
            ('img1.jpg', {'confidence': 0.8}),
            ('img2.jpg', {'confidence': 0.82}),
            ('img3.jpg', {'confidence': 0.81}),
            ('img4.jpg', {'confidence': 0.15})  # Clear outlier
        ]
        
        outliers = self_reviewer.find_statistical_outliers(images, std_threshold=1.5)
        
        assert len(outliers) > 0
        # Check that the outlier is detected
        assert any('img4.jpg' in str(outlier) for outlier in outliers)
    
    def test_find_statistical_outliers_insufficient_data(self):
        """Test outlier detection with insufficient data"""
        images = [
            ('img1.jpg', {'confidence': 0.8}),
            ('img2.jpg', {'confidence': 0.9})
        ]
        
        outliers = self_reviewer.find_statistical_outliers(images)
        assert len(outliers) == 0  # Not enough data for meaningful statistics
    
    def test_calculate_review_statistics(self):
        """Test review statistics calculation"""
        test_results = {
            'img1.jpg': {'category': 'cat1', 'confidence': 0.8},
            'img2.jpg': {'category': 'cat1', 'confidence': 0.9},
            'img3.jpg': {'category': 'cat2', 'confidence': 0.7},
            'img4.jpg': {'category': 'error', 'confidence': 0.0}
        }
        
        stats = self_reviewer.calculate_review_statistics(test_results)
        
        assert stats['total_images'] == 4
        assert stats['categories']['cat1'] == 2
        assert stats['categories']['cat2'] == 1
        assert stats['categories']['error'] == 1
        assert stats['error_count'] == 1
        assert 'confidence_stats' in stats
        assert stats['confidence_stats']['mean'] > 0
    
    def test_save_reviewed_results(self):
        """Test saving reviewed results"""
        test_results = {
            'img1.jpg': {'category': 'cat1', 'confidence': 0.8}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            original_file = tmp.name
        
        try:
            output_file = self_reviewer.save_reviewed_results(test_results, original_file)
            
            # Check that file was created with correct name
            assert output_file.endswith('_reviewed.json')
            assert os.path.exists(output_file)
            
            # Check content
            with open(output_file, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == test_results
            
        finally:
            if os.path.exists(original_file):
                os.unlink(original_file)
            if os.path.exists(output_file):
                os.unlink(output_file)


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from pixel sampling to analysis"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test directory structure
            ref_dir = os.path.join(tmp_dir, 'references')
            cat1_dir = os.path.join(ref_dir, 'red')
            cat2_dir = os.path.join(ref_dir, 'blue')
            os.makedirs(cat1_dir)
            os.makedirs(cat2_dir)
            
            # Create test images
            red_img = Image.new('RGB', (50, 50), color='red')
            blue_img = Image.new('RGB', (50, 50), color='blue')
            red_img.save(os.path.join(cat1_dir, 'red1.jpg'))
            blue_img.save(os.path.join(cat2_dir, 'blue1.jpg'))
            
            # Step 1: Build pixel database
            pixel_db = pixel_sampler.build_pixel_database(ref_dir, sample_size=10)
            assert len(pixel_db) == 2
            assert 'red' in pixel_db
            assert 'blue' in pixel_db
            
            # Step 2: Create test image for analysis
            test_img = Image.new('RGB', (30, 30), color='red')
            test_img_path = os.path.join(tmp_dir, 'test_red.jpg')
            test_img.save(test_img_path)
            
            # Step 3: Analyze image
            weights = {'red': 1.0, 'blue': 1.0}
            category, scores, confidence = correlation_analyzer.analyze_image(
                test_img_path, pixel_db, weights
            )
            
            # Should classify as red
            assert category == 'red'
            assert confidence > 0.5
            
            # Step 4: Test batch processing
            batch_dir = os.path.join(tmp_dir, 'batch')
            os.makedirs(batch_dir)
            
            # Copy test image to batch directory
            shutil.copy2(test_img_path, os.path.join(batch_dir, 'batch_test.jpg'))
            
            results = batch_processor.process_batch(batch_dir, pixel_db, weights)
            assert len(results) == 1
            assert 'batch_test.jpg' in results
            assert results['batch_test.jpg']['category'] == 'red'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
