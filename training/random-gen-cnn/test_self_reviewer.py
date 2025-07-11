"""
Unit tests for self_reviewer_refactored module
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
import tempfile
import json
import numpy as np
from unittest.mock import patch, MagicMock
import self_reviewer_refactored as self_reviewer


class TestSelfReviewer(unittest.TestCase):
    """Test cases for self_reviewer_refactored module"""
    
    def test_load_results_success(self):
        """Test successful results loading"""
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
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(test_results, tmp)
            tmp_path = tmp.name
        
        try:
            loaded_results = self_reviewer.load_results(tmp_path)
            self.assertEqual(loaded_results, test_results)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_results_failure(self):
        """Test failed results loading"""
        with self.assertRaises(ValueError) as context:
            self_reviewer.load_results('/nonexistent/file.json')
        self.assertIn("Error loading results file", str(context.exception))
    
    def test_load_results_invalid_json(self):
        """Test loading results with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write("invalid json content")
            tmp_path = tmp.name
        
        try:
            with self.assertRaises(ValueError) as context:
                self_reviewer.load_results(tmp_path)
            self.assertIn("Error loading results file", str(context.exception))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_group_by_category(self):
        """Test grouping results by category"""
        test_results = {
            'img1.jpg': {'category': 'cat1', 'confidence': 0.8},
            'img2.jpg': {'category': 'cat1', 'confidence': 0.9},
            'img3.jpg': {'category': 'cat2', 'confidence': 0.7},
            'img4.jpg': {'category': 'cat1', 'confidence': 0.6},
            'img5.jpg': {'category': 'cat3', 'confidence': 0.85}
        }
        
        grouped = self_reviewer.group_by_category(test_results)
        
        self.assertEqual(len(grouped), 3)
        self.assertEqual(len(grouped['cat1']), 3)
        self.assertEqual(len(grouped['cat2']), 1)
        self.assertEqual(len(grouped['cat3']), 1)
        
        # Check that tuples contain (image_name, data)
        self.assertEqual(grouped['cat1'][0][0], 'img1.jpg')
        self.assertEqual(grouped['cat1'][0][1]['confidence'], 0.8)
    
    def test_group_by_category_missing_category(self):
        """Test grouping with missing category field"""
        test_results = {
            'img1.jpg': {'category': 'cat1', 'confidence': 0.8},
            'img2.jpg': {'confidence': 0.9},  # Missing category
            'img3.jpg': {'category': 'cat2', 'confidence': 0.7}
        }
        
        grouped = self_reviewer.group_by_category(test_results)
        
        self.assertEqual(len(grouped), 3)
        self.assertIn('cat1', grouped)
        self.assertIn('cat2', grouped)
        self.assertIn('unknown', grouped)  # Default for missing category
        self.assertEqual(len(grouped['unknown']), 1)
    
    def test_find_confidence_inconsistencies(self):
        """Test finding confidence inconsistencies"""
        images = [
            ('img1.jpg', {'confidence': 0.9}),
            ('img2.jpg', {'confidence': 0.8}),
            ('img3.jpg', {'confidence': 0.4}),  # Should be flagged as inconsistent
            ('img4.jpg', {'confidence': 0.85}),
            ('img5.jpg', {'confidence': 0.3})   # Should be flagged as inconsistent
        ]
        
        inconsistencies = self_reviewer.find_confidence_inconsistencies(images, threshold=0.3)
        
        self.assertGreater(len(inconsistencies), 0)
        # Check that low confidence images are involved in inconsistencies
        found_img3 = any('img3.jpg' in (inc[0], inc[1]) for inc in inconsistencies)
        found_img5 = any('img5.jpg' in (inc[0], inc[1]) for inc in inconsistencies)
        self.assertTrue(found_img3 or found_img5)
    
    def test_find_confidence_inconsistencies_no_issues(self):
        """Test finding inconsistencies when all confidences are similar"""
        images = [
            ('img1.jpg', {'confidence': 0.85}),
            ('img2.jpg', {'confidence': 0.82}),
            ('img3.jpg', {'confidence': 0.88}),
            ('img4.jpg', {'confidence': 0.84})
        ]
        
        inconsistencies = self_reviewer.find_confidence_inconsistencies(images, threshold=0.3)
        
        # Should find no major inconsistencies
        self.assertEqual(len(inconsistencies), 0)
    
    def test_find_confidence_inconsistencies_missing_confidence(self):
        """Test finding inconsistencies with missing confidence values"""
        images = [
            ('img1.jpg', {'confidence': 0.9}),
            ('img2.jpg', {}),  # Missing confidence
            ('img3.jpg', {'confidence': 0.8})
        ]
        
        # Should not crash and should handle missing values
        inconsistencies = self_reviewer.find_confidence_inconsistencies(images, threshold=0.3)
        
        # Should find inconsistencies due to missing confidence (treated as 0)
        self.assertGreater(len(inconsistencies), 0)
    
    def test_find_statistical_outliers(self):
        """Test finding statistical outliers"""
        images = [
            ('img1.jpg', {'confidence': 0.8}),
            ('img2.jpg', {'confidence': 0.82}),
            ('img3.jpg', {'confidence': 0.81}),
            ('img4.jpg', {'confidence': 0.79}),
            ('img5.jpg', {'confidence': 0.15})  # Clear outlier
        ]
        
        outliers = self_reviewer.find_statistical_outliers(images, std_threshold=1.5)
        
        self.assertGreater(len(outliers), 0)
        # Check that the outlier is detected
        found_outlier = any('img5.jpg' in str(outlier) for outlier in outliers)
        self.assertTrue(found_outlier)
    
    def test_find_statistical_outliers_insufficient_data(self):
        """Test outlier detection with insufficient data"""
        images = [
            ('img1.jpg', {'confidence': 0.8}),
            ('img2.jpg', {'confidence': 0.9})
        ]
        
        outliers = self_reviewer.find_statistical_outliers(images)
        self.assertEqual(len(outliers), 0)  # Not enough data for meaningful statistics
    
    def test_find_statistical_outliers_no_outliers(self):
        """Test outlier detection with no outliers"""
        images = [
            ('img1.jpg', {'confidence': 0.8}),
            ('img2.jpg', {'confidence': 0.82}),
            ('img3.jpg', {'confidence': 0.81}),
            ('img4.jpg', {'confidence': 0.79}),
            ('img5.jpg', {'confidence': 0.83})
        ]
        
        outliers = self_reviewer.find_statistical_outliers(images, std_threshold=2.0)
        self.assertEqual(len(outliers), 0)
    
    def test_review_category_consistency(self):
        """Test reviewing consistency across categories"""
        categorized = {
            'cat1': [
                ('img1.jpg', {'confidence': 0.9}),
                ('img2.jpg', {'confidence': 0.85}),
                ('img3.jpg', {'confidence': 0.4})  # Outlier
            ],
            'cat2': [
                ('img4.jpg', {'confidence': 0.95}),
                ('img5.jpg', {'confidence': 0.2})   # Inconsistent
            ],
            'cat3': [
                ('img6.jpg', {'confidence': 0.8})   # Only one image, should be skipped
            ]
        }
        
        inconsistencies = self_reviewer.review_category_consistency(categorized)
        
        # Should find issues in cat1 and cat2, but not cat3 (only one image)
        self.assertIn('cat1', inconsistencies)
        self.assertIn('cat2', inconsistencies)
        self.assertNotIn('cat3', inconsistencies)
    
    def test_review_category_consistency_empty(self):
        """Test reviewing consistency with empty categorized data"""
        inconsistencies = self_reviewer.review_category_consistency({})
        self.assertEqual(inconsistencies, {})
    
    def test_calculate_review_statistics(self):
        """Test review statistics calculation"""
        test_results = {
            'img1.jpg': {'category': 'cat1', 'confidence': 0.8},
            'img2.jpg': {'category': 'cat1', 'confidence': 0.9},
            'img3.jpg': {'category': 'cat2', 'confidence': 0.7},
            'img4.jpg': {'category': 'error', 'confidence': 0.0},
            'img5.jpg': {'category': 'cat3', 'confidence': 0.95}
        }
        
        stats = self_reviewer.calculate_review_statistics(test_results)
        
        self.assertEqual(stats['total_images'], 5)
        self.assertEqual(stats['categories']['cat1'], 2)
        self.assertEqual(stats['categories']['cat2'], 1)
        self.assertEqual(stats['categories']['cat3'], 1)
        self.assertEqual(stats['categories']['error'], 1)
        self.assertEqual(stats['error_count'], 1)
        
        # Check confidence statistics (should exclude error images)
        self.assertIn('confidence_stats', stats)
        self.assertGreater(stats['confidence_stats']['mean'], 0)
        self.assertGreater(stats['confidence_stats']['std'], 0)
        self.assertEqual(stats['confidence_stats']['min'], 0.7)
        self.assertEqual(stats['confidence_stats']['max'], 0.95)
    
    def test_calculate_review_statistics_empty(self):
        """Test review statistics with empty results"""
        stats = self_reviewer.calculate_review_statistics({})
        
        self.assertEqual(stats['total_images'], 0)
        self.assertEqual(stats['categories'], {})
        self.assertEqual(stats['error_count'], 0)
        self.assertEqual(stats['confidence_stats'], {})
    
    def test_calculate_review_statistics_all_errors(self):
        """Test review statistics with all error images"""
        test_results = {
            'img1.jpg': {'category': 'error', 'confidence': 0.0},
            'img2.jpg': {'category': 'error', 'confidence': 0.0}
        }
        
        stats = self_reviewer.calculate_review_statistics(test_results)
        
        self.assertEqual(stats['total_images'], 2)
        self.assertEqual(stats['error_count'], 2)
        self.assertEqual(stats['confidence_stats'], {})  # No valid confidences
    
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
            self.assertTrue(output_file.endswith('_reviewed.json'))
            self.assertTrue(os.path.exists(output_file))
            
            # Check content
            with open(output_file, 'r') as f:
                loaded_data = json.load(f)
            self.assertEqual(loaded_data, test_results)
            
        finally:
            if os.path.exists(original_file):
                os.unlink(original_file)
            if 'output_file' in locals() and os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_save_reviewed_results_permission_error(self):
        """Test saving reviewed results with permission error"""
        test_results = {'img1.jpg': {'category': 'cat1'}}
        
        with self.assertRaises(ValueError) as context:
            self_reviewer.save_reviewed_results(test_results, '/root/readonly.json')
        self.assertIn("Error saving reviewed results", str(context.exception))
    
    @patch('self_reviewer_refactored.ca.analyze_image')
    def test_re_analyze_suspicious_images(self, mock_analyze):
        """Test re-analyzing suspicious images"""
        # Setup mock
        mock_analyze.return_value = ('new_cat', {'new_cat': 0.9}, 0.9)
        
        inconsistencies = {
            'cat1': [('img1.jpg', 'outlier', 0.3)]
        }
        
        results = {
            'img1.jpg': {
                'category': 'cat1',
                'confidence': 0.3,
                'path': '/path/to/img1.jpg'
            }
        }
        
        pixel_db = {'cat1': [], 'new_cat': []}
        weights = {'cat1': 1.0, 'new_cat': 1.0}
        
        updated_results, changes = self_reviewer.re_analyze_suspicious_images(
            inconsistencies, results, pixel_db, weights
        )
        
        # Should have updated the category
        self.assertEqual(updated_results['img1.jpg']['category'], 'new_cat')
        self.assertEqual(updated_results['img1.jpg']['confidence'], 0.9)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0], ('img1.jpg', 'cat1', 'new_cat'))
    
    @patch('self_reviewer_refactored.ca.analyze_image')
    def test_re_analyze_suspicious_images_no_changes(self, mock_analyze):
        """Test re-analyzing suspicious images with no changes"""
        # Setup mock to return same category
        mock_analyze.return_value = ('cat1', {'cat1': 0.8}, 0.8)
        
        inconsistencies = {
            'cat1': [('img1.jpg', 'outlier', 0.3)]
        }
        
        results = {
            'img1.jpg': {
                'category': 'cat1',
                'confidence': 0.3,
                'path': '/path/to/img1.jpg'
            }
        }
        
        pixel_db = {'cat1': []}
        weights = {'cat1': 1.0}
        
        updated_results, changes = self_reviewer.re_analyze_suspicious_images(
            inconsistencies, results, pixel_db, weights
        )
        
        # Should not have made changes since category is the same
        self.assertEqual(len(changes), 0)
    
    @patch('self_reviewer_refactored.ca.analyze_image')
    def test_re_analyze_suspicious_images_error(self, mock_analyze):
        """Test re-analyzing suspicious images with analysis error"""
        # Setup mock to raise exception
        mock_analyze.side_effect = Exception("Analysis failed")
        
        inconsistencies = {
            'cat1': [('img1.jpg', 'outlier', 0.3)]
        }
        
        results = {
            'img1.jpg': {
                'category': 'cat1',
                'confidence': 0.3,
                'path': '/path/to/img1.jpg'
            }
        }
        
        pixel_db = {'cat1': []}
        weights = {'cat1': 1.0}
        
        # Should not crash and should return no changes
        updated_results, changes = self_reviewer.re_analyze_suspicious_images(
            inconsistencies, results, pixel_db, weights
        )
        
        self.assertEqual(len(changes), 0)
    
    def test_print_review_summary(self):
        """Test printing review summary"""
        inconsistencies = {
            'cat1': [('img1.jpg', 'img2.jpg', 0.4), ('img3.jpg', 'outlier', 0.2)]
        }
        
        changes = [('img1.jpg', 'old_cat', 'new_cat')]
        
        stats = {
            'total_images': 10,
            'error_count': 1,
            'confidence_stats': {
                'mean': 0.75,
                'std': 0.15,
                'min': 0.4,
                'max': 0.95
            },
            'categories': {
                'cat1': 5,
                'cat2': 3,
                'cat3': 2
            }
        }
        
        # Should not raise any exceptions
        self_reviewer.print_review_summary(inconsistencies, changes, stats)


if __name__ == '__main__':
    unittest.main()
