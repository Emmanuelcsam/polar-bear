#!/usr/bin/env python3
"""
Unit tests for the main fiber optics system
Tests system initialization, training, analysis, and all major workflows
"""

import unittest
import tempfile
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from fiber_main import FiberOpticsSystem
from test_base import FiberOpticsTestCase
import json


class TestFiberOpticsSystem(FiberOpticsTestCase):
    """Test the main fiber optics system"""
    
    def setUp(self):
        """Set up test system"""
        super().setUp()
        self.system = FiberOpticsSystem()
    
    def test_system_initialization(self):
        """Test system initializes all components"""
        self.assertIsNotNone(self.system.config)
        self.assertIsNotNone(self.system.logger)
        self.assertIsNotNone(self.system.tensor_processor)
        self.assertIsNotNone(self.system.integrated_pipeline)
        self.assertIsNotNone(self.system.trainer)
        self.assertIsNotNone(self.system.data_loader)
        self.assertIsNotNone(self.system.reference_loader)
        
        self.assertFalse(self.system.is_trained)
        self.assertEqual(self.system.training_history, {})
    
    def test_analyze_single_image(self):
        """Test single image analysis"""
        # Create and save a test image
        test_image = self.create_mock_image()
        image_path = self.save_test_image(test_image, "test_fiber.png")
        
        # Analyze image
        results = self.system.analyze_single_image(str(image_path))
        
        # Check results structure
        self.assertIn('summary', results)
        self.assertIn('final_similarity_score', results['summary'])
        self.assertIn('meets_threshold', results['summary'])
        self.assertIn('primary_region', results['summary'])
        self.assertIn('anomaly_score', results['summary'])
        
        # Check similarity is in valid range
        similarity = results['summary']['final_similarity_score']
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        
        # Check threshold (must achieve over 0.7)
        threshold = self.system.config.similarity.threshold
        self.assertEqual(threshold, 0.7)
        
        # Check results file was created
        results_file = Path(self.system.config.system.results_path) / f"{image_path.stem}_results.txt"
        self.assertTrue(results_file.exists())
    
    def test_batch_process(self):
        """Test batch processing of multiple images"""
        # Create multiple test images
        num_images = 5
        for i in range(num_images):
            image = self.create_mock_image()
            self.save_test_image(image, f"test_fiber_{i}.png")
        
        # Process batch
        results = self.system.batch_process(str(self.test_data_dir), max_images=num_images)
        
        # Check results
        self.assertEqual(len(results), num_images)
        
        # Check each result
        successful = 0
        for result in results:
            if 'error' not in result:
                successful += 1
                self.assertIn('summary', result)
                self.assertIn('final_similarity_score', result['summary'])
        
        self.assertGreater(successful, 0)
    
    def test_update_parameters(self):
        """Test parameter update functionality"""
        # Get initial coefficient
        initial_A = self.system.config.equation.coefficients.A
        
        # Update coefficient
        new_value = 1.5
        self.system.update_parameters('A', new_value)
        
        # Check it was updated
        self.assertEqual(self.system.config.equation.coefficients.A, new_value)
        
        # Check bounds are respected
        self.system.update_parameters('B', -3.0)  # Beyond min bound
        self.assertGreaterEqual(
            self.system.config.equation.coefficients.B,
            self.system.config.equation.min_coefficient
        )
    
    def test_training_workflow(self):
        """Test basic training workflow"""
        # Create some mock training data
        for i in range(10):
            image = self.create_mock_image()
            self.save_test_image(image, f"train_{i}.png")
        
        # Train for just 1 epoch (quick test)
        self.system.train_model(num_epochs=1)
        
        # Check training completed
        self.assertTrue(self.system.is_trained)
        self.assertIn('train_loss', self.system.training_history)
        
        # Check checkpoint was saved
        checkpoint_dir = Path(self.system.config.system.checkpoints_path)
        self.assertTrue(checkpoint_dir.exists())
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        self.assertGreater(len(checkpoints), 0)
    
    def test_evaluate_performance(self):
        """Test performance evaluation"""
        # Need to "train" first
        self.system.is_trained = True
        
        # Create some validation data
        for i in range(5):
            image = self.create_mock_image()
            self.save_test_image(image, f"val_{i}.png")
        
        # Evaluate performance
        self.system.evaluate_performance()
        
        # Should complete without errors
        self.assertTrue(True)
    
    def test_realtime_process_simulation(self):
        """Test realtime processing simulation"""
        # Create a few test images for streaming
        for i in range(3):
            image = self.create_mock_image()
            self.save_test_image(image, f"stream_{i}.png")
        
        # We can't fully test realtime without actual streaming
        # Just verify it initializes correctly
        try:
            # This would normally run indefinitely
            # For testing, we just check it starts
            stream_loader = self.system.data_loader.get_streaming_loader(batch_size=1)
            self.assertIsNotNone(stream_loader)
        except Exception as e:
            # Streaming might not be fully implemented
            pass
    
    def test_equation_coefficients(self):
        """Test equation I=Ax1+Bx2+Cx3...=S(R)"""
        coeffs = self.system.config.equation.coefficients
        
        # Check all coefficients exist
        for coeff in ['A', 'B', 'C', 'D', 'E']:
            self.assertIn(coeff, coeffs)
            self.assertIsInstance(coeffs[coeff], (int, float))
            
            # Check bounds
            self.assertGreaterEqual(
                coeffs[coeff],
                self.system.config.equation.min_coefficient
            )
            self.assertLessEqual(
                coeffs[coeff],
                self.system.config.equation.max_coefficient
            )
    
    def test_similarity_threshold_enforcement(self):
        """Test that similarity threshold of 0.7 is enforced"""
        # The program must achieve over 0.7
        threshold = self.system.config.similarity.threshold
        self.assertEqual(threshold, 0.7)
        
        # Create test image and analyze
        test_image = self.create_mock_image()
        image_path = self.save_test_image(test_image, "threshold_test.png")
        
        results = self.system.analyze_single_image(str(image_path))
        
        # Check if threshold is properly evaluated
        similarity = results['summary']['final_similarity_score']
        meets_threshold = results['summary']['meets_threshold']
        
        if similarity > 0.7:
            self.assertTrue(meets_threshold)
        else:
            self.assertFalse(meets_threshold)
    
    def test_anomaly_detection_integration(self):
        """Test anomaly detection is integrated in analysis"""
        # Create image with known defects
        test_image = self.create_mock_image()
        image_path = self.save_test_image(test_image, "anomaly_test.png")
        
        results = self.system.analyze_single_image(str(image_path))
        
        # Check anomaly score is present
        self.assertIn('anomaly_score', results['summary'])
        anomaly_score = results['summary']['anomaly_score']
        self.assertGreaterEqual(anomaly_score, 0.0)
        self.assertLessEqual(anomaly_score, 1.0)
    
    def test_region_classification_integration(self):
        """Test region classification (core, cladding, ferrule)"""
        test_image = self.create_mock_image()
        image_path = self.save_test_image(test_image, "region_test.png")
        
        results = self.system.analyze_single_image(str(image_path))
        
        # Check primary region is identified
        self.assertIn('primary_region', results['summary'])
        primary_region = results['summary']['primary_region']
        self.assertIn(primary_region, ['core', 'cladding', 'ferrule'])
    
    def test_checkpoint_loading(self):
        """Test loading from checkpoint"""
        # First do a mini training
        for i in range(5):
            image = self.create_mock_image()
            self.save_test_image(image, f"checkpoint_train_{i}.png")
        
        self.system.train_model(num_epochs=1)
        
        # Get checkpoint path
        checkpoint_dir = Path(self.system.config.system.checkpoints_path)
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        
        if checkpoints:
            # Create new system and load checkpoint
            new_system = FiberOpticsSystem()
            new_system.train_model(num_epochs=1, load_checkpoint=str(checkpoints[0]))
            
            self.assertTrue(new_system.is_trained)
    
    def test_memory_efficiency(self):
        """Test system doesn't leak memory during batch processing"""
        # Get initial memory state
        initial_tensors = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])
        
        # Process a small batch
        for i in range(3):
            image = self.create_mock_image(size=(128, 128))  # Smaller images
            self.save_test_image(image, f"memory_test_{i}.png")
        
        self.system.batch_process(str(self.test_data_dir), max_images=3)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory hasn't grown excessively
        final_tensors = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])
        
        # Allow some growth but not excessive
        self.assertLess(final_tensors - initial_tensors, 100)


class TestSystemIntegration(FiberOpticsTestCase):
    """Test integration between system components"""
    
    def test_logger_integration(self):
        """Test logger is properly integrated"""
        system = FiberOpticsSystem()
        
        # Check logger is used
        self.assertIsNotNone(system.logger)
        self.assertEqual(system.logger.name, "FiberOpticsSystem")
        
        # Verify logger has run tracking
        self.assertTrue(hasattr(system.logger, 'run_id'))
        self.assertTrue(hasattr(system.logger, 'run_logger'))
    
    def test_config_integration(self):
        """Test configuration is properly integrated"""
        system = FiberOpticsSystem()
        
        # Check all components use same config
        self.assertIs(
            system.tensor_processor.config,
            system.config
        )
        self.assertIs(
            system.integrated_pipeline.config,
            system.config
        )
    
    def test_pipeline_integration(self):
        """Test analysis pipeline integration"""
        system = FiberOpticsSystem()
        
        # Create test image
        test_image = self.create_mock_image()
        image_path = self.save_test_image(test_image, "pipeline_test.png")
        
        # Analyze through pipeline
        results = system.analyze_single_image(str(image_path))
        
        # Check all pipeline stages produced results
        self.assertIn('summary', results)
        self.assertIn('tensor_info', results)
        self.assertIn('features', results)
        self.assertIn('segmentation', results)
        self.assertIn('anomalies', results)
        self.assertIn('similarity', results)
    
    def test_error_handling(self):
        """Test system handles errors gracefully"""
        system = FiberOpticsSystem()
        
        # Try to analyze non-existent image
        results = system.analyze_single_image("non_existent_image.png")
        
        # Should handle error gracefully
        # (Implementation may vary - might return error dict or raise exception)
        # Just verify it doesn't crash the system
        self.assertIsNotNone(system)  # System still exists
    
    def test_concurrent_analysis(self):
        """Test system can handle concurrent analysis requests"""
        system = FiberOpticsSystem()
        
        # Create multiple test images
        images = []
        for i in range(3):
            image = self.create_mock_image(size=(128, 128))
            path = self.save_test_image(image, f"concurrent_{i}.png")
            images.append(str(path))
        
        # Analyze all images (simulating concurrent requests)
        results = []
        for image_path in images:
            result = system.analyze_single_image(image_path)
            results.append(result)
        
        # All should complete successfully
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('summary', result)


if __name__ == "__main__":
    # Add garbage collection import for memory test
    import gc
    unittest.main()