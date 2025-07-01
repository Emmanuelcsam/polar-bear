#!/usr/bin/env python3
"""
Comprehensive test suite for the current-process fiber optic analysis pipeline.
Tests each component individually and the full pipeline integration.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
import logging
import traceback
import tempfile
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(funcName)s - %(message)s'
)

# Add current directory to path for imports
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

class TestRunner:
    """Comprehensive test runner for the fiber analysis pipeline"""
    
    def __init__(self):
        self.test_results = {}
        self.test_dir = Path("test_images")
        self.config_path = Path("config.json")
        self.temp_results_dir = None
        
    def setup_test_environment(self):
        """Setup test environment and create temporary directories"""
        logging.info("Setting up test environment...")
        
        # Create temporary results directory
        self.temp_results_dir = Path(tempfile.mkdtemp(prefix="fiber_test_"))
        
        # Ensure test images exist
        if not self.test_dir.exists() or not any(self.test_dir.glob("*.png")):
            logging.warning("Test images not found, creating them...")
            self.create_test_images()
            
        logging.info(f"Test environment ready. Temp dir: {self.temp_results_dir}")
        return True
        
    def create_test_images(self):
        """Create test images if they don't exist"""
        try:
            from create_test_images import main as create_images
            create_images()
            logging.info("Test images created successfully")
        except Exception as e:
            logging.error(f"Failed to create test images: {e}")
            raise
    
    def test_process_module(self):
        """Test the process.py module"""
        logging.info("Testing process module...")
        try:
            from process import reimagine_image
            
            # Test with a sample image
            test_image = next(self.test_dir.glob("*.png"))
            output_dir = self.temp_results_dir / "process_test"
            
            # Test with save_intermediate=False (RAM only)
            images_dict = reimagine_image(
                str(test_image), 
                str(output_dir), 
                save_intermediate=False
            )
            
            # Check if we got results
            if not images_dict:
                raise ValueError("No images returned from process module")
                
            # Test with save_intermediate=True
            images_dict = reimagine_image(
                str(test_image), 
                str(output_dir), 
                save_intermediate=True
            )
            
            # Check if files were saved
            if not output_dir.exists() or not any(output_dir.glob("*.jpg")):
                raise ValueError("No output files created by process module")
                
            self.test_results['process'] = {'status': 'PASS', 'details': f'Generated {len(images_dict)} image variations'}
            logging.info("Process module test PASSED")
            
        except Exception as e:
            self.test_results['process'] = {'status': 'FAIL', 'error': str(e), 'traceback': traceback.format_exc()}
            logging.error(f"Process module test FAILED: {e}")
    
    def test_separation_module(self):
        """Test the separation.py module"""
        logging.info("Testing separation module...")
        try:
            from separation import UnifiedSegmentationSystem
            
            # Test initialization
            segmenter = UnifiedSegmentationSystem()
            
            # Test with a sample image
            test_image = next(self.test_dir.glob("*.png"))
            output_dir = self.temp_results_dir / "separation_test"
            output_dir.mkdir(exist_ok=True)
            
            # Load test image
            img = cv2.imread(str(test_image))
            if img is None:
                raise ValueError(f"Could not load test image: {test_image}")
            
            # Test segmentation using the correct method
            result = segmenter.process_image(
                test_image, 
                str(output_dir)
            )
            
            if result is None:
                raise ValueError("Segmentation returned None")
                
            self.test_results['separation'] = {'status': 'PASS', 'details': 'Segmentation completed successfully'}
            logging.info("Separation module test PASSED")
            
        except Exception as e:
            self.test_results['separation'] = {'status': 'FAIL', 'error': str(e), 'traceback': traceback.format_exc()}
            logging.error(f"Separation module test FAILED: {e}")
    
    def test_detection_module(self):
        """Test the detection.py module"""
        logging.info("Testing detection module...")
        try:
            from detection import OmniFiberAnalyzer, OmniConfig
            
            # Test configuration
            config = OmniConfig()
            
            # Test analyzer initialization
            analyzer = OmniFiberAnalyzer(config)
            
            # Test with a sample image
            test_image = next(self.test_dir.glob("*.png"))
            output_dir = self.temp_results_dir / "detection_test"
            output_dir.mkdir(exist_ok=True)
            
            # Test detection
            results = analyzer.analyze_fiber_defects(str(test_image), str(output_dir))
            
            if results is None:
                raise ValueError("Detection returned None")
                
            self.test_results['detection'] = {'status': 'PASS', 'details': f'Detection completed, found {len(results.get("defects", []))} potential defects'}
            logging.info("Detection module test PASSED")
            
        except Exception as e:
            self.test_results['detection'] = {'status': 'FAIL', 'error': str(e), 'traceback': traceback.format_exc()}
            logging.error(f"Detection module test FAILED: {e}")
    
    def test_data_acquisition_module(self):
        """Test the data_acquisition.py module"""
        logging.info("Testing data acquisition module...")
        try:
            from data_acquisition import integrate_with_pipeline
            
            # Create mock detection results for testing
            test_image = next(self.test_dir.glob("*.png"))
            results_dir = self.temp_results_dir / "data_acquisition_test"
            results_dir.mkdir(exist_ok=True)
            
            # Create a minimal detection result file for testing
            mock_detection_result = {
                "image_path": str(test_image),
                "defects": [
                    {
                        "id": 1,
                        "center": [100, 100],
                        "area": 50,
                        "confidence": 0.8,
                        "severity": "MEDIUM"
                    }
                ],
                "total_defects": 1,
                "analysis_timestamp": "2025-07-01T16:00:00"
            }
            
            detection_result_file = results_dir / "detection_results.json"
            with open(detection_result_file, 'w') as f:
                json.dump(mock_detection_result, f, indent=2)
            
            # Test data acquisition
            config = {"data_acquisition_settings": {"clustering_eps": 30.0, "min_cluster_size": 1}}
            result = integrate_with_pipeline(str(results_dir), str(test_image), config)
            
            if result is None:
                raise ValueError("Data acquisition returned None")
                
            self.test_results['data_acquisition'] = {'status': 'PASS', 'details': 'Data acquisition completed successfully'}
            logging.info("Data acquisition module test PASSED")
            
        except Exception as e:
            self.test_results['data_acquisition'] = {'status': 'FAIL', 'error': str(e), 'traceback': traceback.format_exc()}
            logging.error(f"Data acquisition module test FAILED: {e}")
    
    def test_full_pipeline(self):
        """Test the complete pipeline integration"""
        logging.info("Testing full pipeline integration...")
        try:
            from app import PipelineOrchestrator
            
            # Test configuration loading
            if not self.config_path.exists():
                raise ValueError("config.json not found")
            
            # Initialize orchestrator
            orchestrator = PipelineOrchestrator(str(self.config_path))
            
            # Update config to use our temporary directory
            orchestrator.config['paths']['results_dir'] = str(self.temp_results_dir / "full_pipeline")
            orchestrator.results_base_dir = Path(orchestrator.config['paths']['results_dir'])
            orchestrator.results_base_dir.mkdir(parents=True, exist_ok=True)
            
            # Test with a sample image
            test_image = next(self.test_dir.glob("*.png"))
            
            # Run full pipeline
            result = orchestrator.run_full_pipeline(test_image)
            
            if result is None:
                raise ValueError("Full pipeline returned None")
                
            # Check if results were created
            if not orchestrator.results_base_dir.exists():
                raise ValueError("Results directory was not created")
                
            self.test_results['full_pipeline'] = {'status': 'PASS', 'details': 'Full pipeline completed successfully'}
            logging.info("Full pipeline test PASSED")
            
        except Exception as e:
            self.test_results['full_pipeline'] = {'status': 'FAIL', 'error': str(e), 'traceback': traceback.format_exc()}
            logging.error(f"Full pipeline test FAILED: {e}")
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        logging.info("Starting comprehensive test suite...")
        
        try:
            self.setup_test_environment()
            
            # Run individual module tests
            self.test_process_module()
            self.test_separation_module()
            self.test_detection_module()
            self.test_data_acquisition_module()
            
            # Run full integration test
            self.test_full_pipeline()
            
        except Exception as e:
            logging.error(f"Test setup failed: {e}")
            self.test_results['setup'] = {'status': 'FAIL', 'error': str(e)}
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_results_dir and self.temp_results_dir.exists():
            try:
                shutil.rmtree(self.temp_results_dir)
                logging.info("Cleaned up temporary files")
            except Exception as e:
                logging.warning(f"Failed to cleanup temp directory: {e}")
    
    def print_test_summary(self):
        """Print a summary of all test results"""
        print("\n" + "="*60)
        print("FIBER ANALYSIS PIPELINE TEST SUMMARY")
        print("="*60)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.test_results.items():
            status = result['status']
            if status == 'PASS':
                passed += 1
                print(f"✓ {test_name.upper()}: PASSED - {result.get('details', '')}")
            else:
                failed += 1
                print(f"✗ {test_name.upper()}: FAILED - {result.get('error', '')}")
        
        print("-"*60)
        print(f"TOTAL: {passed + failed} tests, {passed} passed, {failed} failed")
        
        if failed > 0:
            print("\nFAILED TEST DETAILS:")
            print("-"*60)
            for test_name, result in self.test_results.items():
                if result['status'] == 'FAIL':
                    print(f"\n{test_name.upper()} ERROR:")
                    print(f"Error: {result.get('error', 'Unknown error')}")
                    if 'traceback' in result:
                        print("Traceback:")
                        print(result['traceback'])
        
        print("="*60)
        return failed == 0

def main():
    """Main test runner"""
    test_runner = TestRunner()
    test_runner.run_all_tests()
    success = test_runner.print_test_summary()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
