#!/usr/bin/env python3
"""
Comprehensive Integration Test Script
Tests the complete fiber analysis pipeline with real workflow scenarios.
"""

import os
import sys
import json
import logging
import tempfile
import shutil
from pathlib import Path
import traceback
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class IntegrationTester:
    """Comprehensive integration testing for the fiber analysis pipeline"""
    
    def __init__(self):
        self.test_dir = Path("test_images")
        self.config_path = Path("config.json")
        self.temp_results_dir = None
        
    def setup_environment(self):
        """Setup test environment"""
        logging.info("Setting up integration test environment...")
        
        # Create temporary results directory
        self.temp_results_dir = Path(tempfile.mkdtemp(prefix="fiber_integration_test_"))
        
        # Ensure test images exist
        if not self.test_dir.exists() or not any(self.test_dir.glob("*.png")):
            logging.info("Creating test images...")
            self.create_test_images()
            
        logging.info(f"Integration test environment ready. Temp dir: {self.temp_results_dir}")
        
    def create_test_images(self):
        """Create test images if they don't exist"""
        try:
            from create_test_images import main as create_images
            create_images()
            logging.info("Test images created successfully")
        except Exception as e:
            logging.error(f"Failed to create test images: {e}")
            raise
    
    def test_individual_modules(self):
        """Test each module individually"""
        results = {}
        
        # Test process module
        logging.info("Testing process module individually...")
        try:
            from process import reimagine_image
            test_image = next(self.test_dir.glob("*.png"))
            output_dir = self.temp_results_dir / "process_test"
            output_dir.mkdir(exist_ok=True)
            
            # Test with save_intermediate=True
            images_dict = reimagine_image(str(test_image), str(output_dir), save_intermediate=True)
            
            # Check results
            if output_dir.exists() and any(output_dir.glob("*.jpg")):
                results['process'] = {'status': 'PASS', 'details': f'Generated images in {output_dir}'}
            else:
                results['process'] = {'status': 'FAIL', 'details': 'No images generated'}
                
        except Exception as e:
            results['process'] = {'status': 'FAIL', 'error': str(e)}
            
        # Test separation module  
        logging.info("Testing separation module individually...")
        try:
            from separation import UnifiedSegmentationSystem
            
            segmenter = UnifiedSegmentationSystem()
            test_image = next(self.test_dir.glob("*.png"))
            output_dir = self.temp_results_dir / "separation_test"
            output_dir.mkdir(exist_ok=True)
            
            result = segmenter.process_image(test_image, str(output_dir))
            
            if result and result.get('saved_regions'):
                results['separation'] = {'status': 'PASS', 'details': f'Generated {len(result["saved_regions"])} regions'}
            else:
                results['separation'] = {'status': 'FAIL', 'details': 'No regions generated'}
                
        except Exception as e:
            results['separation'] = {'status': 'FAIL', 'error': str(e)}
            
        # Test detection module
        logging.info("Testing detection module individually...")
        try:
            from detection import OmniFiberAnalyzer, OmniConfig
            
            config = OmniConfig()
            analyzer = OmniFiberAnalyzer(config)
            test_image = next(self.test_dir.glob("*.png"))
            output_dir = self.temp_results_dir / "detection_test"
            output_dir.mkdir(exist_ok=True)
            
            result = analyzer.analyze_end_face(str(test_image), str(output_dir))
            
            if result and result.get('success', True):
                results['detection'] = {'status': 'PASS', 'details': f'Analysis completed with {len(result.get("defects", []))} defects found'}
            else:
                results['detection'] = {'status': 'FAIL', 'details': 'Detection failed'}
                
        except Exception as e:
            results['detection'] = {'status': 'FAIL', 'error': str(e)}
            
        return results
    
    def test_full_pipeline(self):
        """Test the complete pipeline"""
        logging.info("Testing full pipeline...")
        
        try:
            from app import PipelineOrchestrator
            
            # Update config to use temp directory
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Modify config for testing
            config['paths']['results_dir'] = str(self.temp_results_dir / "full_pipeline_results")
            
            # Write temporary config
            temp_config_path = self.temp_results_dir / "test_config.json"
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Initialize orchestrator
            orchestrator = PipelineOrchestrator(str(temp_config_path))
            
            # Test with a sample image
            test_image = next(self.test_dir.glob("*.png"))
            
            # Run full pipeline
            start_time = time.time()
            result = orchestrator.run_full_pipeline(test_image)
            end_time = time.time()
            
            # Check results
            results_dir = Path(config['paths']['results_dir'])
            if results_dir.exists():
                # Count generated files
                all_files = list(results_dir.rglob("*"))
                image_files = [f for f in all_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
                json_files = [f for f in all_files if f.suffix == '.json']
                
                return {
                    'status': 'PASS',
                    'details': {
                        'execution_time': f"{end_time - start_time:.2f} seconds",
                        'total_files': len(all_files),
                        'image_files': len(image_files),
                        'json_files': len(json_files),
                        'results_dir': str(results_dir)
                    }
                }
            else:
                return {'status': 'FAIL', 'details': 'No results directory created'}
                
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e), 'traceback': traceback.format_exc()}
    
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        logging.info("Testing error handling...")
        results = {}
        
        # Test with non-existent image
        try:
            from app import PipelineOrchestrator
            orchestrator = PipelineOrchestrator(str(self.config_path))
            result = orchestrator.run_full_pipeline(Path("nonexistent_image.png"))
            results['nonexistent_image'] = {'status': 'UNEXPECTED_SUCCESS', 'details': 'Should have failed'}
        except Exception as e:
            results['nonexistent_image'] = {'status': 'PASS', 'details': 'Correctly handled non-existent image'}
        
        return results
    
    def run_performance_test(self):
        """Run performance benchmarks"""
        logging.info("Running performance tests...")
        
        try:
            from app import PipelineOrchestrator
            
            # Test with multiple images
            test_images = list(self.test_dir.glob("*.png"))[:2]  # Test with 2 images max
            
            orchestrator = PipelineOrchestrator(str(self.config_path))
            
            times = []
            for i, test_image in enumerate(test_images):
                logging.info(f"Performance test {i+1}/{len(test_images)}: {test_image.name}")
                
                start_time = time.time()
                result = orchestrator.run_full_pipeline(test_image)
                end_time = time.time()
                
                execution_time = end_time - start_time
                times.append(execution_time)
                logging.info(f"Execution time: {execution_time:.2f} seconds")
            
            return {
                'status': 'PASS',
                'details': {
                    'images_processed': len(test_images),
                    'total_time': f"{sum(times):.2f} seconds",
                    'average_time': f"{sum(times)/len(times):.2f} seconds",
                    'individual_times': [f"{t:.2f}s" for t in times]
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def cleanup(self):
        """Clean up test environment"""
        if self.temp_results_dir and self.temp_results_dir.exists():
            shutil.rmtree(self.temp_results_dir)
            logging.info("Cleaned up temporary files")
    
    def run_all_tests(self):
        """Run complete integration test suite"""
        logging.info("="*80)
        logging.info("STARTING COMPREHENSIVE INTEGRATION TESTS")
        logging.info("="*80)
        
        try:
            self.setup_environment()
            
            # Run all test categories
            individual_results = self.test_individual_modules()
            full_pipeline_result = self.test_full_pipeline()
            error_handling_results = self.test_error_handling()
            performance_result = self.run_performance_test()
            
            # Generate comprehensive report
            self.generate_final_report(
                individual_results,
                full_pipeline_result,
                error_handling_results,
                performance_result
            )
            
        except Exception as e:
            logging.error(f"Integration test suite failed: {e}")
            logging.error(traceback.format_exc())
            
        finally:
            self.cleanup()
    
    def generate_final_report(self, individual_results, full_pipeline_result, 
                            error_handling_results, performance_result):
        """Generate comprehensive test report"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE INTEGRATION TEST REPORT")
        print("="*80)
        
        print("\n📋 INDIVIDUAL MODULE TESTS:")
        for module, result in individual_results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"{status_icon} {module.upper()}: {result['status']}")
            if 'details' in result:
                print(f"   Details: {result['details']}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
        
        print("\n🔧 FULL PIPELINE TEST:")
        status_icon = "✅" if full_pipeline_result['status'] == 'PASS' else "❌"
        print(f"{status_icon} FULL_PIPELINE: {full_pipeline_result['status']}")
        if 'details' in full_pipeline_result:
            details = full_pipeline_result['details']
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   Details: {details}")
        if 'error' in full_pipeline_result:
            print(f"   Error: {full_pipeline_result['error']}")
        
        print("\n🛡️ ERROR HANDLING TESTS:")
        for test, result in error_handling_results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"{status_icon} {test}: {result['status']}")
            if 'details' in result:
                print(f"   Details: {result['details']}")
        
        print("\n⚡ PERFORMANCE TESTS:")
        status_icon = "✅" if performance_result['status'] == 'PASS' else "❌"
        print(f"{status_icon} PERFORMANCE: {performance_result['status']}")
        if 'details' in performance_result:
            details = performance_result['details']
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"   {key}: {value}")
        
        # Overall summary
        total_tests = (len(individual_results) + 1 + len(error_handling_results) + 1)
        passed_tests = (
            sum(1 for r in individual_results.values() if r['status'] == 'PASS') +
            (1 if full_pipeline_result['status'] == 'PASS' else 0) +
            sum(1 for r in error_handling_results.values() if r['status'] == 'PASS') +
            (1 if performance_result['status'] == 'PASS' else 0)
        )
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! The fiber analysis pipeline is fully functional.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Review the details above.")
        
        print("="*80)

def main():
    """Main entry point"""
    tester = IntegrationTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
