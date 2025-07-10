#!/usr/bin/env python3
"""
Validation script to ensure GPU and CPU modes produce consistent results
"""

import numpy as np
import cv2
import json
import tempfile
import shutil
import os
import time
from pathlib import Path
import logging
from typing import Dict, Tuple, List

# Import pipeline modules
from app_gpu import FiberAnalysisPipelineGPU
from gpu_utils import GPUManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('ValidationScript')


class ConsistencyValidator:
    """Validates consistency between GPU and CPU processing modes"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_images = []
        self.logger = logging.getLogger('ConsistencyValidator')
        
    def __del__(self):
        """Clean up temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_images(self) -> List[str]:
        """Create various test images for validation"""
        test_cases = []
        
        # Test case 1: Perfect fiber
        img1 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img1, (150, 150), 40, (255, 255, 255), -1)  # Core
        cv2.circle(img1, (150, 150), 80, (180, 180, 180), -1)  # Cladding
        cv2.circle(img1, (150, 150), 40, (255, 255, 255), -1)  # Redraw core
        path1 = os.path.join(self.temp_dir, "test_perfect.png")
        cv2.imwrite(path1, img1)
        test_cases.append(("Perfect Fiber", path1))
        
        # Test case 2: Fiber with defects
        img2 = img1.copy()
        cv2.circle(img2, (130, 130), 5, (50, 50, 50), -1)  # Dark defect
        cv2.rectangle(img2, (180, 180), (190, 190), (250, 250, 250), -1)  # Bright defect
        path2 = os.path.join(self.temp_dir, "test_defects.png")
        cv2.imwrite(path2, img2)
        test_cases.append(("Fiber with Defects", path2))
        
        # Test case 3: Off-center fiber
        img3 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img3, (120, 180), 40, (255, 255, 255), -1)  # Core
        cv2.circle(img3, (120, 180), 80, (180, 180, 180), -1)  # Cladding
        cv2.circle(img3, (120, 180), 40, (255, 255, 255), -1)  # Redraw core
        path3 = os.path.join(self.temp_dir, "test_offcenter.png")
        cv2.imwrite(path3, img3)
        test_cases.append(("Off-center Fiber", path3))
        
        # Test case 4: Noisy image
        img4 = img1.copy()
        noise = np.random.normal(0, 20, img4.shape).astype(np.uint8)
        img4 = cv2.add(img4, noise)
        path4 = os.path.join(self.temp_dir, "test_noisy.png")
        cv2.imwrite(path4, img4)
        test_cases.append(("Noisy Fiber", path4))
        
        # Test case 5: Small image
        img5 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img5, (50, 50), 15, (255, 255, 255), -1)  # Core
        cv2.circle(img5, (50, 50), 30, (180, 180, 180), -1)  # Cladding
        cv2.circle(img5, (50, 50), 15, (255, 255, 255), -1)  # Redraw core
        path5 = os.path.join(self.temp_dir, "test_small.png")
        cv2.imwrite(path5, img5)
        test_cases.append(("Small Fiber", path5))
        
        self.test_images = test_cases
        return test_cases
    
    def run_pipeline(self, image_path: str, force_cpu: bool) -> Dict:
        """Run pipeline and return results"""
        output_dir = os.path.join(self.temp_dir, f"output_{'cpu' if force_cpu else 'gpu'}")
        
        try:
            pipeline = FiberAnalysisPipelineGPU(force_cpu=force_cpu)
            result = pipeline.analyze_image(image_path, output_dir)
            
            # Load additional results
            with open(os.path.join(output_dir, "analysis_summary.json"), 'r') as f:
                detailed_summary = json.load(f)
            
            return {
                'success': True,
                'summary': result,
                'detailed': detailed_summary,
                'output_dir': output_dir
            }
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'output_dir': output_dir
            }
    
    def compare_results(self, cpu_result: Dict, gpu_result: Dict) -> Dict[str, any]:
        """Compare CPU and GPU results"""
        comparison = {
            'consistent': True,
            'differences': [],
            'metrics': {}
        }
        
        # Check if both succeeded
        if not (cpu_result['success'] and gpu_result['success']):
            comparison['consistent'] = False
            comparison['differences'].append("One or both pipelines failed")
            return comparison
        
        cpu_summary = cpu_result['summary']
        gpu_summary = gpu_result['summary']
        
        # Compare quality scores
        quality_diff = abs(cpu_summary['quality_score'] - gpu_summary['quality_score'])
        comparison['metrics']['quality_score_diff'] = quality_diff
        
        if quality_diff > 5.0:  # Allow 5% difference
            comparison['consistent'] = False
            comparison['differences'].append(
                f"Quality score difference: {quality_diff:.2f}% "
                f"(CPU: {cpu_summary['quality_score']:.1f}, GPU: {gpu_summary['quality_score']:.1f})"
            )
        
        # Compare defect counts
        defect_diff = abs(cpu_summary['total_defects'] - gpu_summary['total_defects'])
        comparison['metrics']['defect_count_diff'] = defect_diff
        
        if defect_diff > 2:  # Allow difference of 2 defects
            comparison['consistent'] = False
            comparison['differences'].append(
                f"Defect count difference: {defect_diff} "
                f"(CPU: {cpu_summary['total_defects']}, GPU: {gpu_summary['total_defects']})"
            )
        
        # Compare timing (GPU should be faster)
        cpu_time = cpu_summary['processing_time']['total']
        gpu_time = gpu_summary['processing_time']['total']
        comparison['metrics']['speedup'] = cpu_time / gpu_time if gpu_time > 0 else 0
        
        # Compare pass/fail status
        if cpu_summary['pass_fail_status'].split(' - ')[0] != gpu_summary['pass_fail_status'].split(' - ')[0]:
            comparison['consistent'] = False
            comparison['differences'].append(
                f"Pass/fail status mismatch: CPU={cpu_summary['pass_fail_status']}, "
                f"GPU={gpu_summary['pass_fail_status']}"
            )
        
        # Compare detailed results
        cpu_detailed = cpu_result['detailed']
        gpu_detailed = gpu_result['detailed']
        
        # Compare separation results
        if 'separation_metadata' in cpu_detailed and 'separation_metadata' in gpu_detailed:
            cpu_sep = cpu_detailed['separation_metadata']
            gpu_sep = gpu_detailed['separation_metadata']
            
            # Compare center coordinates
            center_dist = np.sqrt(
                (cpu_sep['center'][0] - gpu_sep['center'][0])**2 +
                (cpu_sep['center'][1] - gpu_sep['center'][1])**2
            )
            comparison['metrics']['center_distance'] = center_dist
            
            if center_dist > 5.0:  # Allow 5 pixel difference
                comparison['consistent'] = False
                comparison['differences'].append(f"Center location difference: {center_dist:.2f} pixels")
            
            # Compare radii
            core_diff = abs(cpu_sep['core_radius'] - gpu_sep['core_radius'])
            cladding_diff = abs(cpu_sep['cladding_radius'] - gpu_sep['cladding_radius'])
            
            comparison['metrics']['core_radius_diff'] = core_diff
            comparison['metrics']['cladding_radius_diff'] = cladding_diff
            
            if core_diff > 3.0 or cladding_diff > 3.0:
                comparison['consistent'] = False
                comparison['differences'].append(
                    f"Radius differences - Core: {core_diff:.2f}, Cladding: {cladding_diff:.2f}"
                )
        
        return comparison
    
    def validate_consistency(self) -> Dict[str, any]:
        """Run full consistency validation"""
        self.logger.info("Starting GPU/CPU consistency validation...")
        
        # Check GPU availability
        gpu_manager = GPUManager(force_cpu=False)
        if not gpu_manager.use_gpu:
            self.logger.warning("GPU not available, skipping GPU validation")
            return {
                'gpu_available': False,
                'message': "GPU not available for validation"
            }
        
        # Create test images
        test_cases = self.create_test_images()
        self.logger.info(f"Created {len(test_cases)} test images")
        
        results = {
            'gpu_available': True,
            'test_cases': [],
            'overall_consistency': True,
            'performance_metrics': {
                'average_speedup': 0,
                'cpu_total_time': 0,
                'gpu_total_time': 0
            }
        }
        
        # Process each test case
        for test_name, image_path in test_cases:
            self.logger.info(f"\nProcessing: {test_name}")
            
            # Run CPU pipeline
            self.logger.info("  Running CPU pipeline...")
            cpu_start = time.time()
            cpu_result = self.run_pipeline(image_path, force_cpu=True)
            cpu_time = time.time() - cpu_start
            
            # Run GPU pipeline
            self.logger.info("  Running GPU pipeline...")
            gpu_start = time.time()
            gpu_result = self.run_pipeline(image_path, force_cpu=False)
            gpu_time = time.time() - gpu_start
            
            # Compare results
            self.logger.info("  Comparing results...")
            comparison = self.compare_results(cpu_result, gpu_result)
            
            # Store results
            test_result = {
                'name': test_name,
                'image_path': image_path,
                'cpu_success': cpu_result['success'],
                'gpu_success': gpu_result['success'],
                'consistent': comparison['consistent'],
                'comparison': comparison,
                'timing': {
                    'cpu_time': cpu_time,
                    'gpu_time': gpu_time,
                    'speedup': cpu_time / gpu_time if gpu_time > 0 else 0
                }
            }
            
            results['test_cases'].append(test_result)
            
            if not comparison['consistent']:
                results['overall_consistency'] = False
            
            # Update performance metrics
            results['performance_metrics']['cpu_total_time'] += cpu_time
            results['performance_metrics']['gpu_total_time'] += gpu_time
            
            # Log results
            self.logger.info(f"  Consistent: {comparison['consistent']}")
            self.logger.info(f"  Speedup: {test_result['timing']['speedup']:.2f}x")
            if comparison['differences']:
                self.logger.warning(f"  Differences: {comparison['differences']}")
        
        # Calculate average speedup
        if results['performance_metrics']['gpu_total_time'] > 0:
            results['performance_metrics']['average_speedup'] = (
                results['performance_metrics']['cpu_total_time'] / 
                results['performance_metrics']['gpu_total_time']
            )
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate validation report"""
        report = []
        report.append("="*60)
        report.append("GPU/CPU Consistency Validation Report")
        report.append("="*60)
        report.append("")
        
        if not results['gpu_available']:
            report.append("GPU not available for validation")
            return "\n".join(report)
        
        report.append(f"Overall Consistency: {'PASS' if results['overall_consistency'] else 'FAIL'}")
        report.append(f"Average Speedup: {results['performance_metrics']['average_speedup']:.2f}x")
        report.append(f"Total CPU Time: {results['performance_metrics']['cpu_total_time']:.2f}s")
        report.append(f"Total GPU Time: {results['performance_metrics']['gpu_total_time']:.2f}s")
        report.append("")
        
        # Detailed test results
        report.append("Test Case Results:")
        report.append("-"*60)
        
        for test in results['test_cases']:
            report.append(f"\n{test['name']}:")
            report.append(f"  CPU Success: {test['cpu_success']}")
            report.append(f"  GPU Success: {test['gpu_success']}")
            report.append(f"  Consistent: {test['consistent']}")
            report.append(f"  Speedup: {test['timing']['speedup']:.2f}x")
            
            if test['consistent']:
                comp = test['comparison']
                report.append(f"  Quality Score Diff: {comp['metrics'].get('quality_score_diff', 0):.2f}%")
                report.append(f"  Defect Count Diff: {comp['metrics'].get('defect_count_diff', 0)}")
            else:
                report.append(f"  Differences: {test['comparison']['differences']}")
        
        report.append("")
        report.append("="*60)
        
        return "\n".join(report)


def main():
    """Run validation"""
    validator = ConsistencyValidator()
    
    try:
        # Run validation
        results = validator.validate_consistency()
        
        # Generate report
        report = validator.generate_report(results)
        print(report)
        
        # Save report
        report_path = "gpu_cpu_validation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save detailed results
        results_path = "gpu_cpu_validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        print(f"Detailed results saved to: {results_path}")
        
        # Return exit code based on consistency
        return 0 if results.get('overall_consistency', False) else 1
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())