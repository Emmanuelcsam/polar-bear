#!/usr/bin/env python3
"""
GPU-Accelerated Fiber Optic Analysis Pipeline
Main application that orchestrates all GPU-accelerated modules
"""

import os
import sys
import json
import cv2
import numpy as np
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Import GPU modules
from gpu_utils import GPUManager, log_gpu_memory, clear_gpu_memory
from process_gpu import ImageProcessorGPU
from separation_gpu import UnifiedSeparationGPU
from detection_gpu import OmniFiberAnalyzerGPU, OmniConfigGPU
from data_acquisition_gpu import DataAcquisitionGPU

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fiber_analysis_gpu.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('FiberAnalysisPipelineGPU')


class FiberAnalysisPipelineGPU:
    """GPU-accelerated fiber optic analysis pipeline"""
    
    def __init__(self, config_path: Optional[str] = None, force_cpu: bool = False):
        """
        Initialize the GPU-accelerated pipeline
        
        Args:
            config_path: Path to configuration file
            force_cpu: Force CPU mode for testing
        """
        self.force_cpu = force_cpu
        self.gpu_manager = GPUManager(force_cpu)
        self.logger = logging.getLogger('FiberAnalysisPipelineGPU')
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize modules
        self.processor = ImageProcessorGPU(self.config.get('process', {}), force_cpu)
        self.separator = UnifiedSeparationGPU(self.config.get('separation', {}), force_cpu)
        self.detector = OmniFiberAnalyzerGPU(
            OmniConfigGPU(**self.config.get('detection', {})), 
            force_cpu
        )
        self.acquisitor = DataAcquisitionGPU(self.config.get('acquisition', {}), force_cpu)
        
        self.logger.info(f"Initialized FiberAnalysisPipelineGPU with GPU={self.gpu_manager.use_gpu}")
        log_gpu_memory()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'process': {
                'enable_all_filters': True
            },
            'separation': {
                'consensus_threshold': 3
            },
            'detection': {
                'min_defect_size': 10,
                'max_defect_size': 5000,
                'anomaly_threshold_multiplier': 2.5,
                'confidence_threshold': 0.3,
                'enable_visualization': True
            },
            'acquisition': {
                'clustering_eps': 20,
                'clustering_min_samples': 2,
                'quality_thresholds': {
                    'perfect': 95,
                    'good': 85,
                    'acceptable': 70,
                    'poor': 50
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key in loaded_config:
                        default_config[key].update(loaded_config[key])
                return default_config
        
        return default_config
    
    def analyze_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Analyze a single fiber optic image
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output files
            
        Returns:
            Dictionary containing complete analysis results
        """
        start_time = time.time()
        self.logger.info(f"Starting analysis of {image_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        try:
            # Stage 1: Process image
            self.logger.info("Stage 1: Processing image with GPU acceleration...")
            stage1_start = time.time()
            
            processed_arrays = self.processor.process_single_image(
                image_path, 
                str(output_path / "processed"),
                return_arrays=True
            )
            
            stage1_time = time.time() - stage1_start
            self.logger.info(f"Stage 1 completed in {stage1_time:.2f}s")
            log_gpu_memory()
            
            # Stage 2: Separation
            self.logger.info("Stage 2: Separating fiber zones...")
            stage2_start = time.time()
            
            separation_result = self.separator.process_image(image_path, processed_arrays)
            
            stage2_time = time.time() - stage2_start
            self.logger.info(f"Stage 2 completed in {stage2_time:.2f}s")
            log_gpu_memory()
            
            # Stage 3: Detection
            self.logger.info("Stage 3: Detecting defects in regions...")
            stage3_start = time.time()
            
            detection_result = self.detector.analyze_regions(
                separation_result.regions,
                original_image
            )
            
            stage3_time = time.time() - stage3_start
            self.logger.info(f"Stage 3 completed in {stage3_time:.2f}s")
            log_gpu_memory()
            
            # Stage 4: Data Acquisition
            self.logger.info("Stage 4: Aggregating results...")
            stage4_start = time.time()
            
            # Convert detection result to format expected by acquisition
            detection_results = [
                {
                    'core_defects': detection_result.core_result.defects,
                    'cladding_defects': detection_result.cladding_result.defects,
                    'ferrule_defects': detection_result.ferrule_result.defects,
                    'quality_score': detection_result.overall_quality
                }
            ]
            
            acquisition_result = self.acquisitor.aggregate_results(
                detection_results,
                original_image,
                {'masks': separation_result.masks}
            )
            
            stage4_time = time.time() - stage4_start
            self.logger.info(f"Stage 4 completed in {stage4_time:.2f}s")
            log_gpu_memory()
            
            # Save all results
            self._save_results(
                output_path,
                {
                    'separation': separation_result,
                    'detection': detection_result,
                    'acquisition': acquisition_result,
                    'timing': {
                        'stage1_process': stage1_time,
                        'stage2_separation': stage2_time,
                        'stage3_detection': stage3_time,
                        'stage4_acquisition': stage4_time,
                        'total': time.time() - start_time
                    }
                }
            )
            
            # Prepare summary
            total_time = time.time() - start_time
            summary = {
                'image_path': image_path,
                'output_dir': str(output_path),
                'quality_score': acquisition_result.quality_metrics['quality_score'],
                'pass_fail_status': acquisition_result.pass_fail_status,
                'total_defects': acquisition_result.quality_metrics['total_defects'],
                'critical_defects': acquisition_result.quality_metrics['critical_defects'],
                'processing_time': {
                    'stage1_process': stage1_time,
                    'stage2_separation': stage2_time,
                    'stage3_detection': stage3_time,
                    'stage4_acquisition': stage4_time,
                    'total': total_time
                },
                'gpu_info': {
                    'gpu_used': self.gpu_manager.use_gpu,
                    'device_name': self.gpu_manager.device.name if self.gpu_manager.use_gpu else 'CPU'
                }
            }
            
            self.logger.info(f"Analysis completed in {total_time:.2f}s")
            self.logger.info(f"Quality Score: {summary['quality_score']:.1f}%")
            self.logger.info(f"Status: {summary['pass_fail_status']}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            raise
        finally:
            # Clear GPU memory
            clear_gpu_memory()
    
    def _save_results(self, output_path: Path, results: Dict[str, Any]):
        """Save all analysis results"""
        # Save separation masks
        masks_dir = output_path / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        for mask_name, mask in results['separation'].masks.items():
            cv2.imwrite(str(masks_dir / f"{mask_name}_mask.png"), mask * 255)
        
        # Save separated regions
        regions_dir = output_path / "regions"
        regions_dir.mkdir(exist_ok=True)
        
        for region_name, region in results['separation'].regions.items():
            cv2.imwrite(str(regions_dir / f"{region_name}_region.png"), region)
        
        # Save detection results
        self.detector.save_results(
            results['detection'],
            str(output_path / "detection_report.json")
        )
        
        # Save acquisition results
        self.acquisitor.save_results(
            results['acquisition'],
            str(output_path)
        )
        
        # Save timing information
        with open(output_path / "timing_report.json", 'w') as f:
            json.dump(results['timing'], f, indent=2)
        
        # Save complete summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'separation_metadata': results['separation'].metadata,
            'detection_summary': {
                'overall_quality': results['detection'].overall_quality,
                'total_defects': results['detection'].total_defects,
                'critical_defects': results['detection'].critical_defects
            },
            'acquisition_summary': {
                'quality_score': results['acquisition'].quality_metrics['quality_score'],
                'pass_fail_status': results['acquisition'].pass_fail_status,
                'aggregated_defects': len(results['acquisition'].aggregated_defects)
            },
            'timing': results['timing'],
            'gpu_used': self.gpu_manager.use_gpu
        }
        
        with open(output_path / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def batch_analyze(self, input_dir: str, output_dir: str, 
                     pattern: str = "*.png") -> List[Dict[str, Any]]:
        """
        Analyze multiple images in batch
        
        Args:
            input_dir: Directory containing input images
            output_dir: Base directory for outputs
            pattern: File pattern to match
            
        Returns:
            List of analysis summaries
        """
        input_path = Path(input_dir)
        image_files = list(input_path.glob(pattern))
        
        if not image_files:
            self.logger.warning(f"No images found matching {pattern} in {input_dir}")
            return []
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        results = []
        for i, image_file in enumerate(image_files):
            self.logger.info(f"Processing image {i+1}/{len(image_files)}: {image_file.name}")
            
            # Create output subdirectory for each image
            image_output_dir = Path(output_dir) / image_file.stem
            
            try:
                summary = self.analyze_image(str(image_file), str(image_output_dir))
                results.append(summary)
            except Exception as e:
                self.logger.error(f"Failed to process {image_file}: {str(e)}")
                results.append({
                    'image_path': str(image_file),
                    'error': str(e),
                    'status': 'FAILED'
                })
            
            # Clear GPU memory between images
            clear_gpu_memory()
        
        # Save batch summary
        batch_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'successful': len([r for r in results if 'error' not in r]),
            'failed': len([r for r in results if 'error' in r]),
            'results': results
        }
        
        summary_path = Path(output_dir) / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        self.logger.info(f"Batch processing completed. Summary saved to {summary_path}")
        
        return results


def main():
    """Main entry point for the GPU-accelerated pipeline"""
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Fiber Optic Analysis Pipeline"
    )
    
    parser.add_argument(
        "input",
        help="Input image file or directory"
    )
    
    parser.add_argument(
        "output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file",
        default=None
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (disable GPU acceleration)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process directory of images in batch mode"
    )
    
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="File pattern for batch mode (default: *.png)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FiberAnalysisPipelineGPU(args.config, args.cpu)
    
    # Process based on mode
    if args.batch or os.path.isdir(args.input):
        # Batch mode
        results = pipeline.batch_analyze(args.input, args.output, args.pattern)
        
        # Print summary
        successful = len([r for r in results if 'error' not in r])
        failed = len([r for r in results if 'error' in r])
        
        print(f"\nBatch processing completed:")
        print(f"  Total images: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        if successful > 0:
            avg_quality = np.mean([r['quality_score'] for r in results if 'quality_score' in r])
            print(f"  Average quality score: {avg_quality:.1f}%")
    else:
        # Single image mode
        summary = pipeline.analyze_image(args.input, args.output)
        
        # Print results
        print(f"\nAnalysis completed:")
        print(f"  Image: {summary['image_path']}")
        print(f"  Quality Score: {summary['quality_score']:.1f}%")
        print(f"  Status: {summary['pass_fail_status']}")
        print(f"  Total Defects: {summary['total_defects']}")
        print(f"  Critical Defects: {summary['critical_defects']}")
        print(f"  Processing Time: {summary['processing_time']['total']:.2f}s")
        print(f"  GPU Used: {summary['gpu_info']['gpu_used']}")
        print(f"  Results saved to: {summary['output_dir']}")


if __name__ == "__main__":
    main()