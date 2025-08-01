#!/usr/bin/env python3
"""
GPU-Accelerated Fiber Optic Analysis Pipeline - Interactive Version
Main application with interactive prompts instead of command-line arguments
"""

import os
import sys
import json
import cv2
import numpy as np
import logging
import time
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


class InteractiveInterface:
    """Interactive interface for user input"""
    
    @staticmethod
    def get_yes_no(prompt: str, default: bool = True) -> bool:
        """Get yes/no response from user"""
        default_str = "Y/n" if default else "y/N"
        while True:
            response = input(f"{prompt} [{default_str}]: ").strip().lower()
            if response == '':
                return default
            elif response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    @staticmethod
    def get_file_path(prompt: str, must_exist: bool = True, 
                     file_type: str = "file") -> str:
        """Get file or directory path from user"""
        while True:
            path = input(f"{prompt}: ").strip()
            
            # Expand user home directory
            path = os.path.expanduser(path)
            
            if must_exist:
                if file_type == "file" and os.path.isfile(path):
                    return path
                elif file_type == "directory" and os.path.isdir(path):
                    return path
                else:
                    print(f"Error: {file_type.capitalize()} '{path}' does not exist. Please try again.")
            else:
                # For output paths, just check if parent directory exists
                parent = os.path.dirname(path) if file_type == "file" else path
                if parent == '' or os.path.exists(os.path.dirname(parent) or '.'):
                    return path
                else:
                    print(f"Error: Parent directory does not exist. Please try again.")
    
    @staticmethod
    def get_choice(prompt: str, choices: List[Tuple[str, str]], 
                  default: Optional[int] = None) -> str:
        """Get choice from list of options"""
        print(f"\n{prompt}")
        for i, (value, description) in enumerate(choices):
            default_marker = " (default)" if default is not None and i == default else ""
            print(f"  {i+1}. {description}{default_marker}")
        
        while True:
            response = input(f"Enter choice [1-{len(choices)}]: ").strip()
            
            if response == '' and default is not None:
                return choices[default][0]
            
            try:
                choice_idx = int(response) - 1
                if 0 <= choice_idx < len(choices):
                    return choices[choice_idx][0]
                else:
                    print(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                print("Please enter a valid number")
    
    @staticmethod
    def get_number(prompt: str, min_val: Optional[float] = None,
                  max_val: Optional[float] = None, 
                  default: Optional[float] = None,
                  is_int: bool = False) -> float:
        """Get numeric input from user"""
        default_str = f" (default: {default})" if default is not None else ""
        full_prompt = f"{prompt}{default_str}: "
        
        while True:
            response = input(full_prompt).strip()
            
            if response == '' and default is not None:
                return default
            
            try:
                value = int(response) if is_int else float(response)
                
                if min_val is not None and value < min_val:
                    print(f"Value must be at least {min_val}")
                    continue
                
                if max_val is not None and value > max_val:
                    print(f"Value must be at most {max_val}")
                    continue
                
                return value
            except ValueError:
                print(f"Please enter a valid {'integer' if is_int else 'number'}")


class FiberAnalysisPipelineGPU:
    """GPU-accelerated fiber optic analysis pipeline with interactive interface"""
    
    def __init__(self, config: Optional[Dict] = None, force_cpu: Optional[bool] = None):
        """
        Initialize the GPU-accelerated pipeline
        
        Args:
            config: Configuration dictionary (if None, will use interactive config)
            force_cpu: Force CPU mode (if None, will ask user)
        """
        self.interface = InteractiveInterface()
        
        # Determine GPU mode interactively if not specified
        if force_cpu is None:
            force_cpu = self._get_gpu_preference()
        
        self.force_cpu = force_cpu
        self.gpu_manager = GPUManager(force_cpu)
        self.logger = logging.getLogger('FiberAnalysisPipelineGPU')
        
        # Get configuration
        if config is None:
            config = self._get_configuration()
        self.config = config
        
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
    
    def _get_gpu_preference(self) -> bool:
        """Interactively determine GPU usage preference"""
        print("\n" + "="*60)
        print("GPU Configuration")
        print("="*60)
        
        # Check GPU availability
        temp_manager = GPUManager(force_cpu=False)
        
        if temp_manager.use_gpu:
            print(f"\nGPU detected: {temp_manager.device.name}")
            print(f"GPU Memory: {temp_manager.device.mem_info[1] / 1e9:.2f} GB")
            
            use_gpu = self.interface.get_yes_no(
                "\nWould you like to use GPU acceleration?",
                default=True
            )
            return not use_gpu  # force_cpu is inverse of use_gpu
        else:
            print("\nNo GPU detected. The pipeline will run in CPU mode.")
            input("Press Enter to continue...")
            return True
    
    def _get_configuration(self) -> Dict:
        """Interactively get configuration from user"""
        print("\n" + "="*60)
        print("Pipeline Configuration")
        print("="*60)
        
        # Ask if user wants to use a config file
        use_config_file = self.interface.get_yes_no(
            "\nWould you like to load configuration from a file?",
            default=False
        )
        
        if use_config_file:
            config_path = self.interface.get_file_path(
                "Enter configuration file path",
                must_exist=True,
                file_type="file"
            )
            return self._load_config(config_path)
        
        # Ask if user wants default or custom configuration
        config_mode = self.interface.get_choice(
            "Select configuration mode:",
            [
                ("default", "Use default settings (recommended)"),
                ("custom", "Customize settings"),
                ("minimal", "Minimal settings (fastest processing)")
            ],
            default=0
        )
        
        if config_mode == "default":
            return self._get_default_config()
        elif config_mode == "minimal":
            return self._get_minimal_config()
        else:
            return self._get_custom_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
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
    
    def _get_minimal_config(self) -> Dict:
        """Get minimal configuration for fastest processing"""
        config = self._get_default_config()
        config['process']['enable_all_filters'] = False
        config['detection']['enable_visualization'] = False
        return config
    
    def _get_custom_config(self) -> Dict:
        """Interactively get custom configuration"""
        print("\n" + "-"*40)
        print("Custom Configuration")
        print("-"*40)
        
        config = self._get_default_config()
        
        # Process configuration
        print("\n[Image Processing Settings]")
        config['process']['enable_all_filters'] = self.interface.get_yes_no(
            "Enable all image filters? (slower but more thorough)",
            default=True
        )
        
        # Detection configuration
        print("\n[Defect Detection Settings]")
        config['detection']['min_defect_size'] = self.interface.get_number(
            "Minimum defect size in pixels",
            min_val=1,
            max_val=1000,
            default=10,
            is_int=True
        )
        
        config['detection']['anomaly_threshold_multiplier'] = self.interface.get_number(
            "Anomaly threshold multiplier (higher = less sensitive)",
            min_val=1.0,
            max_val=5.0,
            default=2.5
        )
        
        config['detection']['enable_visualization'] = self.interface.get_yes_no(
            "Generate visualization images?",
            default=True
        )
        
        # Quality thresholds
        print("\n[Quality Thresholds]")
        print("Define quality score thresholds (0-100)")
        
        config['acquisition']['quality_thresholds']['perfect'] = self.interface.get_number(
            "Perfect quality threshold",
            min_val=90,
            max_val=100,
            default=95,
            is_int=True
        )
        
        config['acquisition']['quality_thresholds']['good'] = self.interface.get_number(
            "Good quality threshold",
            min_val=70,
            max_val=94,
            default=85,
            is_int=True
        )
        
        config['acquisition']['quality_thresholds']['acceptable'] = self.interface.get_number(
            "Acceptable quality threshold",
            min_val=50,
            max_val=84,
            default=70,
            is_int=True
        )
        
        # Ask if user wants to save configuration
        save_config = self.interface.get_yes_no(
            "\nWould you like to save this configuration for future use?",
            default=False
        )
        
        if save_config:
            config_path = self.interface.get_file_path(
                "Enter path to save configuration (e.g., my_config.json)",
                must_exist=False,
                file_type="file"
            )
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to: {config_path}")
        
        return config
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file with defaults"""
        default_config = self._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Merge with defaults
            for key in default_config:
                if key in loaded_config:
                    default_config[key].update(loaded_config[key])
            
            print(f"Configuration loaded from: {config_path}")
            return default_config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration instead.")
            return default_config
    
    def run_interactive(self):
        """Run the pipeline with interactive prompts"""
        print("\n" + "="*60)
        print("Fiber Optic Analysis Pipeline")
        print("="*60)
        
        # Get processing mode
        mode = self.interface.get_choice(
            "Select processing mode:",
            [
                ("single", "Process a single image"),
                ("batch", "Process multiple images in a directory"),
                ("exit", "Exit program")
            ],
            default=0
        )
        
        if mode == "exit":
            print("Exiting...")
            return
        elif mode == "single":
            self._process_single_interactive()
        else:
            self._process_batch_interactive()
        
        # Ask if user wants to process more
        if self.interface.get_yes_no("\nWould you like to process more images?", default=False):
            self.run_interactive()
    
    def _process_single_interactive(self):
        """Process a single image interactively"""
        print("\n[Single Image Processing]")
        
        # Get input image
        image_path = self.interface.get_file_path(
            "Enter path to input image",
            must_exist=True,
            file_type="file"
        )
        
        # Get output directory
        default_output = os.path.join(
            os.path.dirname(image_path),
            f"{Path(image_path).stem}_analysis"
        )
        print(f"\nDefault output directory: {default_output}")
        
        use_default = self.interface.get_yes_no(
            "Use default output directory?",
            default=True
        )
        
        if use_default:
            output_dir = default_output
        else:
            output_dir = self.interface.get_file_path(
                "Enter output directory path",
                must_exist=False,
                file_type="directory"
            )
        
        # Process image
        print("\n" + "-"*40)
        print("Processing image...")
        print("-"*40)
        
        try:
            summary = self.analyze_image(image_path, output_dir)
            
            # Display results
            print("\n" + "="*60)
            print("Analysis Results")
            print("="*60)
            print(f"Image: {summary['image_path']}")
            print(f"Quality Score: {summary['quality_score']:.1f}%")
            print(f"Status: {summary['pass_fail_status']}")
            print(f"Total Defects: {summary['total_defects']}")
            print(f"Critical Defects: {summary['critical_defects']}")
            print(f"Processing Time: {summary['processing_time']['total']:.2f}s")
            print(f"Results saved to: {summary['output_dir']}")
            
            # Show detailed timing if user wants
            if self.interface.get_yes_no("\nShow detailed timing information?", default=False):
                print("\nDetailed Timing:")
                for stage, time in summary['processing_time'].items():
                    if stage != 'total':
                        print(f"  {stage}: {time:.2f}s")
            
        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
    
    def _process_batch_interactive(self):
        """Process batch of images interactively"""
        print("\n[Batch Processing]")
        
        # Get input directory
        input_dir = self.interface.get_file_path(
            "Enter directory containing images",
            must_exist=True,
            file_type="directory"
        )
        
        # Get file pattern
        pattern_choice = self.interface.get_choice(
            "Select file pattern:",
            [
                ("*.png", "PNG files only"),
                ("*.jpg", "JPEG files only"),
                ("*.*", "All image files"),
                ("custom", "Custom pattern")
            ],
            default=0
        )
        
        if pattern_choice == "custom":
            pattern = input("Enter custom file pattern (e.g., fiber_*.png): ").strip()
        else:
            pattern = pattern_choice
        
        # Count matching files
        from pathlib import Path
        matching_files = list(Path(input_dir).glob(pattern))
        
        if not matching_files:
            print(f"\nNo files matching pattern '{pattern}' found in {input_dir}")
            return
        
        print(f"\nFound {len(matching_files)} files matching pattern '{pattern}'")
        
        # Get output directory
        default_output = os.path.join(input_dir, "analysis_results")
        print(f"\nDefault output directory: {default_output}")
        
        use_default = self.interface.get_yes_no(
            "Use default output directory?",
            default=True
        )
        
        if use_default:
            output_dir = default_output
        else:
            output_dir = self.interface.get_file_path(
                "Enter output directory path",
                must_exist=False,
                file_type="directory"
            )
        
        # Confirm processing
        if not self.interface.get_yes_no(
            f"\nProcess {len(matching_files)} images?",
            default=True
        ):
            print("Batch processing cancelled.")
            return
        
        # Process batch
        print("\n" + "-"*40)
        print("Processing batch...")
        print("-"*40)
        
        try:
            results = self.batch_analyze(input_dir, output_dir, pattern)
            
            # Display summary
            successful = len([r for r in results if 'error' not in r])
            failed = len([r for r in results if 'error' in r])
            
            print("\n" + "="*60)
            print("Batch Processing Results")
            print("="*60)
            print(f"Total images: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            if successful > 0:
                avg_quality = np.mean([r['quality_score'] for r in results if 'quality_score' in r])
                print(f"Average quality score: {avg_quality:.1f}%")
            
            print(f"\nResults saved to: {output_dir}")
            
            # Show failed images if any
            if failed > 0 and self.interface.get_yes_no("\nShow failed images?", default=True):
                print("\nFailed images:")
                for r in results:
                    if 'error' in r:
                        print(f"  - {r['image_path']}: {r['error']}")
            
        except Exception as e:
            print(f"\nError during batch processing: {str(e)}")
            self.logger.error(f"Batch processing failed: {str(e)}", exc_info=True)
    
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
            print(f"\nProcessing image {i+1}/{len(image_files)}: {image_file.name}")
            
            # Create output subdirectory for each image
            image_output_dir = Path(output_dir) / image_file.stem
            
            try:
                summary = self.analyze_image(str(image_file), str(image_output_dir))
                results.append(summary)
                print(f"  ✓ Quality: {summary['quality_score']:.1f}% - {summary['pass_fail_status']}")
            except Exception as e:
                self.logger.error(f"Failed to process {image_file}: {str(e)}")
                results.append({
                    'image_path': str(image_file),
                    'error': str(e),
                    'status': 'FAILED'
                })
                print(f"  ✗ Failed: {str(e)}")
            
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
    """Main entry point with interactive interface"""
    print("="*60)
    print("GPU-Accelerated Fiber Optic Analysis Pipeline")
    print("Interactive Mode")
    print("="*60)
    
    try:
        # Create pipeline (will prompt for configuration)
        pipeline = FiberAnalysisPipelineGPU()
        
        # Run interactive interface
        pipeline.run_interactive()
        
        print("\nThank you for using the Fiber Optic Analysis Pipeline!")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())