#!/usr/bin/env python3
"""
Pipeline Orchestrator Module

A comprehensive, modular pipeline management system for image processing workflows.
This module provides a flexible framework for orchestrating multi-stage image analysis
pipelines with configuration management, logging, and error handling.

Key Features:
- JSON-based configuration management
- Multi-stage pipeline orchestration
- Flexible path resolution
- Comprehensive logging
- Batch processing capabilities
- Interactive CLI interface
- Error handling and recovery

Extracted from: app.py
Author: GitHub Copilot
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
import logging
import shlex
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration container for pipeline settings."""
    config_path: Path
    results_dir: Path
    process_settings: Dict[str, Any]
    separation_settings: Dict[str, Any]
    detection_settings: Dict[str, Any]
    data_acquisition_settings: Dict[str, Any]
    paths: Dict[str, str]


class PipelineLogger:
    """Centralized logging configuration for pipeline operations."""
    
    @staticmethod
    def setup_logging(level=logging.INFO, format_string=None):
        """Setup structured logging for pipeline operations."""
        if format_string is None:
            format_string = '%(asctime)s - [%(levelname)s] - %(message)s'
        
        logging.basicConfig(
            level=level,
            format=format_string,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger(__name__)


class ConfigurationManager:
    """Handles configuration loading and path resolution."""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path).resolve()
        self.config = self.load_config()
        self.config = self.resolve_config_paths()
    
    def load_config(self) -> Dict[str, Any]:
        """Load JSON configuration file with error handling."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Successfully loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise
    
    def resolve_config_paths(self) -> Dict[str, Any]:
        """Convert relative paths in config to absolute paths."""
        config_dir = self.config_path.parent
        
        # Update paths to be absolute
        for key in ['results_dir', 'zones_methods_dir', 'detection_knowledge_base']:
            if key in self.config.get('paths', {}):
                path = Path(self.config['paths'][key])
                if not path.is_absolute():
                    self.config['paths'][key] = str(config_dir / path)
        
        return self.config
    
    def get_pipeline_config(self) -> PipelineConfig:
        """Create structured configuration object."""
        return PipelineConfig(
            config_path=self.config_path,
            results_dir=Path(self.config['paths']['results_dir']),
            process_settings=self.config.get('process_settings', {}),
            separation_settings=self.config.get('separation_settings', {}),
            detection_settings=self.config.get('detection_settings', {}),
            data_acquisition_settings=self.config.get('data_acquisition_settings', {}),
            paths=self.config.get('paths', {})
        )


class PipelineOrchestrator:
    """
    Main pipeline orchestration class that manages multi-stage image processing workflows.
    
    This class provides a flexible framework for running complex image analysis pipelines
    with proper error handling, logging, and result management.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize the orchestrator with configuration."""
        logging.info("Initializing Pipeline Orchestrator...")
        
        self.config_manager = ConfigurationManager(config_path)
        self.config = self.config_manager.get_pipeline_config()
        
        # Create results directory
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Results will be saved in: {self.config.results_dir}")
    
    def create_run_directory(self, input_image_path: Path) -> Path:
        """Create a unique directory for pipeline run results."""
        run_dir = self.config.results_dir / input_image_path.stem
        run_dir.mkdir(exist_ok=True)
        return run_dir
    
    def run_full_pipeline(self, input_image_path: Path, 
                         custom_processors: Optional[Dict[str, callable]] = None) -> Optional[Dict[str, Any]]:
        """
        Run the complete analysis pipeline for a single image.
        
        Args:
            input_image_path: Path to the input image
            custom_processors: Optional dictionary of custom processing functions
                             Keys: 'process', 'separate', 'detect', 'acquire'
        
        Returns:
            Final analysis report or None if pipeline fails
        """
        start_time = time.time()
        logging.info(f"--- Starting full pipeline for: {input_image_path.name} ---")
        
        # Create run directory
        run_dir = self.create_run_directory(input_image_path)
        
        try:
            # Stage 1: Processing
            all_images_to_separate = self.run_processing_stage(
                input_image_path, run_dir, custom_processors
            )
            
            # Stage 2: Separation
            all_images_to_detect = self.run_separation_stage(
                all_images_to_separate, run_dir, input_image_path, custom_processors
            )
            
            # Stage 3: Detection
            self.run_detection_stage(all_images_to_detect, run_dir, custom_processors)
            
            # Stage 4: Data Acquisition
            final_report = self.run_data_acquisition_stage(
                input_image_path, run_dir, custom_processors
            )
            
            # Log completion
            end_time = time.time()
            logging.info(f"--- Pipeline for {input_image_path.name} completed in {end_time - start_time:.2f} seconds ---")
            
            # Create summary
            self.create_pipeline_summary(input_image_path, run_dir, final_report)
            
            return final_report
            
        except Exception as e:
            logging.error(f"Pipeline failed for {input_image_path.name}: {e}", exc_info=True)
            return None
    
    def run_processing_stage(self, input_image_path: Path, run_dir: Path, 
                           custom_processors: Optional[Dict[str, callable]] = None) -> List[Path]:
        """
        Run the processing stage of the pipeline.
        
        Args:
            input_image_path: Input image path
            run_dir: Run directory for outputs
            custom_processors: Optional custom processing functions
        
        Returns:
            List of all images to pass to separation stage
        """
        logging.info(">>> STAGE 1: PROCESSING - Reimagining images...")
        
        process_cfg = self.config.process_settings
        reimagined_dir = run_dir / process_cfg.get('output_folder_name', 'processed')
        
        all_images_to_separate = [input_image_path]
        
        try:
            # Use custom processor if provided
            if custom_processors and 'process' in custom_processors:
                processor = custom_processors['process']
                reimagined_files = processor(str(input_image_path), str(reimagined_dir))
            else:
                logging.warning("No processing function available. Using original image only.")
                reimagined_files = []
            
            # Add reimagined files to processing list
            if isinstance(reimagined_files, list):
                all_images_to_separate.extend([Path(f) for f in reimagined_files])
            elif reimagined_dir.exists():
                reimagined_files = list(reimagined_dir.glob('*.jpg'))
                all_images_to_separate.extend(reimagined_files)
            
            logging.info(f"Processing stage complete. Found {len(all_images_to_separate) - 1} reimagined images.")
            
        except Exception as e:
            logging.error(f"Error during processing stage: {e}")
        
        return all_images_to_separate
    
    def run_separation_stage(self, image_paths: List[Path], run_dir: Path, 
                           original_image_path: Path,
                           custom_processors: Optional[Dict[str, callable]] = None) -> List[Path]:
        """
        Run the separation stage of the pipeline.
        
        Args:
            image_paths: List of images to separate
            run_dir: Run directory for outputs
            original_image_path: Original input image
            custom_processors: Optional custom processing functions
        
        Returns:
            List of all images to pass to detection stage
        """
        logging.info(">>> STAGE 2: SEPARATION - Generating zoned regions...")
        
        separation_cfg = self.config.separation_settings
        separated_dir = run_dir / separation_cfg.get('output_folder_name', 'separated')
        separated_dir.mkdir(exist_ok=True)
        
        all_separated_regions = []
        
        try:
            # Use custom separator if provided
            if custom_processors and 'separate' in custom_processors:
                separator = custom_processors['separate']
                for image_path in image_paths:
                    logging.info(f"Separating image: {image_path.name}")
                    image_separation_output_dir = separated_dir / image_path.stem
                    
                    regions = separator(image_path, str(image_separation_output_dir))
                    if regions:
                        all_separated_regions.extend([Path(r) for r in regions])
            else:
                logging.warning("No separation function available. Using original images only.")
        
        except Exception as e:
            logging.error(f"Error during separation stage: {e}", exc_info=True)
        
        # Include original image and all separated regions
        all_images_to_detect = all_separated_regions + [original_image_path]
        
        logging.info(f"Separation stage complete. Generated {len(all_separated_regions)} separated regions.")
        logging.info(f"Total inputs for detection stage: {len(all_images_to_detect)}")
        
        return all_images_to_detect
    
    def run_detection_stage(self, image_paths: List[Path], run_dir: Path,
                          custom_processors: Optional[Dict[str, callable]] = None):
        """
        Run the detection stage of the pipeline.
        
        Args:
            image_paths: List of images to analyze
            run_dir: Run directory for outputs
            custom_processors: Optional custom processing functions
        """
        logging.info(">>> STAGE 3: DETECTION - Analyzing for defects...")
        
        detection_cfg = self.config.detection_settings
        detection_output_dir = run_dir / detection_cfg.get('output_folder_name', 'detection')
        detection_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use custom detector if provided
            if custom_processors and 'detect' in custom_processors:
                detector = custom_processors['detect']
                
                for image_path in image_paths:
                    logging.info(f"Detecting defects in: {image_path.name}")
                    image_detection_output_dir = detection_output_dir / image_path.stem
                    image_detection_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    detector(str(image_path), str(image_detection_output_dir))
            else:
                logging.warning("No detection function available. Skipping detection stage.")
        
        except Exception as e:
            logging.error(f"Error during detection stage: {e}", exc_info=True)
        
        logging.info("Detection stage complete.")
    
    def run_data_acquisition_stage(self, original_image_path: Path, run_dir: Path,
                                 custom_processors: Optional[Dict[str, callable]] = None) -> Optional[Dict[str, Any]]:
        """
        Run the data acquisition stage of the pipeline.
        
        Args:
            original_image_path: Original input image
            run_dir: Run directory for outputs
            custom_processors: Optional custom processing functions
        
        Returns:
            Final analysis report or None
        """
        logging.info(">>> STAGE 4: DATA ACQUISITION - Aggregating and analyzing all results...")
        
        try:
            # Copy original image to results directory if needed
            original_image_dest = run_dir / original_image_path.name
            if not original_image_dest.exists():
                shutil.copy2(original_image_path, original_image_dest)
            
            # Use custom data acquisition if provided
            if custom_processors and 'acquire' in custom_processors:
                data_acq_cfg = self.config.data_acquisition_settings
                clustering_eps = data_acq_cfg.get('clustering_eps', 30.0)
                
                acquirer = custom_processors['acquire']
                final_report = acquirer(
                    str(run_dir),
                    original_image_path.name,
                    clustering_eps=clustering_eps
                )
                
                if final_report:
                    summary = final_report.get('analysis_summary', {})
                    logging.info(f"Data acquisition complete. Final status: {summary.get('pass_fail_status', 'UNKNOWN')}")
                    return final_report
                else:
                    logging.error("Data acquisition stage failed to produce a report")
                    return None
            else:
                logging.warning("No data acquisition function available. Creating basic report.")
                return self.create_basic_report(original_image_path, run_dir)
                
        except Exception as e:
            logging.error(f"Error during data acquisition stage: {e}", exc_info=True)
            return None
    
    def create_basic_report(self, original_image_path: Path, run_dir: Path) -> Dict[str, Any]:
        """Create a basic analysis report when no data acquisition function is available."""
        return {
            'analysis_summary': {
                'pass_fail_status': 'UNKNOWN',
                'quality_score': 0,
                'total_merged_defects': 0,
                'failure_reasons': ['No data acquisition function available']
            },
            'image_path': str(original_image_path),
            'results_directory': str(run_dir),
            'timestamp': time.time()
        }
    
    def create_pipeline_summary(self, original_image_path: Path, run_dir: Path, 
                              final_report: Optional[Dict[str, Any]]):
        """Create a summary file for the pipeline run."""
        summary_path = run_dir / "PIPELINE_SUMMARY.txt"
        
        with open(summary_path, 'w') as f:
            f.write("PIPELINE EXECUTION SUMMARY\n")
            f.write("=========================\n\n")
            f.write(f"Image: {original_image_path.name}\n")
            f.write(f"Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results Directory: {run_dir}\n\n")
            
            if final_report and 'analysis_summary' in final_report:
                summary = final_report['analysis_summary']
                f.write(f"Status: {summary.get('pass_fail_status', 'UNKNOWN')}\n")
                f.write(f"Quality Score: {summary.get('quality_score', 0)}/100\n")
                f.write(f"Total Defects: {summary.get('total_merged_defects', 0)}\n")
                
                if summary.get('failure_reasons'):
                    f.write(f"\nFailure Reasons:\n")
                    for reason in summary['failure_reasons']:
                        f.write(f"  - {reason}\n")
            else:
                f.write("Status: PIPELINE_ERROR\n")
                f.write("No analysis report generated\n")
            
            f.write(f"\nDetailed results available in subdirectories.\n")


class BatchProcessor:
    """Handles batch processing of multiple images through the pipeline."""
    
    def __init__(self, orchestrator: PipelineOrchestrator):
        self.orchestrator = orchestrator
    
    def process_image_list(self, image_paths: List[Path],
                          custom_processors: Optional[Dict[str, callable]] = None) -> Dict[str, Any]:
        """
        Process a list of images through the pipeline.
        
        Args:
            image_paths: List of image paths to process
            custom_processors: Optional custom processing functions
        
        Returns:
            Batch processing summary
        """
        logging.info(f"Starting batch processing for {len(image_paths)} image(s).")
        
        results = {
            'total_images': len(image_paths),
            'successful': 0,
            'failed': 0,
            'results': []
        }
        
        for image_path in image_paths:
            try:
                final_report = self.orchestrator.run_full_pipeline(image_path, custom_processors)
                
                if final_report and 'analysis_summary' in final_report:
                    summary = final_report['analysis_summary']
                    result = {
                        'image': image_path.name,
                        'status': summary['pass_fail_status'],
                        'quality_score': summary['quality_score'],
                        'defects': summary['total_merged_defects'],
                        'success': True
                    }
                    results['successful'] += 1
                else:
                    result = {
                        'image': image_path.name,
                        'status': 'ERROR',
                        'success': False
                    }
                    results['failed'] += 1
                
                results['results'].append(result)
                
            except Exception as e:
                logging.error(f"Failed to process {image_path}: {e}")
                results['failed'] += 1
                results['results'].append({
                    'image': image_path.name,
                    'status': 'ERROR',
                    'error': str(e),
                    'success': False
                })
        
        logging.info("Finished batch processing.")
        return results
    
    def process_folder(self, folder_path: Path, 
                      image_extensions: Optional[List[str]] = None,
                      custom_processors: Optional[Dict[str, callable]] = None) -> Dict[str, Any]:
        """
        Process all images in a folder through the pipeline.
        
        Args:
            folder_path: Path to folder containing images
            image_extensions: List of image file extensions to process
            custom_processors: Optional custom processing functions
        
        Returns:
            Batch processing summary
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        logging.info(f"Searching for images in directory: {folder_path}")
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
            image_files.extend(folder_path.glob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            logging.warning(f"No images with extensions ({', '.join(image_extensions)}) found in {folder_path}")
            return {'total_images': 0, 'successful': 0, 'failed': 0, 'results': []}
        
        return self.process_image_list(image_files, custom_processors)


class InteractiveCLI:
    """Interactive command-line interface for the pipeline."""
    
    def __init__(self, orchestrator: PipelineOrchestrator):
        self.orchestrator = orchestrator
        self.batch_processor = BatchProcessor(orchestrator)
    
    def ask_for_images(self) -> List[Path]:
        """Prompt user for image paths with validation."""
        print("\nEnter one or more full image paths. Separate paths with spaces.")
        print("Example: C:\\Users\\Test\\img1.png \"C:\\My Images\\test.png\"")
        paths_input = input("> ").strip()
        
        if not paths_input:
            return []
        
        # Use shlex to correctly parse command-line style input
        path_strings = shlex.split(paths_input)
        
        valid_paths = []
        invalid_paths = []
        
        for path_str in path_strings:
            path = Path(path_str)
            if path.is_file():
                valid_paths.append(path)
            else:
                invalid_paths.append(str(path))
        
        if invalid_paths:
            logging.warning(f"The following paths were not found: {', '.join(invalid_paths)}")
        
        return valid_paths
    
    def ask_for_folder(self) -> Optional[Path]:
        """Prompt user for folder path with validation."""
        folder_path_str = input("\nEnter the full path to the folder containing images: ").strip()
        
        if folder_path_str:
            folder_path_str = shlex.split(folder_path_str)[0] if folder_path_str else ""
        
        if not folder_path_str:
            return None
        
        folder_path = Path(folder_path_str)
        
        if folder_path.is_dir():
            return folder_path
        else:
            logging.error(f"Directory not found: {folder_path}")
            return None
    
    def run_interactive_menu(self, custom_processors: Optional[Dict[str, callable]] = None):
        """Run the interactive menu system."""
        print("\n" + "="*80)
        print("MODULAR PIPELINE ORCHESTRATOR".center(80))
        print("Interactive Command-Line Interface".center(80))
        print("="*80)
        
        while True:
            print("\n--- MAIN MENU ---")
            print("1. Process a list of specific images")
            print("2. Process all images in a folder")
            print("3. Exit")
            
            choice = input("Please select an option (1-3): ").strip()
            
            if choice == '1':
                image_paths = self.ask_for_images()
                if not image_paths:
                    logging.warning("No valid image paths provided.")
                    continue
                
                results = self.batch_processor.process_image_list(image_paths, custom_processors)
                self.display_batch_summary(results)
                
            elif choice == '2':
                folder_path = self.ask_for_folder()
                if not folder_path:
                    continue
                
                results = self.batch_processor.process_folder(folder_path, custom_processors=custom_processors)
                self.display_batch_summary(results)
                
            elif choice == '3':
                print("\nExiting the application. Goodbye!")
                break
                
            else:
                logging.warning("Invalid choice. Please enter a number from 1 to 3.")
    
    def display_batch_summary(self, results: Dict[str, Any]):
        """Display batch processing summary."""
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total Images: {results['total_images']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"{'='*60}")
        
        # Display individual results
        for result in results['results']:
            if result['success']:
                print(f"✓ {result['image']}: {result['status']} "
                      f"(Score: {result.get('quality_score', 0)}/100)")
            else:
                print(f"✗ {result['image']}: {result['status']}")


def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Modular Pipeline Orchestrator")
    parser.add_argument("--config", "-c", default="config.json", 
                       help="Path to configuration file")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--image", help="Single image to process")
    parser.add_argument("--folder", help="Folder of images to process")
    
    args = parser.parse_args()
    
    # Setup logging
    PipelineLogger.setup_logging()
    
    # Check config file
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(config_path)
        
        if args.interactive:
            # Run interactive CLI
            cli = InteractiveCLI(orchestrator)
            cli.run_interactive_menu()
            
        elif args.image:
            # Process single image
            image_path = Path(args.image)
            if image_path.exists():
                result = orchestrator.run_full_pipeline(image_path)
                if result:
                    print("Processing completed successfully.")
                else:
                    print("Processing failed.")
            else:
                logging.error(f"Image not found: {image_path}")
                
        elif args.folder:
            # Process folder
            folder_path = Path(args.folder)
            if folder_path.exists():
                batch_processor = BatchProcessor(orchestrator)
                results = batch_processor.process_folder(folder_path)
                
                cli = InteractiveCLI(orchestrator)
                cli.display_batch_summary(results)
            else:
                logging.error(f"Folder not found: {folder_path}")
                
        else:
            # Default to interactive mode
            cli = InteractiveCLI(orchestrator)
            cli.run_interactive_menu()
            
    except Exception as e:
        logging.error(f"Pipeline initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
