#!/usr/bin/env python3
"""
Enhanced Fiber Optic Defect Detection Application
Main orchestrator that brings all components together
- No argparse, uses interactive configuration
- ML-powered processing
- Real-time and batch modes
- Full logging and testing
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import shutil

from config_manager import get_config_manager, get_config
from enhanced_logging import get_logger, ProgressLogger
from process import EnhancedProcessor
from separation import EnhancedSeparator
from detection import EnhancedDetector, Defect
from realtime_processor import RealtimeProcessor

logger = get_logger(__name__)


class DataAcquisition:
    """Final stage - aggregate results and generate reports"""
    
    def __init__(self):
        self.config = get_config()
    
    def process_results(self, image_path: Path, image: np.ndarray,
                       zones: Dict[str, np.ndarray], defects: List[Defect]) -> Dict:
        """Process detection results and generate final output"""
        
        # Calculate pass/fail
        is_pass = self._determine_pass_fail(defects)
        
        # Generate visualizations
        visualizations = self._generate_visualizations(image, zones, defects)
        
        # Calculate metrics
        metrics = self._calculate_metrics(image, zones, defects)
        
        # Create report
        report = {
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'pass': is_pass,
            'total_defects': len(defects),
            'defects': [d.to_dict() for d in defects],
            'metrics': metrics,
            'zones': {
                'core': {'area': int(np.sum(zones.get('core', 0) > 0))},
                'cladding': {'area': int(np.sum(zones.get('cladding', 0) > 0))},
                'ferrule': {'area': int(np.sum(zones.get('ferrule', 0) > 0))}
            }
        }
        
        return {
            'report': report,
            'visualizations': visualizations,
            'pass': is_pass
        }
    
    def _determine_pass_fail(self, defects: List[Defect]) -> bool:
        """Determine if fiber passes inspection"""
        # Configurable pass/fail criteria
        max_defects = 5
        max_severity = 0.7
        critical_types = ['fiber_damage', 'crack']
        
        if len(defects) > max_defects:
            return False
        
        for defect in defects:
            if defect.severity > max_severity:
                return False
            if defect.type in critical_types:
                return False
        
        return True
    
    def _generate_visualizations(self, image: np.ndarray, 
                               zones: Dict[str, np.ndarray], 
                               defects: List[Defect]) -> Dict[str, np.ndarray]:
        """Generate all visualization outputs"""
        visualizations = {}
        
        # Enhanced defect overlay with measurements
        overlay = image.copy()
        
        # Calculate zone centers and diameters
        zone_info = self._calculate_zone_properties(zones)
        
        # Draw zone boundaries and centers
        for zone_name, info in zone_info.items():
            if info is None:
                continue
                
            color = {
                'core': (0, 255, 0),
                'cladding': (255, 255, 0),
                'ferrule': (255, 0, 0)
            }.get(zone_name, (255, 255, 255))
            
            # Draw center
            cv2.circle(overlay, info['center'], 3, color, -1)
            cv2.circle(overlay, info['center'], info['radius'], color, 1)
            
            # Add diameter label
            label = f"{zone_name}: Ø{info['diameter']}px"
            cv2.putText(overlay, label, 
                       (info['center'][0] - 60, info['center'][1] - info['radius'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Calculate and display concentricity
        if 'core' in zone_info and 'cladding' in zone_info:
            core_center = zone_info['core']['center']
            clad_center = zone_info['cladding']['center']
            offset = np.sqrt((core_center[0] - clad_center[0])**2 + 
                           (core_center[1] - clad_center[1])**2)
            
            # Draw concentricity line
            cv2.line(overlay, core_center, clad_center, (0, 255, 255), 2)
            
            # Add concentricity label
            cv2.putText(overlay, f"Concentricity: {offset:.1f}px",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw defects
        for defect in defects:
            color = self.config.visualization.defect_colors.get(
                defect.type, (255, 0, 255)
            )
            x, y, w, h = defect.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{defect.type} ({defect.severity:.2f})"
            cv2.putText(overlay, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        visualizations['defect_overlay'] = overlay
        
        # Zone visualization
        zone_vis = image.copy()
        zone_colors = {
            'core': (0, 255, 0),
            'cladding': (255, 255, 0),
            'ferrule': (255, 0, 0)
        }
        
        for zone_name, mask in zones.items():
            if mask is not None:
                colored = np.zeros_like(image)
                colored[mask > 0] = zone_colors.get(zone_name, (255, 255, 255))
                zone_vis = cv2.addWeighted(zone_vis, 0.7, colored, 0.3, 0)
        
        visualizations['zone_overlay'] = zone_vis
        
        # Heatmap
        if self.config.visualization.generate_heatmaps:
            h, w = image.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)
            
            for defect in defects:
                x, y = defect.location
                size = max(defect.bbox[2], defect.bbox[3])
                
                # Create Gaussian
                y_coords, x_coords = np.ogrid[:h, :w]
                mask = ((x_coords - x)**2 + (y_coords - y)**2) <= size**2
                heatmap[mask] += defect.severity
            
            # Normalize and colorize
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            visualizations['defect_heatmap'] = heatmap_color
        
        return visualizations
    
    def _calculate_zone_properties(self, zones: Dict[str, np.ndarray]) -> Dict:
        """Calculate center and diameter for each zone"""
        zone_info = {}
        
        for zone_name, mask in zones.items():
            if mask is None or np.sum(mask) == 0:
                zone_info[zone_name] = None
                continue
                
            # Find contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                zone_info[zone_name] = None
                continue
                
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate center and radius
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            zone_info[zone_name] = {
                'center': center,
                'radius': radius,
                'diameter': radius * 2
            }
        
        return zone_info
    
    def _calculate_metrics(self, image: np.ndarray, 
                         zones: Dict[str, np.ndarray], 
                         defects: List[Defect]) -> Dict:
        """Calculate quality metrics"""
        metrics = {
            'total_area': image.shape[0] * image.shape[1],
            'defect_density': len(defects) / (image.shape[0] * image.shape[1]) * 10000,
            'average_severity': np.mean([d.severity for d in defects]) if defects else 0,
            'zone_defects': {},
            'zone_measurements': {}
        }
        
        # Calculate zone properties
        zone_info = self._calculate_zone_properties(zones)
        
        # Add zone measurements
        for zone_name, info in zone_info.items():
            if info is not None:
                metrics['zone_measurements'][zone_name] = {
                    'diameter_pixels': info['diameter'],
                    'center': info['center'],
                    'area_pixels': int(np.sum(zones.get(zone_name, 0) > 0))
                }
        
        # Calculate concentricity
        if 'core' in zone_info and 'cladding' in zone_info:
            if zone_info['core'] and zone_info['cladding']:
                core_center = zone_info['core']['center']
                clad_center = zone_info['cladding']['center']
                offset = np.sqrt((core_center[0] - clad_center[0])**2 + 
                               (core_center[1] - clad_center[1])**2)
                metrics['concentricity_offset_pixels'] = offset
                metrics['concentricity_percentage'] = (offset / zone_info['cladding']['radius']) * 100
        
        # Count defects per zone
        for zone_name in zones:
            zone_defects = [d for d in defects if d.zone == zone_name]
            metrics['zone_defects'][zone_name] = {
                'count': len(zone_defects),
                'types': list(set(d.type for d in zone_defects))
            }
        
        return metrics


class EnhancedApplication:
    """Main application orchestrator"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.config = self.config_manager.get_config()
        
        # Initialize components
        self.processor = EnhancedProcessor()
        self.separator = EnhancedSeparator()
        self.detector = EnhancedDetector()
        self.data_acquisition = DataAcquisition()
        
        # Setup output directory
        self._setup_output_directory()
    
    def _setup_output_directory(self):
        """Create output directory structure"""
        self.config.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = ['passed', 'failed', 'reports', 'visualizations']
        for subdir in subdirs:
            (self.config.output_dir / subdir).mkdir(exist_ok=True)
    
    def run(self):
        """Main application entry point"""
        logger.info("Starting Enhanced Fiber Optic Defect Detection")
        
        while True:
            mode = self._select_mode()
            
            if mode == 'batch':
                self._run_batch_mode()
            elif mode == 'single':
                self._run_single_mode()
            elif mode == 'realtime':
                self._run_realtime_mode()
            elif mode == 'test':
                self._run_test_mode()
            elif mode == 'config':
                self._reconfigure()
            elif mode == 'quit':
                logger.info("Exiting application")
                break
            
            print("\n" + "="*50 + "\n")
    
    def _select_mode(self) -> str:
        """Select operation mode"""
        print("\n=== Fiber Optic Defect Detection ===")
        print("1. Batch processing")
        print("2. Single image")
        print("3. Real-time camera")
        print("4. Run tests")
        print("5. Reconfigure")
        print("6. Quit")
        
        choice = input("\nSelect mode (1-6): ").strip()
        
        mode_map = {
            '1': 'batch',
            '2': 'single',
            '3': 'realtime',
            '4': 'test',
            '5': 'config',
            '6': 'quit'
        }
        
        return mode_map.get(choice, 'quit')
    
    def _run_batch_mode(self):
        """Process multiple images"""
        print("\n--- Batch Processing ---")
        
        # Get input directory
        input_path = input("Enter input directory path (or press Enter for default): ").strip()
        if not input_path:
            input_path = self.config.input_dir
        else:
            input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_path}")
            return
        
        # Find images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No images found in {input_path}")
            return
        
        print(f"\nFound {len(image_files)} images")
        confirm = input("Process all? [yes/no]: ").strip().lower()
        
        if confirm not in ['yes', 'y']:
            return
        
        # Process images
        self._process_batch(image_files)
    
    def _run_single_mode(self):
        """Process single image"""
        print("\n--- Single Image Processing ---")
        
        image_path = input("Enter image path: ").strip()
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return
        
        # Process image
        result = self._process_image(image_path)
        
        # Display results
        if result:
            print(f"\nResult: {'PASS' if result['pass'] else 'FAIL'}")
            print(f"Defects found: {result['report']['total_defects']}")
            
            # Show visualizations
            if input("\nShow visualizations? [yes/no]: ").strip().lower() in ['yes', 'y']:
                self._display_visualizations(result['visualizations'])
    
    def _run_realtime_mode(self):
        """Run real-time processing"""
        print("\n--- Real-time Processing ---")
        
        # Check if enabled
        if not self.config.processing.realtime_enabled:
            enable = input("Real-time not enabled. Enable now? [yes/no]: ").strip().lower()
            if enable in ['yes', 'y']:
                self.config.processing.realtime_enabled = True
            else:
                return
        
        # Get camera source
        source = input("Enter camera index (default 0): ").strip()
        source = int(source) if source else 0
        
        print("\nStarting real-time processing...")
        print("Press 'q' to quit, 's' to save frame, 'p' to pause/resume")
        
        # Create and run processor
        processor = RealtimeProcessor()
        try:
            processor.start(source)
            processor.display_loop()
        except Exception as e:
            logger.error(f"Real-time processing failed: {e}")
        finally:
            processor.stop()
    
    def _run_test_mode(self):
        """Run system tests"""
        print("\n--- Running Tests ---")
        print("1. Run unit tests")
        print("2. Process test image (img(303).jpg)")
        
        choice = input("\nSelect test mode (1-2): ").strip()
        
        if choice == '1':
            # Import and run tests
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-v", "tests/"],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent
                )
                
                print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                    
            except Exception as e:
                logger.error(f"Failed to run tests: {e}")
                print("Make sure pytest is installed: pip install pytest")
        
        elif choice == '2':
            # Process test image
            test_image_path = Path(__file__).parent / "test_image" / "img(303).jpg"
            if not test_image_path.exists():
                logger.error(f"Test image not found: {test_image_path}")
                return
                
            print(f"\nProcessing test image: {test_image_path}")
            result = self._process_image(test_image_path)
            
            if result:
                print(f"\nResult: {'PASS' if result['pass'] else 'FAIL'}")
                print(f"Defects found: {result['report']['total_defects']}")
                
                # Show detailed metrics
                print("\n=== Detailed Metrics ===")
                metrics = result['report']['metrics']
                print(f"Total area: {metrics['total_area']} pixels")
                print(f"Defect density: {metrics['defect_density']:.2f}")
                print(f"Average severity: {metrics['average_severity']:.2f}")
                
                # Zone measurements
                print("\n=== Zone Measurements ===")
                for zone_name, measurements in metrics.get('zone_measurements', {}).items():
                    print(f"{zone_name}:")
                    print(f"  - Diameter: {measurements['diameter_pixels']} pixels")
                    print(f"  - Center: {measurements['center']}")
                    print(f"  - Area: {measurements['area_pixels']} pixels")
                
                # Concentricity
                if 'concentricity_offset_pixels' in metrics:
                    print("\n=== Concentricity ===")
                    print(f"Offset: {metrics['concentricity_offset_pixels']:.2f} pixels")
                    print(f"Percentage: {metrics['concentricity_percentage']:.2f}%")
                
                # Zone defects
                print("\n=== Zone Defects ===")
                for zone_name, defect_info in metrics['zone_defects'].items():
                    print(f"{zone_name}: {defect_info['count']} defects")
                    if defect_info['types']:
                        print(f"  Types: {', '.join(defect_info['types'])}")
                
                # Show visualizations
                if input("\nShow visualizations? [yes/no]: ").strip().lower() in ['yes', 'y']:
                    self._display_visualizations(result['visualizations'])
    
    def _reconfigure(self):
        """Reconfigure system"""
        print("\n--- Reconfiguration ---")
        
        # Create new config manager
        new_config = get_config_manager()
        
        # Update our config reference
        self.config = new_config.get_config()
        
        # Reinitialize components with new config
        self.processor = EnhancedProcessor()
        self.separator = EnhancedSeparator()
        self.detector = EnhancedDetector()
        
        print("Configuration updated!")
    
    def _process_batch(self, image_files: List[Path]):
        """Process multiple images"""
        results_summary = {
            'passed': 0,
            'failed': 0,
            'errors': 0
        }
        
        progress = ProgressLogger("batch_processing", len(image_files))
        
        for image_path in image_files:
            try:
                result = self._process_image(image_path)
                
                if result:
                    if result['pass']:
                        results_summary['passed'] += 1
                    else:
                        results_summary['failed'] += 1
                else:
                    results_summary['errors'] += 1
                
                progress.update(1, f"Processed {image_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results_summary['errors'] += 1
        
        progress.complete()
        
        # Print summary
        print("\n=== Batch Processing Complete ===")
        print(f"Passed: {results_summary['passed']}")
        print(f"Failed: {results_summary['failed']}")
        print(f"Errors: {results_summary['errors']}")
        print(f"Total: {len(image_files)}")
    
    def _process_image(self, image_path: Path) -> Optional[Dict]:
        """Process single image through complete pipeline"""
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Step 1: Generate variations
            variations = self.processor.process_image(image_path)
            logger.info(f"Generated {len(variations)} variations")
            
            # Step 2: Separate zones
            zones = self.separator.separate_zones(image, variations)
            logger.info(f"Separated into {len(zones)} zones")
            
            # Step 3: Detect defects
            defects = self.detector.detect_defects(image, zones, variations)
            logger.info(f"Detected {len(defects)} defects")
            
            # Step 4: Generate final results
            result = self.data_acquisition.process_results(image_path, image, zones, defects)
            
            # Save results
            self._save_results(image_path, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed for {image_path}: {e}")
            return None
    
    def _save_results(self, image_path: Path, result: Dict):
        """Save processing results"""
        # Determine output directory
        if result['pass']:
            output_base = self.config.output_dir / 'passed'
        else:
            output_base = self.config.output_dir / 'failed'
        
        # Create unique output name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{image_path.stem}_{timestamp}"
        
        # Save report
        report_path = self.config.output_dir / 'reports' / f"{base_name}.json"
        with open(report_path, 'w') as f:
            json.dump(result['report'], f, indent=2)
        
        # Save visualizations
        if self.config.visualization.generate_overlays:
            for vis_name, vis_image in result['visualizations'].items():
                vis_path = self.config.output_dir / 'visualizations' / f"{base_name}_{vis_name}.jpg"
                cv2.imwrite(str(vis_path), vis_image)
        
        # Copy original to pass/fail directory
        shutil.copy2(image_path, output_base / f"{base_name}_original.jpg")
        
        logger.info(f"Results saved: {base_name}")
    
    def _display_visualizations(self, visualizations: Dict[str, np.ndarray]):
        """Display visualization images"""
        for name, image in visualizations.items():
            cv2.imshow(name, image)
        
        print("\nPress any key to close visualizations...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     Enhanced Fiber Optic Defect Detection System          ║
    ║                                                           ║
    ║     - ML-Powered Analysis                                 ║
    ║     - Real-time Processing                                ║
    ║     - No argparse Required                                ║
    ║     - Full Debug Logging                                  ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        app = EnhancedApplication()
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.exception(f"Application error: {e}")
        print(f"\nError: {e}")
        print("Check logs for details")


if __name__ == "__main__":
    main()