#!/usr/bin/env python3
"""
Comprehensive Ferrule Image Analysis and Reorganization Script
=============================================================
This script analyzes all ferrule images in the current directory,
detects defects, and reorganizes them according to the established
naming convention and directory structure.
"""

import os
import cv2
import numpy as np
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ferrule_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FerruleImageAnalyzer:
    """Analyzes ferrule images for defects and quality"""
    
    def __init__(self):
        self.defect_types = {
            'scratch': 0,
            'clean': 1,
            'contaminated': 2,
            'unknown': 3
        }
        
    def detect_scratches(self, img: np.ndarray) -> Tuple[bool, List[np.ndarray]]:
        """Detect linear scratches in the image"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Apply edge detection to find scratches
        edges = cv2.Canny(gray, 50, 150)
        
        # Use HoughLinesP to detect linear features
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        has_scratches = lines is not None and len(lines) > 2
        scratch_lines = lines if lines is not None else []
        
        return has_scratches, scratch_lines
    
    def detect_contamination(self, img: np.ndarray) -> Tuple[bool, float]:
        """Detect contamination or spots in the image"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Find the ferrule region (bright area)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Get the largest contour (ferrule)
        ferrule_contour = max(contours, key=cv2.contourArea)
        
        # Create mask for ferrule region
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [ferrule_contour], -1, 255, -1)
        
        # Extract ferrule region
        ferrule_region = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Calculate statistics
        mean_intensity = np.mean(ferrule_region[mask > 0])
        std_intensity = np.std(ferrule_region[mask > 0])
        
        # High standard deviation indicates contamination
        contamination_score = std_intensity / (mean_intensity + 1e-6)
        has_contamination = contamination_score > 0.15
        
        return has_contamination, contamination_score
    
    def analyze_image(self, img_path: str) -> Dict:
        """Comprehensive analysis of a ferrule image"""
        logger.info(f"Analyzing image: {img_path}")
        
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            return None
            
        # Detect various defects
        has_scratches, scratch_lines = self.detect_scratches(img)
        has_contamination, contamination_score = self.detect_contamination(img)
        
        # Determine primary defect type
        if has_scratches:
            defect_type = 'scratch'
            defect_severity = min(len(scratch_lines) / 10.0, 1.0)  # Normalize to 0-1
        elif has_contamination:
            defect_type = 'contaminated'
            defect_severity = contamination_score
        else:
            defect_type = 'clean'
            defect_severity = 0.0
            
        # Calculate image hash for unique identification
        img_hash = hashlib.md5(img.tobytes()).hexdigest()[:8]
        
        analysis_result = {
            'original_path': img_path,
            'filename': os.path.basename(img_path),
            'defect_type': defect_type,
            'defect_severity': float(defect_severity),
            'has_scratches': bool(has_scratches),
            'scratch_count': len(scratch_lines) if has_scratches else 0,
            'has_contamination': bool(has_contamination),
            'contamination_score': float(contamination_score),
            'image_hash': img_hash,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Analysis complete: {defect_type} (severity: {defect_severity:.3f})")
        return analysis_result


class FerruleDatasetOrganizer:
    """Organizes ferrule images into proper dataset structure"""
    
    def __init__(self, base_dir: str = '.'):
        self.base_dir = Path(base_dir)
        self.analyzer = FerruleImageAnalyzer()
        self.analysis_results = []
        
    def create_directory_structure(self):
        """Create the dataset directory structure"""
        dirs = [
            'dataset/train/scratch',
            'dataset/train/clean',
            'dataset/train/contaminated',
            'dataset/val/scratch',
            'dataset/val/clean',
            'dataset/val/contaminated',
            'dataset/test/scratch',
            'dataset/test/clean',
            'dataset/test/contaminated',
            'dataset/masks',
            'dataset/analysis_reports'
        ]
        
        for dir_path in dirs:
            (self.base_dir / dir_path).mkdir(parents=True, exist_ok=True)
            
        logger.info("Created dataset directory structure")
        
    def analyze_all_images(self):
        """Analyze all PNG images in the current directory"""
        image_files = list(self.base_dir.glob('*.png'))
        logger.info(f"Found {len(image_files)} PNG images to analyze")
        
        for img_path in image_files:
            # Skip mask images
            if 'mask' in img_path.name.lower() or 'region' in img_path.name.lower():
                logger.info(f"Skipping mask/region image: {img_path.name}")
                continue
                
            result = self.analyzer.analyze_image(str(img_path))
            if result:
                self.analysis_results.append(result)
                
        # Save analysis results
        with open(self.base_dir / 'ferrule_analysis_results.json', 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
            
        logger.info(f"Analysis complete. Results saved to ferrule_analysis_results.json")
        
    def generate_new_filename(self, result: Dict, split: str, index: int) -> str:
        """Generate new filename according to naming convention"""
        defect_type = result['defect_type']
        img_hash = result['image_hash']
        
        # Format: {class}_{split}_{index:04d}_{hash}.png
        new_name = f"{defect_type}_{split}_{index:04d}_{img_hash}.png"
        return new_name
        
    def reorganize_images(self):
        """Reorganize images based on analysis results"""
        if not self.analysis_results:
            logger.error("No analysis results available. Run analyze_all_images first.")
            return
            
        # Group images by defect type
        by_defect = {
            'scratch': [],
            'clean': [],
            'contaminated': []
        }
        
        for result in self.analysis_results:
            defect_type = result['defect_type']
            if defect_type in by_defect:
                by_defect[defect_type].append(result)
                
        # Sort by severity within each group
        for defect_type in by_defect:
            by_defect[defect_type].sort(key=lambda x: x['defect_severity'], reverse=True)
            
        # Distribute images across train/val/test splits
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        
        reorganization_log = []
        
        for defect_type, images in by_defect.items():
            n_images = len(images)
            n_train = int(n_images * split_ratios['train'])
            n_val = int(n_images * split_ratios['val'])
            n_test = n_images - n_train - n_val
            
            # Ensure at least one image in each split if possible
            if n_images >= 3:
                n_train = max(1, n_train)
                n_val = max(1, n_val)
                n_test = max(1, n_test)
                
            # Assign images to splits
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Copy images to new locations
            for split, split_images in [('train', train_images), 
                                       ('val', val_images), 
                                       ('test', test_images)]:
                for idx, img_data in enumerate(split_images, 1):
                    src_path = Path(img_data['original_path'])
                    new_filename = self.generate_new_filename(img_data, split, idx)
                    dst_path = self.base_dir / 'dataset' / split / defect_type / new_filename
                    
                    try:
                        shutil.copy2(src_path, dst_path)
                        logger.info(f"Copied {src_path.name} -> {dst_path}")
                        
                        reorganization_log.append({
                            'original': str(src_path),
                            'new_path': str(dst_path),
                            'defect_type': defect_type,
                            'split': split,
                            'severity': img_data['defect_severity']
                        })
                    except Exception as e:
                        logger.error(f"Failed to copy {src_path}: {e}")
                        
        # Save reorganization log
        with open(self.base_dir / 'ferrule_reorganization_log.json', 'w') as f:
            json.dump(reorganization_log, f, indent=2)
            
        logger.info("Reorganization complete. Log saved to ferrule_reorganization_log.json")
        
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report"""
        if not self.analysis_results:
            logger.error("No analysis results available.")
            return
            
        report_lines = [
            "# Ferrule Image Analysis Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Summary",
            f"Total images analyzed: {len(self.analysis_results)}",
            "\n## Defect Distribution"
        ]
        
        # Count defects
        defect_counts = {'scratch': 0, 'clean': 0, 'contaminated': 0}
        severity_stats = {'scratch': [], 'clean': [], 'contaminated': []}
        
        for result in self.analysis_results:
            defect_type = result['defect_type']
            defect_counts[defect_type] += 1
            severity_stats[defect_type].append(result['defect_severity'])
            
        # Add statistics
        for defect_type, count in defect_counts.items():
            percentage = (count / len(self.analysis_results)) * 100
            report_lines.append(f"- {defect_type.capitalize()}: {count} images ({percentage:.1f}%)")
            
            if severity_stats[defect_type]:
                avg_severity = np.mean(severity_stats[defect_type])
                max_severity = np.max(severity_stats[defect_type])
                report_lines.append(f"  - Average severity: {avg_severity:.3f}")
                report_lines.append(f"  - Maximum severity: {max_severity:.3f}")
                
        # Add detailed analysis
        report_lines.extend([
            "\n## Detailed Analysis",
            "\n### Scratch Defects"
        ])
        
        scratch_images = [r for r in self.analysis_results if r['defect_type'] == 'scratch']
        scratch_images.sort(key=lambda x: x['defect_severity'], reverse=True)
        
        for img in scratch_images[:5]:  # Top 5 most severe
            report_lines.append(f"- {img['filename']}: {img['scratch_count']} scratches detected (severity: {img['defect_severity']:.3f})")
            
        report_lines.extend([
            "\n### Contaminated Images"
        ])
        
        contaminated_images = [r for r in self.analysis_results if r['defect_type'] == 'contaminated']
        contaminated_images.sort(key=lambda x: x['contamination_score'], reverse=True)
        
        for img in contaminated_images[:5]:  # Top 5 most contaminated
            report_lines.append(f"- {img['filename']}: contamination score {img['contamination_score']:.3f}")
            
        # Save report
        report_path = self.base_dir / 'ferrule_analysis_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
            
        logger.info(f"Analysis report saved to {report_path}")


def main():
    """Main execution function"""
    logger.info("Starting Ferrule Image Analysis and Reorganization")
    
    # Create organizer
    organizer = FerruleDatasetOrganizer()
    
    # Step 1: Create directory structure
    organizer.create_directory_structure()
    
    # Step 2: Analyze all images
    organizer.analyze_all_images()
    
    # Step 3: Generate analysis report
    organizer.generate_analysis_report()
    
    # Step 4: Reorganize images
    print("\nAnalysis complete. Proceeding with automatic reorganization.")
    organizer.reorganize_images()
    logger.info("Dataset reorganization complete!")
        
    logger.info("Process complete. Check the log files for details.")


if __name__ == "__main__":
    main()