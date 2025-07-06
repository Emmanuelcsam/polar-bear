#!/usr/bin/env python3
"""
Advanced Ferrule Analysis using APS Framework
============================================
This script leverages the Automated Processing Studio (APS) framework
for comprehensive ferrule image analysis with multiple detection methods.
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Add the studio-tools path to sys.path
sys.path.append('/home/jarvis/Documents/GitHub/polar-bear/studio-tools')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('advanced_ferrule_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdvancedFerruleAnalyzer:
    """Advanced analyzer using multiple detection techniques"""
    
    def __init__(self):
        self.detection_methods = {
            'gradient': self.gradient_based_detection,
            'morphological': self.morphological_detection,
            'frequency': self.frequency_based_detection,
            'statistical': self.statistical_anomaly_detection,
            'edge': self.edge_based_detection
        }
        
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Advanced preprocessing for ferrule images"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return denoised
    
    def extract_ferrule_region(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the ferrule region and create a mask"""
        gray = self.preprocess_image(img)
        
        # Use adaptive thresholding to find ferrule
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return gray, np.ones_like(gray) * 255
            
        # Find the circular ferrule contour
        best_contour = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Skip small contours
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > best_circularity and circularity > 0.7:
                best_circularity = circularity
                best_contour = contour
                
        # Create mask
        mask = np.zeros_like(gray)
        if best_contour is not None:
            cv2.drawContours(mask, [best_contour], -1, 255, -1)
            
        return gray, mask
    
    def gradient_based_detection(self, img: np.ndarray) -> Dict:
        """Detect defects using gradient analysis"""
        gray, mask = self.extract_ferrule_region(img)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply mask
        grad_mag_masked = cv2.bitwise_and(grad_mag.astype(np.uint8), 
                                        grad_mag.astype(np.uint8), mask=mask)
        
        # Detect high gradient regions (potential defects)
        threshold = np.percentile(grad_mag_masked[mask > 0], 95)
        defect_mask = grad_mag_masked > threshold
        
        # Calculate metrics
        defect_pixels = np.sum(defect_mask)
        total_pixels = np.sum(mask > 0)
        defect_ratio = defect_pixels / (total_pixels + 1e-6)
        
        return {
            'method': 'gradient',
            'defect_ratio': defect_ratio,
            'max_gradient': np.max(grad_mag_masked),
            'mean_gradient': np.mean(grad_mag_masked[mask > 0]),
            'defect_mask': defect_mask
        }
    
    def morphological_detection(self, img: np.ndarray) -> Dict:
        """Detect defects using morphological operations"""
        gray, mask = self.extract_ferrule_region(img)
        
        # Apply morphological operations to detect defects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Top-hat transform to detect bright spots
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Black-hat transform to detect dark spots
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine both
        defects = cv2.add(tophat, blackhat)
        defects_masked = cv2.bitwise_and(defects, defects, mask=mask)
        
        # Threshold to get defect regions
        _, defect_mask = cv2.threshold(defects_masked, 20, 255, cv2.THRESH_BINARY)
        
        # Calculate metrics
        defect_pixels = np.sum(defect_mask > 0)
        total_pixels = np.sum(mask > 0)
        defect_ratio = defect_pixels / (total_pixels + 1e-6)
        
        return {
            'method': 'morphological',
            'defect_ratio': defect_ratio,
            'tophat_mean': np.mean(tophat[mask > 0]),
            'blackhat_mean': np.mean(blackhat[mask > 0]),
            'defect_mask': defect_mask
        }
    
    def frequency_based_detection(self, img: np.ndarray) -> Dict:
        """Detect defects using frequency domain analysis"""
        gray, mask = self.extract_ferrule_region(img)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Create high-pass filter
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask for high frequencies
        high_pass_mask = np.ones((rows, cols), np.uint8)
        cv2.circle(high_pass_mask, (ccol, crow), 30, 0, -1)
        
        # Apply filter
        f_shift_filtered = f_shift * high_pass_mask
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_filtered = np.fft.ifft2(f_ishift)
        img_filtered = np.abs(img_filtered)
        
        # Normalize and apply spatial mask
        img_filtered_norm = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
        img_filtered_masked = cv2.bitwise_and(img_filtered_norm.astype(np.uint8),
                                            img_filtered_norm.astype(np.uint8), mask=mask)
        
        # Detect high-frequency regions (defects)
        _, defect_mask = cv2.threshold(img_filtered_masked, 50, 255, cv2.THRESH_BINARY)
        
        # Calculate metrics
        high_freq_energy = np.sum(img_filtered_masked[mask > 0])
        total_energy = np.sum(gray[mask > 0])
        freq_ratio = high_freq_energy / (total_energy + 1e-6)
        
        return {
            'method': 'frequency',
            'freq_ratio': freq_ratio,
            'high_freq_energy': high_freq_energy,
            'defect_mask': defect_mask
        }
    
    def statistical_anomaly_detection(self, img: np.ndarray) -> Dict:
        """Detect anomalies using statistical methods"""
        gray, mask = self.extract_ferrule_region(img)
        
        # Extract ferrule pixels
        ferrule_pixels = gray[mask > 0]
        
        # Calculate statistics
        mean_val = np.mean(ferrule_pixels)
        std_val = np.std(ferrule_pixels)
        
        # Detect outliers (anomalies)
        z_scores = np.abs((gray - mean_val) / (std_val + 1e-6))
        anomaly_mask = (z_scores > 2.5) & (mask > 0)
        
        # Calculate local statistics using sliding window
        window_size = 15
        kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_diff = np.abs(gray.astype(np.float32) - local_mean)
        
        # Local anomaly detection
        local_anomaly_mask = (local_diff > 2 * std_val) & (mask > 0)
        
        # Combine global and local anomalies
        combined_anomaly_mask = anomaly_mask | local_anomaly_mask
        
        # Calculate metrics
        anomaly_pixels = np.sum(combined_anomaly_mask)
        total_pixels = np.sum(mask > 0)
        anomaly_ratio = anomaly_pixels / (total_pixels + 1e-6)
        
        return {
            'method': 'statistical',
            'anomaly_ratio': anomaly_ratio,
            'mean_intensity': mean_val,
            'std_intensity': std_val,
            'max_z_score': np.max(z_scores[mask > 0]),
            'defect_mask': combined_anomaly_mask.astype(np.uint8) * 255
        }
    
    def edge_based_detection(self, img: np.ndarray) -> Dict:
        """Detect defects using advanced edge detection"""
        gray, mask = self.extract_ferrule_region(img)
        
        # Multi-scale edge detection
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 30, 100)
        edges3 = cv2.Canny(gray, 70, 200)
        
        # Combine multi-scale edges
        combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Apply mask
        edges_masked = cv2.bitwise_and(combined_edges, combined_edges, mask=mask)
        
        # Detect linear structures (scratches)
        lines = cv2.HoughLinesP(edges_masked, 1, np.pi/180, threshold=30,
                              minLineLength=20, maxLineGap=5)
        
        # Create scratch mask
        scratch_mask = np.zeros_like(gray)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(scratch_mask, (x1, y1), (x2, y2), 255, 2)
                
        # Detect edge density anomalies
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        edge_density = cv2.filter2D(edges_masked.astype(np.float32), -1, kernel)
        
        # High edge density indicates defects
        density_threshold = np.percentile(edge_density[mask > 0], 90)
        density_defect_mask = (edge_density > density_threshold) & (mask > 0)
        
        # Combine scratch and density defects
        combined_defect_mask = cv2.bitwise_or(scratch_mask, 
                                            density_defect_mask.astype(np.uint8) * 255)
        
        # Calculate metrics
        scratch_count = len(lines) if lines is not None else 0
        defect_pixels = np.sum(combined_defect_mask > 0)
        total_pixels = np.sum(mask > 0)
        defect_ratio = defect_pixels / (total_pixels + 1e-6)
        
        return {
            'method': 'edge',
            'defect_ratio': defect_ratio,
            'scratch_count': scratch_count,
            'mean_edge_density': np.mean(edge_density[mask > 0]),
            'defect_mask': combined_defect_mask
        }
    
    def comprehensive_analysis(self, img_path: str) -> Dict:
        """Perform comprehensive analysis using all methods"""
        logger.info(f"Performing comprehensive analysis on: {img_path}")
        
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            return None
            
        # Run all detection methods
        results = {}
        for method_name, method_func in self.detection_methods.items():
            try:
                results[method_name] = method_func(img)
                logger.info(f"{method_name} analysis complete")
            except Exception as e:
                logger.error(f"Error in {method_name} analysis: {e}")
                results[method_name] = {'error': str(e)}
                
        # Combine results for final classification
        defect_scores = {
            'scratch': 0,
            'contamination': 0,
            'clean': 0
        }
        
        # Analyze results from each method
        if 'edge' in results and 'scratch_count' in results['edge']:
            if results['edge']['scratch_count'] > 2:
                defect_scores['scratch'] += 2
                
        if 'gradient' in results and 'defect_ratio' in results['gradient']:
            if results['gradient']['defect_ratio'] > 0.05:
                defect_scores['contamination'] += 1
                
        if 'statistical' in results and 'anomaly_ratio' in results['statistical']:
            if results['statistical']['anomaly_ratio'] > 0.1:
                defect_scores['contamination'] += 1
            elif results['statistical']['anomaly_ratio'] < 0.01:
                defect_scores['clean'] += 1
                
        if 'morphological' in results and 'defect_ratio' in results['morphological']:
            if results['morphological']['defect_ratio'] > 0.05:
                defect_scores['contamination'] += 1
                
        # Determine final classification
        final_class = max(defect_scores, key=defect_scores.get)
        confidence = defect_scores[final_class] / sum(defect_scores.values()) if sum(defect_scores.values()) > 0 else 0
        
        # Calculate overall severity
        severity_scores = []
        for method_results in results.values():
            if isinstance(method_results, dict) and 'defect_ratio' in method_results:
                severity_scores.append(method_results['defect_ratio'])
                
        overall_severity = np.mean(severity_scores) if severity_scores else 0
        
        return {
            'filename': os.path.basename(img_path),
            'path': img_path,
            'classification': final_class,
            'confidence': confidence,
            'severity': overall_severity,
            'defect_scores': defect_scores,
            'method_results': results,
            'timestamp': datetime.now().isoformat()
        }


class VisualizationGenerator:
    """Generate visualization reports for analysis results"""
    
    def __init__(self, output_dir: str = 'analysis_visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_defect_overlay(self, img_path: str, analysis_result: Dict) -> np.ndarray:
        """Create an overlay visualization of detected defects"""
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        overlay = img.copy()
        
        # Combine all defect masks
        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        for method_name, method_results in analysis_result['method_results'].items():
            if isinstance(method_results, dict) and 'defect_mask' in method_results:
                mask = method_results['defect_mask']
                if mask is not None and mask.shape == combined_mask.shape:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                    
        # Apply colored overlay
        overlay[combined_mask > 0] = [0, 0, 255]  # Red for defects
        
        # Blend with original
        result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Add text annotation
        text = f"Class: {analysis_result['classification']} (conf: {analysis_result['confidence']:.2f})"
        cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        severity_text = f"Severity: {analysis_result['severity']:.3f}"
        cv2.putText(result, severity_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)
        
        return result
    
    def save_visualization(self, img_path: str, analysis_result: Dict):
        """Save visualization for an analyzed image"""
        vis_img = self.create_defect_overlay(img_path, analysis_result)
        if vis_img is not None:
            filename = Path(img_path).stem + '_visualization.png'
            output_path = self.output_dir / filename
            cv2.imwrite(str(output_path), vis_img)
            logger.info(f"Saved visualization: {output_path}")


def main():
    """Main execution function"""
    logger.info("Starting Advanced Ferrule Analysis")
    
    # Create analyzer
    analyzer = AdvancedFerruleAnalyzer()
    visualizer = VisualizationGenerator()
    
    # Find all ferrule images
    image_files = list(Path('.').glob('*ferrule*.png'))
    
    # Filter out masks and processed images
    filtered_files = []
    for f in image_files:
        if not any(skip in f.name.lower() for skip in ['mask', 'region', 'visualization']):
            filtered_files.append(f)
            
    logger.info(f"Found {len(filtered_files)} ferrule images to analyze")
    
    # Analyze all images
    all_results = []
    for img_path in filtered_files:
        result = analyzer.comprehensive_analysis(str(img_path))
        if result:
            all_results.append(result)
            visualizer.save_visualization(str(img_path), result)
            
    # Save comprehensive results
    with open('advanced_ferrule_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
        
    # Generate summary report
    generate_summary_report(all_results)
    
    logger.info("Advanced analysis complete!")


def generate_summary_report(results: List[Dict]):
    """Generate a summary report of the analysis"""
    report_lines = [
        "# Advanced Ferrule Analysis Summary Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## Overview",
        f"Total images analyzed: {len(results)}",
        "\n## Classification Results"
    ]
    
    # Count classifications
    class_counts = {'scratch': 0, 'contamination': 0, 'clean': 0}
    high_confidence = []
    high_severity = []
    
    for result in results:
        class_counts[result['classification']] += 1
        
        if result['confidence'] > 0.8:
            high_confidence.append(result)
            
        if result['severity'] > 0.1:
            high_severity.append(result)
            
    # Add classification summary
    for class_name, count in class_counts.items():
        percentage = (count / len(results)) * 100 if results else 0
        report_lines.append(f"- {class_name.capitalize()}: {count} ({percentage:.1f}%)")
        
    # Add high confidence detections
    report_lines.extend([
        "\n## High Confidence Detections (>80%)"
    ])
    
    for result in sorted(high_confidence, key=lambda x: x['confidence'], reverse=True)[:10]:
        report_lines.append(
            f"- {result['filename']}: {result['classification']} "
            f"(confidence: {result['confidence']:.2%}, severity: {result['severity']:.3f})"
        )
        
    # Add high severity defects
    report_lines.extend([
        "\n## High Severity Defects (>0.1)"
    ])
    
    for result in sorted(high_severity, key=lambda x: x['severity'], reverse=True)[:10]:
        report_lines.append(
            f"- {result['filename']}: {result['classification']} "
            f"(severity: {result['severity']:.3f})"
        )
        
    # Method effectiveness
    report_lines.extend([
        "\n## Detection Method Effectiveness"
    ])
    
    method_stats = {}
    for result in results:
        for method_name, method_results in result['method_results'].items():
            if isinstance(method_results, dict) and 'defect_ratio' in method_results:
                if method_name not in method_stats:
                    method_stats[method_name] = []
                method_stats[method_name].append(method_results['defect_ratio'])
                
    for method, ratios in method_stats.items():
        avg_ratio = np.mean(ratios)
        report_lines.append(f"- {method}: avg defect ratio = {avg_ratio:.4f}")
        
    # Save report
    with open('advanced_ferrule_analysis_summary.md', 'w') as f:
        f.write('\n'.join(report_lines))
        
    logger.info("Summary report saved to advanced_ferrule_analysis_summary.md")


if __name__ == "__main__":
    main()