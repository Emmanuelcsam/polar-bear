#!/usr/bin/env python3
"""
Demo script for the Fiber Optic Analysis System
Shows system capabilities using the existing good.bmp image.
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import logging
from datetime import datetime

# Import our modules
from detection import OmniFiberAnalyzer, OmniConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FiberAnalysisDemo:
    """Demo class for fiber optic analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fiber_analyzer = None
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """Initialize the fiber analyzer."""
        try:
            # Create configuration for fiber analyzer
            fiber_config = OmniConfig(
                confidence_threshold=0.3,
                anomaly_threshold_multiplier=2.5,
                enable_visualization=True
            )
            
            self.fiber_analyzer = OmniFiberAnalyzer(fiber_config)
            
            # Build a simple reference model from the test image if available
            if os.path.exists("good.bmp"):
                self.logger.info("Building reference model from good.bmp...")
                self.fiber_analyzer._build_minimal_reference("good.bmp")
                self.logger.info("Reference model built successfully")
            
            self.logger.info("Fiber analyzer initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize fiber analyzer: {e}")
            self.fiber_analyzer = None
    
    def analyze_image(self, image_path):
        """Analyze a single image."""
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            return None
        
        self.logger.info(f"Analyzing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            return None
        
        # Perform basic analysis
        basic_results = self._basic_image_analysis(image)
        
        # Perform fiber analysis if analyzer is available
        fiber_results = None
        if self.fiber_analyzer:
            try:
                fiber_results = self.fiber_analyzer.detect_anomalies_comprehensive(image_path)
            except Exception as e:
                self.logger.warning(f"Fiber analysis failed: {e}")
        
        return {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'basic_analysis': basic_results,
            'fiber_analysis': fiber_results
        }
    
    def _basic_image_analysis(self, image):
        """Perform basic image analysis."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Basic statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze largest contour
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            else:
                area = 0
                circularity = 0
            
            return {
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'edge_density': float(edge_density),
                'largest_area': float(area),
                'circularity': float(circularity)
            }
        except Exception as e:
            self.logger.error(f"Basic analysis error: {e}")
            return {}
    
    def create_visualization(self, image_path, results):
        """Create visualization of analysis results."""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Create display image
        display = image.copy()
        
        # Add basic analysis results
        if results.get('basic_analysis'):
            basic = results['basic_analysis']
            cv2.putText(display, f"Mean Intensity: {basic.get('mean_intensity', 0):.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Edge Density: {basic.get('edge_density', 0):.3f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Circularity: {basic.get('circularity', 0):.3f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add fiber analysis results
        if results.get('fiber_analysis') and results['fiber_analysis']:
            fiber_results = results['fiber_analysis']
            if 'verdict' in fiber_results:
                verdict = fiber_results['verdict']
                status = "ANOMALY" if verdict['is_anomalous'] else "NORMAL"
                color = (0, 0, 255) if verdict['is_anomalous'] else (0, 255, 0)
                cv2.putText(display, f"Fiber Status: {status}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(display, f"Confidence: {verdict['confidence']:.2f}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display, timestamp, (10, display.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display
    
    def save_results(self, results, output_dir="demo_output"):
        """Save analysis results."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save JSON results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_path / f"analysis_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save visualization
            if results.get('image_path'):
                viz = self.create_visualization(results['image_path'], results)
                if viz is not None:
                    viz_file = output_path / f"visualization_{timestamp}.jpg"
                    cv2.imwrite(str(viz_file), viz)
            
            self.logger.info(f"Results saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return None

def main():
    """Main demo function."""
    print("Fiber Optic Analysis System - Demo")
    print("=" * 40)
    
    # Create demo instance
    demo = FiberAnalysisDemo()
    
    # Test image
    test_image = "good.bmp"
    
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found!")
        print("Please ensure good.bmp is in the current directory.")
        return
    
    print(f"Analyzing {test_image}...")
    
    # Perform analysis
    results = demo.analyze_image(test_image)
    
    if results:
        print("\nAnalysis Results:")
        print("-" * 20)
        
        # Display basic analysis
        if results.get('basic_analysis'):
            basic = results['basic_analysis']
            print(f"Mean Intensity: {basic.get('mean_intensity', 0):.1f}")
            print(f"Standard Deviation: {basic.get('std_intensity', 0):.1f}")
            print(f"Edge Density: {basic.get('edge_density', 0):.3f}")
            print(f"Largest Area: {basic.get('largest_area', 0):.0f} pixels")
            print(f"Circularity: {basic.get('circularity', 0):.3f}")
        
        # Display fiber analysis
        if results.get('fiber_analysis') and results['fiber_analysis']:
            fiber_results = results['fiber_analysis']
            if 'verdict' in fiber_results:
                verdict = fiber_results['verdict']
                status = "ANOMALY" if verdict['is_anomalous'] else "NORMAL"
                print(f"\nFiber Status: {status}")
                print(f"Confidence: {verdict['confidence']:.3f}")
                
                if verdict['is_anomalous']:
                    print("⚠️  Anomaly detected in fiber!")
                else:
                    print("✅ Fiber appears normal")
        
        # Save results
        output_dir = demo.save_results(results)
        if output_dir:
            print(f"\nResults saved to: {output_dir}")
        
        # Show visualization
        viz = demo.create_visualization(test_image, results)
        if viz is not None:
            cv2.imshow('Fiber Analysis Demo', viz)
            print("\nPress any key to close the visualization...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    else:
        print("Analysis failed!")

if __name__ == "__main__":
    main() 