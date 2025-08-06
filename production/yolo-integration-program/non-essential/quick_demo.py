#!/usr/bin/env python3
"""
Quick Demo for Fiber Optic Analysis System
Shows basic functionality without heavy computational features.
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

def basic_fiber_analysis(image_path):
    """Perform basic fiber analysis without heavy computations."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    print(f"Analyzing {image_path}...")
    print(f"Image shape: {image.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Basic statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    min_intensity = np.min(gray)
    max_intensity = np.max(gray)
    
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
        
        # Get bounding circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
    else:
        area = 0
        circularity = 0
        center = None
        radius = 0
    
    # Create results
    results = {
        'image_path': image_path,
        'timestamp': datetime.now().isoformat(),
        'image_shape': image.shape,
        'analysis': {
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'min_intensity': float(min_intensity),
            'max_intensity': float(max_intensity),
            'edge_density': float(edge_density),
            'largest_area': float(area),
            'circularity': float(circularity),
            'center': center,
            'radius': radius
        }
    }
    
    return results, image, gray, edges

def create_visualization(image, gray, edges, results):
    """Create visualization of analysis results."""
    # Create display image
    display = image.copy()
    
    # Add analysis results
    analysis = results['analysis']
    cv2.putText(display, f"Mean Intensity: {analysis['mean_intensity']:.1f}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Std Intensity: {analysis['std_intensity']:.1f}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Edge Density: {analysis['edge_density']:.3f}", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Circularity: {analysis['circularity']:.3f}", 
               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw center and radius if available
    if analysis['center'] and analysis['radius'] > 0:
        cv2.circle(display, analysis['center'], analysis['radius'], (0, 255, 0), 2)
        cv2.circle(display, analysis['center'], 5, (0, 0, 255), -1)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(display, timestamp, (10, display.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return display

def save_results(results, output_dir="quick_demo_output"):
    """Save analysis results."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save JSON results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"analysis_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Failed to save results: {e}")
        return None

def main():
    """Main demo function."""
    print("Quick Fiber Analysis Demo")
    print("=" * 30)
    
    # Test image
    test_image = "good.bmp"
    
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found!")
        print("Please ensure good.bmp is in the current directory.")
        return
    
    # Perform analysis
    results = basic_fiber_analysis(test_image)
    
    if results:
        results_dict, image, gray, edges = results
        
        print("\nAnalysis Results:")
        print("-" * 20)
        
        analysis = results_dict['analysis']
        print(f"Mean Intensity: {analysis['mean_intensity']:.1f}")
        print(f"Standard Deviation: {analysis['std_intensity']:.1f}")
        print(f"Min/Max Intensity: {analysis['min_intensity']:.1f}/{analysis['max_intensity']:.1f}")
        print(f"Edge Density: {analysis['edge_density']:.3f}")
        print(f"Largest Area: {analysis['largest_area']:.0f} pixels")
        print(f"Circularity: {analysis['circularity']:.3f}")
        
        if analysis['center']:
            print(f"Center: {analysis['center']}")
            print(f"Radius: {analysis['radius']} pixels")
        
        # Quality assessment
        if analysis['circularity'] > 0.8:
            print("\n✅ High circularity detected - likely a good fiber")
        elif analysis['circularity'] > 0.5:
            print("\n⚠️  Moderate circularity - fiber may have issues")
        else:
            print("\n❌ Low circularity - fiber likely defective")
        
        # Save results
        output_dir = save_results(results_dict)
        
        # Create and show visualizations
        print("\nCreating visualizations...")
        
        # Original with analysis
        viz1 = create_visualization(image, gray, edges, results_dict)
        
        # Grayscale
        viz2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(viz2, "Grayscale", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Edges
        viz3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.putText(viz3, "Edge Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualizations
        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(output_dir / f"analysis_{timestamp}.jpg"), viz1)
            cv2.imwrite(str(output_dir / f"grayscale_{timestamp}.jpg"), viz2)
            cv2.imwrite(str(output_dir / f"edges_{timestamp}.jpg"), viz3)
            print(f"Visualizations saved to: {output_dir}")
        
        # Show visualizations
        print("\nPress any key to close each window...")
        
        cv2.imshow('Fiber Analysis', viz1)
        cv2.waitKey(0)
        
        cv2.imshow('Grayscale', viz2)
        cv2.waitKey(0)
        
        cv2.imshow('Edge Detection', viz3)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        
        print("\nDemo completed successfully!")
    
    else:
        print("Analysis failed!")

if __name__ == "__main__":
    main() 