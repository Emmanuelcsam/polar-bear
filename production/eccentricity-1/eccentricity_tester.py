"""
Eccentricity Tester Module
Analyzes the circularity of objects in real-time images by combining:
1. Hough circle detection
2. Intensity profile analysis
3. Gradient analysis
4. Shape analysis

This provides a comprehensive measure of how circular an object is and by how much it deviates.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import math


class EccentricityTester:
    """
    Analyzes the eccentricity/circularity of objects in images by comparing
    Hough circle fits with actual intensity profiles and gradients.
    """
    
    def __init__(self, 
                 num_radial_samples=360,
                 num_radius_samples=50,
                 gradient_threshold=20,
                 intensity_smoothing=5):
        """
        Initialize the eccentricity tester.
        
        Args:
            num_radial_samples: Number of angular samples for radial profile
            num_radius_samples: Number of radius samples for intensity profile
            gradient_threshold: Minimum gradient magnitude to consider
            intensity_smoothing: Smoothing factor for intensity profiles
        """
        self.num_radial_samples = num_radial_samples
        self.num_radius_samples = num_radius_samples
        self.gradient_threshold = gradient_threshold
        self.intensity_smoothing = intensity_smoothing
        
        # Statistics
        self.last_eccentricity_score = 0.0
        self.last_analysis_results = {}
        
    def analyze_eccentricity(self, frame: np.ndarray, circle: Tuple[int, int, int]) -> Dict:
        """
        Analyze the eccentricity of an object given a Hough circle fit.
        
        Args:
            frame: Input image (BGR format)
            circle: Tuple of (center_x, center_y, radius) from Hough detection
            
        Returns:
            Dictionary containing:
                - eccentricity_score: Overall score (0-100%)
                - radial_deviation: Standard deviation of edge positions
                - intensity_uniformity: How uniform the intensity profile is
                - gradient_consistency: How consistent the gradients are
                - shape_metrics: Additional shape analysis metrics
        """
        if frame is None or circle is None:
            return self._empty_results()
            
        cx, cy, radius = circle
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Ensure we're within image bounds
        h, w = gray.shape
        if cx - radius < 0 or cx + radius >= w or cy - radius < 0 or cy + radius >= h:
            return self._empty_results()
        
        # Extract region of interest
        roi = gray[max(0, cy-radius-10):min(h, cy+radius+10), 
                   max(0, cx-radius-10):min(w, cx+radius+10)]
        
        # Adjust coordinates to ROI
        roi_cx = radius + 10 if cx >= radius + 10 else cx
        roi_cy = radius + 10 if cy >= radius + 10 else cy
        
        # Perform analyses
        radial_profile = self._analyze_radial_profile(roi, roi_cx, roi_cy, radius)
        intensity_profile = self._analyze_intensity_profile(roi, roi_cx, roi_cy, radius)
        gradient_analysis = self._analyze_gradients(roi, roi_cx, roi_cy, radius)
        shape_metrics = self._analyze_shape_metrics(roi, roi_cx, roi_cy, radius)
        
        # Calculate overall eccentricity score
        eccentricity_score = self._calculate_eccentricity_score(
            radial_profile, intensity_profile, gradient_analysis, shape_metrics
        )
        
        results = {
            'eccentricity_score': eccentricity_score,
            'radial_deviation': radial_profile['deviation'],
            'radial_uniformity': radial_profile['uniformity'],
            'intensity_uniformity': intensity_profile['uniformity'],
            'intensity_symmetry': intensity_profile['symmetry'],
            'gradient_consistency': gradient_analysis['consistency'],
            'gradient_circularity': gradient_analysis['circularity'],
            'shape_roundness': shape_metrics['roundness'],
            'shape_eccentricity': shape_metrics['eccentricity'],
            'center': (cx, cy),
            'radius': radius,
            'detailed_metrics': {
                'radial_profile': radial_profile,
                'intensity_profile': intensity_profile,
                'gradient_analysis': gradient_analysis,
                'shape_metrics': shape_metrics
            }
        }
        
        self.last_analysis_results = results
        self.last_eccentricity_score = eccentricity_score
        
        return results
    
    def _analyze_radial_profile(self, roi: np.ndarray, cx: int, cy: int, radius: int) -> Dict:
        """
        Analyze the radial profile by sampling edge positions at different angles.
        """
        angles = np.linspace(0, 2 * np.pi, self.num_radial_samples, endpoint=False)
        edge_distances = []
        
        # Apply edge detection
        edges = cv2.Canny(roi, 50, 150)
        
        for angle in angles:
            # Sample along ray from center outward
            max_dist = min(radius * 2, 
                          min(roi.shape[0] - cy, roi.shape[1] - cx,
                              cy, cx))
            
            for r in range(int(radius * 0.5), int(max_dist)):
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                
                if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                    if edges[y, x] > 0:
                        edge_distances.append(r)
                        break
            else:
                # No edge found, use expected radius
                edge_distances.append(radius)
        
        edge_distances = np.array(edge_distances)
        
        # Calculate metrics
        mean_radius = np.mean(edge_distances)
        deviation = np.std(edge_distances)
        uniformity = 1.0 - (deviation / mean_radius) if mean_radius > 0 else 0
        
        # Check for periodicity (indicates regular polygon rather than circle)
        fft = np.fft.fft(edge_distances - mean_radius)
        power_spectrum = np.abs(fft[:len(fft)//2])
        
        # Find peaks in power spectrum (excluding DC component)
        peaks, _ = find_peaks(power_spectrum[1:], height=np.max(power_spectrum[1:]) * 0.3)
        periodicity = len(peaks) > 0
        
        return {
            'edge_distances': edge_distances,
            'mean_radius': mean_radius,
            'deviation': deviation,
            'uniformity': uniformity,
            'periodicity': periodicity,
            'num_peaks': len(peaks)
        }
    
    def _analyze_intensity_profile(self, roi: np.ndarray, cx: int, cy: int, radius: int) -> Dict:
        """
        Analyze the intensity profile from center to edge.
        """
        angles = np.linspace(0, 2 * np.pi, self.num_radial_samples, endpoint=False)
        radial_profiles = []
        
        for angle in angles:
            profile = []
            for r in np.linspace(0, radius, self.num_radius_samples):
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                
                if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                    profile.append(roi[y, x])
                else:
                    profile.append(0)
            
            if self.intensity_smoothing > 1:
                profile = ndimage.gaussian_filter1d(profile, self.intensity_smoothing)
            radial_profiles.append(profile)
        
        radial_profiles = np.array(radial_profiles)
        
        # Calculate uniformity across different angles
        profile_variance = np.var(radial_profiles, axis=0)
        uniformity = 1.0 - (np.mean(profile_variance) / (np.mean(radial_profiles) + 1e-6))
        
        # Check symmetry by comparing opposite angles
        symmetry_scores = []
        for i in range(len(angles) // 2):
            opposite = (i + len(angles) // 2) % len(angles)
            correlation = np.corrcoef(radial_profiles[i], radial_profiles[opposite])[0, 1]
            symmetry_scores.append(correlation)
        
        symmetry = np.mean(symmetry_scores) if symmetry_scores else 0
        
        return {
            'profiles': radial_profiles,
            'uniformity': max(0, min(1, uniformity)),
            'symmetry': max(0, min(1, symmetry)),
            'mean_profile': np.mean(radial_profiles, axis=0)
        }
    
    def _analyze_gradients(self, roi: np.ndarray, cx: int, cy: int, radius: int) -> Dict:
        """
        Analyze gradient orientations and magnitudes.
        """
        # Calculate gradients
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and orientation
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)
        
        # Create mask for analysis region (annulus around detected circle)
        y, x = np.ogrid[:roi.shape[0], :roi.shape[1]]
        inner_mask = (x - cx)**2 + (y - cy)**2 < (radius * 0.8)**2
        outer_mask = (x - cx)**2 + (y - cy)**2 < (radius * 1.2)**2
        annulus_mask = outer_mask & ~inner_mask
        
        # Only consider significant gradients
        significant_mask = (magnitude > self.gradient_threshold) & annulus_mask
        
        if np.sum(significant_mask) < 10:
            return {
                'consistency': 0,
                'circularity': 0,
                'mean_magnitude': 0
            }
        
        # For circular objects, gradients should point radially
        expected_orientations = np.arctan2(y - cy, x - cx)
        
        # Calculate angular difference
        angle_diff = np.abs(orientation - expected_orientations)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        
        # Calculate consistency (how well gradients align with radial direction)
        consistency = 1.0 - (np.mean(angle_diff[significant_mask]) / np.pi)
        
        # Calculate circularity based on gradient distribution
        hist, _ = np.histogram(orientation[significant_mask], bins=36, range=(-np.pi, np.pi))
        hist = hist / (np.sum(hist) + 1e-6)
        entropy = -np.sum(hist * np.log(hist + 1e-6))
        max_entropy = np.log(36)
        circularity = entropy / max_entropy  # High entropy = more circular
        
        return {
            'consistency': max(0, min(1, consistency)),
            'circularity': max(0, min(1, circularity)),
            'mean_magnitude': np.mean(magnitude[significant_mask])
        }
    
    def _analyze_shape_metrics(self, roi: np.ndarray, cx: int, cy: int, radius: int) -> Dict:
        """
        Analyze shape-based metrics using contour analysis.
        """
        # Threshold to get binary image
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'roundness': 0,
                'eccentricity': 1,
                'solidity': 0
            }
        
        # Find contour closest to detected circle center
        min_dist = float('inf')
        best_contour = None
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                contour_cx = int(M['m10'] / M['m00'])
                contour_cy = int(M['m01'] / M['m00'])
                dist = np.sqrt((contour_cx - cx)**2 + (contour_cy - cy)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_contour = contour
        
        if best_contour is None:
            return {
                'roundness': 0,
                'eccentricity': 1,
                'solidity': 0
            }
        
        # Calculate shape metrics
        area = cv2.contourArea(best_contour)
        perimeter = cv2.arcLength(best_contour, True)
        
        # Roundness (circularity)
        if perimeter > 0:
            roundness = 4 * np.pi * area / (perimeter ** 2)
        else:
            roundness = 0
        
        # Fit ellipse for eccentricity
        if len(best_contour) >= 5:
            ellipse = cv2.fitEllipse(best_contour)
            (_, (width, height), _) = ellipse
            if width > 0 and height > 0:
                eccentricity = 1 - min(width, height) / max(width, height)
            else:
                eccentricity = 1
        else:
            eccentricity = 1
        
        # Solidity (convexity)
        hull = cv2.convexHull(best_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'roundness': max(0, min(1, roundness)),
            'eccentricity': max(0, min(1, eccentricity)),
            'solidity': max(0, min(1, solidity))
        }
    
    def _calculate_eccentricity_score(self, radial_profile: Dict, intensity_profile: Dict,
                                     gradient_analysis: Dict, shape_metrics: Dict) -> float:
        """
        Calculate overall eccentricity score from all analyses.
        """
        # Weight different components
        weights = {
            'radial_uniformity': 0.25,
            'intensity_uniformity': 0.15,
            'intensity_symmetry': 0.15,
            'gradient_consistency': 0.15,
            'gradient_circularity': 0.10,
            'shape_roundness': 0.15,
            'shape_eccentricity': 0.05
        }
        
        scores = {
            'radial_uniformity': radial_profile['uniformity'],
            'intensity_uniformity': intensity_profile['uniformity'],
            'intensity_symmetry': intensity_profile['symmetry'],
            'gradient_consistency': gradient_analysis['consistency'],
            'gradient_circularity': gradient_analysis['circularity'],
            'shape_roundness': shape_metrics['roundness'],
            'shape_eccentricity': 1 - shape_metrics['eccentricity']
        }
        
        # Apply penalty for periodicity (indicates polygon rather than circle)
        if radial_profile.get('periodicity', False):
            periodicity_penalty = 0.1 * (1 + radial_profile.get('num_peaks', 0) / 10)
            scores['radial_uniformity'] *= (1 - periodicity_penalty)
        
        # Calculate weighted score
        total_score = sum(weights[key] * scores[key] for key in weights)
        
        # Convert to percentage
        return max(0, min(100, total_score * 100))
    
    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'eccentricity_score': 0.0,
            'radial_deviation': 0.0,
            'radial_uniformity': 0.0,
            'intensity_uniformity': 0.0,
            'intensity_symmetry': 0.0,
            'gradient_consistency': 0.0,
            'gradient_circularity': 0.0,
            'shape_roundness': 0.0,
            'shape_eccentricity': 1.0,
            'center': (0, 0),
            'radius': 0,
            'detailed_metrics': {}
        }
    
    def visualize_analysis(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Create visualization of the eccentricity analysis.
        """
        if not results or results['eccentricity_score'] == 0:
            return frame
        
        output = frame.copy()
        cx, cy = results['center']
        radius = results['radius']
        
        # Draw the detected circle
        cv2.circle(output, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)
        
        # Draw radial edge points if available
        if 'detailed_metrics' in results and 'radial_profile' in results['detailed_metrics']:
            radial_data = results['detailed_metrics']['radial_profile']
            edge_distances = radial_data.get('edge_distances', [])
            
            if len(edge_distances) > 0:
                angles = np.linspace(0, 2 * np.pi, len(edge_distances), endpoint=False)
                
                # Draw actual edge points
                for angle, dist in zip(angles, edge_distances):
                    x = int(cx + dist * np.cos(angle))
                    y = int(cy + dist * np.sin(angle))
                    cv2.circle(output, (x, y), 1, (255, 0, 0), -1)
        
        # Add text overlay with results
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Background for text
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # Add text
        y_offset = 30
        cv2.putText(output, f"Eccentricity: {results['eccentricity_score']:.1f}%",
                   (20, y_offset), font, font_scale, (0, 255, 255), thickness)
        
        y_offset += 25
        cv2.putText(output, f"Radial Uniformity: {results['radial_uniformity']:.2f}",
                   (20, y_offset), font, font_scale, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(output, f"Intensity Symmetry: {results['intensity_symmetry']:.2f}",
                   (20, y_offset), font, font_scale, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(output, f"Gradient Consistency: {results['gradient_consistency']:.2f}",
                   (20, y_offset), font, font_scale, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(output, f"Shape Roundness: {results['shape_roundness']:.2f}",
                   (20, y_offset), font, font_scale, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(output, f"Eccentricity: {results['shape_eccentricity']:.2f}",
                   (20, y_offset), font, font_scale, (255, 255, 255), 1)
        
        # Color code the circle based on eccentricity
        score = results['eccentricity_score']
        if score > 90:
            color = (0, 255, 0)  # Green - excellent
        elif score > 75:
            color = (0, 255, 255)  # Yellow - good
        elif score > 60:
            color = (0, 165, 255)  # Orange - fair
        else:
            color = (0, 0, 255)  # Red - poor
        
        # Draw colored arc to indicate score
        thickness = 5
        start_angle = -90
        end_angle = int(-90 + (score / 100) * 360)
        cv2.ellipse(output, (cx, cy), (radius + 10, radius + 10), 
                   0, start_angle, end_angle, color, thickness)
        
        return output
    
    def plot_detailed_analysis(self, results: Dict) -> None:
        """
        Create detailed plots of the analysis results.
        """
        if not results or 'detailed_metrics' not in results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Eccentricity Analysis - Score: {results["eccentricity_score"]:.1f}%')
        
        # Plot 1: Radial edge distances
        ax1 = axes[0, 0]
        if 'radial_profile' in results['detailed_metrics']:
            radial_data = results['detailed_metrics']['radial_profile']
            edge_distances = radial_data.get('edge_distances', [])
            if len(edge_distances) > 0:
                angles = np.degrees(np.linspace(0, 2 * np.pi, len(edge_distances), endpoint=False))
                ax1.plot(angles, edge_distances, 'b-', label='Detected edge')
                ax1.axhline(y=radial_data['mean_radius'], color='r', linestyle='--', 
                           label=f'Mean radius: {radial_data["mean_radius"]:.1f}')
                ax1.fill_between(angles, 
                               radial_data['mean_radius'] - radial_data['deviation'],
                               radial_data['mean_radius'] + radial_data['deviation'],
                               alpha=0.3, color='red', label=f'±1 std: {radial_data["deviation"]:.1f}')
                ax1.set_xlabel('Angle (degrees)')
                ax1.set_ylabel('Distance from center (pixels)')
                ax1.set_title('Radial Edge Profile')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
        
        # Plot 2: Intensity profiles
        ax2 = axes[0, 1]
        if 'intensity_profile' in results['detailed_metrics']:
            intensity_data = results['detailed_metrics']['intensity_profile']
            mean_profile = intensity_data.get('mean_profile', [])
            if len(mean_profile) > 0:
                radii = np.linspace(0, results['radius'], len(mean_profile))
                ax2.plot(radii, mean_profile, 'g-', linewidth=2, label='Mean profile')
                
                # Plot a few individual profiles
                profiles = intensity_data.get('profiles', [])
                if len(profiles) > 0:
                    for i in range(0, len(profiles), len(profiles) // 8):
                        ax2.plot(radii, profiles[i], 'gray', alpha=0.3, linewidth=0.5)
                
                ax2.set_xlabel('Distance from center (pixels)')
                ax2.set_ylabel('Intensity')
                ax2.set_title('Radial Intensity Profile')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: Polar plot of edge variations
        ax3 = plt.subplot(2, 2, 3, projection='polar')
        if 'radial_profile' in results['detailed_metrics']:
            radial_data = results['detailed_metrics']['radial_profile']
            edge_distances = radial_data.get('edge_distances', [])
            if len(edge_distances) > 0:
                angles = np.linspace(0, 2 * np.pi, len(edge_distances), endpoint=False)
                ax3.plot(angles, edge_distances, 'b-')
                ax3.fill(angles, edge_distances, alpha=0.3)
                mean_radius = radial_data['mean_radius']
                circle = plt.Circle((0, 0), mean_radius, transform=ax3.transData._b, 
                                  fill=False, edgecolor='red', linewidth=2, linestyle='--')
                ax3.add_artist(circle)
                ax3.set_title('Edge Distance Polar Plot')
        
        # Plot 4: Score breakdown
        ax4 = axes[1, 1]
        categories = ['Radial\nUniformity', 'Intensity\nUniformity', 'Intensity\nSymmetry', 
                     'Gradient\nConsistency', 'Shape\nRoundness']
        scores = [
            results.get('radial_uniformity', 0),
            results.get('intensity_uniformity', 0),
            results.get('intensity_symmetry', 0),
            results.get('gradient_consistency', 0),
            results.get('shape_roundness', 0)
        ]
        
        bars = ax4.bar(categories, scores, color=['blue', 'green', 'orange', 'red', 'purple'])
        ax4.set_ylim(0, 1.1)
        ax4.set_ylabel('Score')
        ax4.set_title('Component Score Breakdown')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


class EccentricityProcessor:
    """
    High-level processor that combines Hough circle detection with eccentricity analysis.
    """
    
    def __init__(self, hough_detector=None, eccentricity_tester=None):
        """
        Initialize the processor.
        
        Args:
            hough_detector: HoughCirclesDetector instance
            eccentricity_tester: EccentricityTester instance
        """
        # Import here to avoid circular dependency
        from hough_circles import HoughCirclesDetector
        
        self.hough_detector = hough_detector or HoughCirclesDetector()
        self.eccentricity_tester = eccentricity_tester or EccentricityTester()
        self.processing_enabled = True
        self.visualization_mode = 'full'  # 'full', 'simple', 'none'
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a frame with both circle detection and eccentricity analysis.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, analysis_results)
        """
        if not self.processing_enabled or frame is None:
            return frame, {}
        
        # Detect circles
        circles, hough_frame = self.hough_detector.detect_circles(frame)
        
        if circles is None or len(circles) == 0:
            return hough_frame, {'eccentricity_score': 0, 'message': 'No circles detected'}
        
        # Analyze the most prominent circle (first one returned by Hough)
        circle = circles[0]  # (x, y, radius)
        
        # Perform eccentricity analysis
        results = self.eccentricity_tester.analyze_eccentricity(frame, circle)
        
        # Create visualization based on mode
        if self.visualization_mode == 'full':
            output_frame = self.eccentricity_tester.visualize_analysis(frame, results)
        elif self.visualization_mode == 'simple':
            output_frame = hough_frame
            # Add eccentricity score
            cv2.putText(output_frame, f"Eccentricity: {results['eccentricity_score']:.1f}%",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            output_frame = frame
        
        return output_frame, results
    
    def toggle_processing(self) -> bool:
        """Toggle processing on/off."""
        self.processing_enabled = not self.processing_enabled
        logging.info(f"Eccentricity processing {'enabled' if self.processing_enabled else 'disabled'}")
        return self.processing_enabled
    
    def set_visualization_mode(self, mode: str):
        """Set visualization mode: 'full', 'simple', or 'none'."""
        if mode in ['full', 'simple', 'none']:
            self.visualization_mode = mode
            logging.info(f"Visualization mode set to: {mode}")


def main():
    """
    Example usage and testing of the eccentricity tester.
    """
    import sys
    
    # Create a test image with a slightly imperfect circle
    test_image = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Draw a slightly elliptical shape
    center = (250, 250)
    axes = (100, 95)  # Slightly elliptical
    angle = 0
    cv2.ellipse(test_image, center, axes, angle, 0, 360, (255, 255, 255), -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, test_image.shape).astype(np.uint8)
    test_image = cv2.add(test_image, noise)
    
    # Create tester
    tester = EccentricityTester()
    
    # Analyze
    results = tester.analyze_eccentricity(test_image, (250, 250, 98))
    
    print("Eccentricity Analysis Results:")
    print(f"Overall Score: {results['eccentricity_score']:.1f}%")
    print(f"Radial Uniformity: {results['radial_uniformity']:.3f}")
    print(f"Intensity Symmetry: {results['intensity_symmetry']:.3f}")
    print(f"Gradient Consistency: {results['gradient_consistency']:.3f}")
    print(f"Shape Roundness: {results['shape_roundness']:.3f}")
    
    # Visualize
    output = tester.visualize_analysis(test_image, results)
    
    cv2.imshow('Eccentricity Test', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Show detailed plots
    tester.plot_detailed_analysis(results)


if __name__ == "__main__":
    main()
