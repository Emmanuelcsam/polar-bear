import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Linux compatibility
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import least_squares, minimize
from skimage.feature import local_binary_pattern
import warnings
warnings.filterwarnings('ignore')
import json
import os

class UltimateFiberSeparator:
    """
    Ultimate Fiber Optic Region Separator
    
    Philosophy: Data-driven boundary detection (2nd derivative minima and gradient peaks)
    with geometric fitting used only for validation and refinement.
    """
    
    def __init__(self, image_path):
        """Initialize with image"""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.height, self.width = self.original.shape[:2]
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Storage for analysis
        self.data_driven_boundaries = []
        self.preprocessing_versions = []
        
        print(f"Loaded image: {self.width}x{self.height} pixels")
    
    def analyze(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*60)
        print("ULTIMATE FIBER OPTIC REGION SEPARATOR")
        print("Data-Driven Analysis with Geometric Validation")
        print("="*60)
        
        # Stage 1: Find initial center using brightness (most reliable)
        self.find_data_driven_center()
        
        # Iterative refinement loop
        max_iterations = 10
        convergence_threshold = 0.3  # pixels - reduced for better convergence
        min_improvement = 0.1  # Stop if improvement is less than this
        
        previous_centers = []
        
        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}")
            print(f"{'='*60}")
            
            # Stage 2: Create preprocessing variants
            self.create_preprocessing_variants()
            
            # Stage 3: Find boundaries using data-driven methods
            self.find_data_driven_boundaries()
            
            # Stage 4: Find consensus boundaries
            self.find_consensus_boundaries()
            
            # Stage 5: Refine center using boundaries
            old_center_x = self.center_x
            old_center_y = self.center_y
            center_refined = self.refine_center_using_boundaries()
            
            # Check convergence
            center_shift = np.sqrt((self.center_x - old_center_x)**2 + 
                                 (self.center_y - old_center_y)**2)
            
            print(f"\nCenter shift: {center_shift:.3f} pixels")
            
            # Store center history
            previous_centers.append([self.center_x, self.center_y])
            
            # Check for oscillation
            if len(previous_centers) >= 3:
                # Check if we're oscillating between positions
                recent_centers = np.array(previous_centers[-3:])
                center_std = np.std(recent_centers, axis=0)
                if np.max(center_std) < 0.5:
                    print("✓ Center stabilized (oscillation detected)")
                    # Use average of recent centers
                    self.center_x = np.mean(recent_centers[:, 0])
                    self.center_y = np.mean(recent_centers[:, 1])
                    break
            
            if center_shift < convergence_threshold and center_refined:
                print("✓ Center converged!")
                break
            elif center_shift < min_improvement:
                print("✓ Minimal improvement - stopping refinement")
                break
            elif not center_refined:
                print("✗ Center refinement failed, keeping current center")
                break
        else:
            print("\n⚠ Maximum iterations reached")
            # Use average of last few centers if we didn't converge
            if len(previous_centers) >= 3:
                recent_centers = np.array(previous_centers[-3:])
                self.center_x = np.mean(recent_centers[:, 0])
                self.center_y = np.mean(recent_centers[:, 1])
                print(f"Using average of recent centers: ({self.center_x:.1f}, {self.center_y:.1f})")
        
        # Final stages with converged center
        print("\n" + "="*60)
        print("FINAL ANALYSIS WITH CONVERGED CENTER")
        print("="*60)
        
        # Re-run boundary detection with final center
        self.create_preprocessing_variants()
        self.find_data_driven_boundaries()
        self.find_consensus_boundaries()
        
        # Stage 5: Optional geometric validation (not primary method)
        self.validate_with_geometry()
        
        # Stage 6: Create masks and apply refinement
        self.create_masks_and_refine()
        
        # Stage 7: Extract regions and output
        self.extract_regions_and_output()
        
        print("\nAnalysis Complete!")
        return self.results
    
    def find_data_driven_center(self):
        """Find center using brightness and texture (no Hough circles)"""
        print("\nStage 1: Data-Driven Center Finding")
        print("-" * 40)
        
        # Apply smoothing
        blurred = cv2.GaussianBlur(self.gray, (9, 9), 2)
        
        # Method 1: Brightness centroid (most reliable for fiber cores)
        brightness_threshold = np.percentile(self.gray, 95)
        _, bright_mask = cv2.threshold(blurred, brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        
        # Keep only largest component
        num_labels, labels = cv2.connectedComponents(bright_mask)
        if num_labels > 1:
            # Find largest component (excluding background)
            largest_label = 1 + np.argmax([np.sum(labels == i) for i in range(1, num_labels)])
            bright_mask = (labels == largest_label).astype(np.uint8) * 255
        
        M_bright = cv2.moments(bright_mask)
        
        if M_bright["m00"] > 0:
            cx_bright = M_bright["m10"] / M_bright["m00"]
            cy_bright = M_bright["m01"] / M_bright["m00"]
        else:
            cx_bright = self.width / 2
            cy_bright = self.height / 2
        
        print(f"  Brightness center: ({cx_bright:.1f}, {cy_bright:.1f})")
        
        # Method 2: Texture uniformity
        lbp = local_binary_pattern(self.gray, P=8, R=1, method='uniform')
        texture_threshold = np.percentile(lbp, 25)
        texture_mask = (lbp <= texture_threshold).astype(np.uint8) * 255
        
        # Clean texture mask
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, kernel)
        
        M_texture = cv2.moments(texture_mask)
        
        if M_texture["m00"] > 0:
            cx_texture = M_texture["m10"] / M_texture["m00"]
            cy_texture = M_texture["m01"] / M_texture["m00"]
        else:
            cx_texture = cx_bright
            cy_texture = cy_bright
        
        print(f"  Texture center: ({cx_texture:.1f}, {cy_texture:.1f})")
        
        # Method 3: Gradient-based center
        # Find the point with minimum total gradient (center of circular pattern)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply heavy gaussian blur to find smooth center
        grad_smooth = cv2.GaussianBlur(grad_mag, (31, 31), 10)
        
        # Find minimum gradient region (center)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grad_smooth)
        cx_gradient = min_loc[0]
        cy_gradient = min_loc[1]
        
        print(f"  Gradient center: ({cx_gradient:.1f}, {cy_gradient:.1f})")
        
        # Combine all methods with validation
        # Check which centers are close to each other
        centers = np.array([[cx_bright, cy_bright], 
                           [cx_texture, cy_texture], 
                           [cx_gradient, cy_gradient]])
        
        # Calculate pairwise distances
        valid_centers = []
        weights = []
        
        for i, (center, name, weight) in enumerate(zip(centers, 
                                                       ['brightness', 'texture', 'gradient'],
                                                       [0.5, 0.3, 0.2])):
            # Check if this center is reasonable (not at image edges)
            margin = 50
            if margin < center[0] < self.width - margin and margin < center[1] < self.height - margin:
                valid_centers.append(center)
                weights.append(weight)
        
        if valid_centers:
            valid_centers = np.array(valid_centers)
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
            
            # Weighted average
            self.center_x = np.sum(valid_centers[:, 0] * weights)
            self.center_y = np.sum(valid_centers[:, 1] * weights)
        else:
            # Fallback to image center
            self.center_x = self.width / 2
            self.center_y = self.height / 2
        
        print(f"  INITIAL CENTER: ({self.center_x:.1f}, {self.center_y:.1f})")
    
    def create_preprocessing_variants(self):
        """Create multiple preprocessed versions"""
        print("\nStage 2: Creating Preprocessing Variants")
        print("-" * 40)
        
        self.preprocessing_versions = []
        
        # Key variants for robustness
        variants = [
            ('Original', self.gray.copy()),
            ('Gaussian_σ1', cv2.GaussianBlur(self.gray, (0, 0), 1)),
            ('Gaussian_σ2', cv2.GaussianBlur(self.gray, (0, 0), 2)),
            ('Median_k3', median_filter(self.gray, size=3)),
            ('Median_k5', median_filter(self.gray, size=5)),
            ('Bilateral', cv2.bilateralFilter(self.gray, 9, 75, 75))
        ]
        
        for name, img in variants:
            self.preprocessing_versions.append({
                'name': name,
                'image': img
            })
        
        print(f"  Created {len(self.preprocessing_versions)} preprocessing variants")
    
    def find_data_driven_boundaries(self):
        """Find boundaries using gradient peaks and 2nd derivative minima"""
        print("\nStage 3: Data-Driven Boundary Detection")
        print("-" * 40)
        
        self.data_driven_boundaries = []
        
        for i, prep in enumerate(self.preprocessing_versions):
            print(f"\n  Analyzing variant {i+1}/{len(self.preprocessing_versions)}: {prep['name']}")
            
            # Compute radial profiles
            profiles = self._compute_radial_profiles(prep['image'])
            
            # Method 1: Second Derivative Minima (most accurate)
            deriv2_boundaries = self._find_second_derivative_minima(profiles)
            if deriv2_boundaries and len(deriv2_boundaries) >= 2:
                # Ensure proper ordering (inner boundary < outer boundary)
                boundaries = sorted(deriv2_boundaries[:2])
                self.data_driven_boundaries.append({
                    'method': '2nd_derivative',
                    'preprocessing': prep['name'],
                    'boundaries': boundaries,
                    'confidence': 1.0,  # Highest confidence
                    'profiles': profiles
                })
                print(f"    2nd derivative minima: {[int(b) for b in boundaries]}")
            
            # Method 2: Gradient Peaks
            gradient_boundaries = self._find_gradient_peaks(profiles)
            if gradient_boundaries and len(gradient_boundaries) >= 2:
                # Ensure proper ordering
                boundaries = sorted(gradient_boundaries[:2])
                self.data_driven_boundaries.append({
                    'method': 'gradient',
                    'preprocessing': prep['name'],
                    'boundaries': boundaries,
                    'confidence': 0.8,  # Slightly lower confidence
                    'profiles': profiles
                })
                print(f"    Gradient peaks: {[int(b) for b in boundaries]}")
        
        print(f"\n  Total boundary detections: {len(self.data_driven_boundaries)}")
    
    def _compute_radial_profiles(self, img):
        """Compute radial intensity profiles and derivatives"""
        max_radius = int(min(self.center_x, self.center_y, 
                            self.width - self.center_x, 
                            self.height - self.center_y))
        
        radii = np.arange(max_radius)
        
        # Sample along many angles for robustness
        num_angles = 360
        angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        intensity_profiles = []
        
        for angle in angles:
            x_coords = self.center_x + radii * np.cos(angle)
            y_coords = self.center_y + radii * np.sin(angle)
            
            intensities = []
            for x, y in zip(x_coords, y_coords):
                if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                    intensities.append(img[int(y), int(x)])
            
            if len(intensities) == len(radii):
                intensity_profiles.append(intensities)
        
        # Use median for robustness
        median_intensity = np.median(intensity_profiles, axis=0)
        
        # Smooth before taking derivatives
        smooth_intensity = gaussian_filter1d(median_intensity, sigma=2)
        
        # Compute derivatives
        first_derivative = np.gradient(smooth_intensity)
        second_derivative = np.gradient(first_derivative)
        
        # Additional smoothing on derivatives
        first_derivative_smooth = gaussian_filter1d(first_derivative, sigma=1)
        second_derivative_smooth = gaussian_filter1d(second_derivative, sigma=1)
        
        return {
            'radii': radii,
            'intensity': median_intensity,
            'intensity_smooth': smooth_intensity,
            'gradient': np.abs(first_derivative_smooth),
            'gradient_signed': first_derivative_smooth,
            'second_derivative': second_derivative_smooth
        }
    
    def _find_second_derivative_minima(self, profiles):
        """Find minima in second derivative (inflection points)"""
        second_deriv = profiles['second_derivative']
        
        # Find minima by inverting and finding peaks
        inverted = -second_deriv
        
        # Find all peaks (minima in original)
        peaks, properties = find_peaks(
            inverted,
            distance=10,  # Minimum distance between peaks
            height=None   # No minimum height requirement
        )
        
        if len(peaks) == 0:
            return None
        
        # Get prominences to rank peaks
        prominences = peak_prominences(inverted, peaks)[0]
        
        # Sort by prominence (most prominent first)
        sorted_indices = np.argsort(prominences)[::-1]
        sorted_peaks = [peaks[i] for i in sorted_indices]
        
        return sorted_peaks
    
    def _find_gradient_peaks(self, profiles):
        """Find peaks in gradient magnitude"""
        gradient = profiles['gradient']
        
        # Find all peaks
        peaks, properties = find_peaks(
            gradient,
            distance=10,
            height=None
        )
        
        if len(peaks) == 0:
            return None
        
        # Get peak heights
        peak_heights = gradient[peaks]
        
        # Sort by height (highest first)
        sorted_indices = np.argsort(peak_heights)[::-1]
        sorted_peaks = [peaks[i] for i in sorted_indices]
        
        return sorted_peaks
    
    def find_consensus_boundaries(self):
        """Find consensus from all data-driven boundaries"""
        print("\nStage 4: Finding Consensus Boundaries")
        print("-" * 40)
        
        if not self.data_driven_boundaries:
            raise ValueError("No boundaries detected!")
        
        # Collect all boundaries with weights
        weighted_boundaries = []
        
        for detection in self.data_driven_boundaries:
            weight = detection['confidence']
            # 2nd derivative gets extra weight
            if detection['method'] == '2nd_derivative':
                weight *= 1.5
            
            for boundary in detection['boundaries']:
                weighted_boundaries.extend([boundary] * int(weight * 10))
        
        # Convert to numpy array
        all_boundaries = np.array(weighted_boundaries)
        
        # Find two main clusters
        sorted_boundaries = np.sort(all_boundaries)
        
        # Find the largest gap to separate into two clusters
        if len(sorted_boundaries) > 10:
            gaps = np.diff(sorted_boundaries)
            gap_idx = np.argmax(gaps)
            
            if gap_idx > 0 and gap_idx < len(sorted_boundaries) - 1:
                cluster1 = sorted_boundaries[:gap_idx+1]
                cluster2 = sorted_boundaries[gap_idx+1:]
                
                boundary1 = int(np.median(cluster1))
                boundary2 = int(np.median(cluster2))
            else:
                # Fallback
                boundary1 = int(np.percentile(sorted_boundaries, 25))
                boundary2 = int(np.percentile(sorted_boundaries, 75))
        else:
            # Very few boundaries, just take extremes
            boundary1 = int(np.min(sorted_boundaries))
            boundary2 = int(np.max(sorted_boundaries))
        
        self.consensus_boundaries = sorted([boundary1, boundary2])
        
        # Validate boundary order based on intensity pattern
        self._validate_boundary_order()
        
        # Calculate confidence based on agreement
        self.boundary_confidence = self._calculate_boundary_confidence(all_boundaries)
        
        print(f"  Consensus boundaries: {self.consensus_boundaries} pixels")
        print(f"  Confidence: {self.boundary_confidence:.2%}")
    
    def _validate_boundary_order(self):
        """Ensure boundaries follow expected intensity pattern"""
        # Sample intensity at different radii
        test_radii = [
            self.consensus_boundaries[0] - 10,  # Inside core
            (self.consensus_boundaries[0] + self.consensus_boundaries[1]) // 2,  # In cladding
            self.consensus_boundaries[1] + 10  # In ferrule
        ]
        
        intensities = []
        for r in test_radii:
            if r >= 0:
                # Sample points at this radius
                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                samples = []
                for angle in angles:
                    x = int(self.center_x + r * np.cos(angle))
                    y = int(self.center_y + r * np.sin(angle))
                    if 0 <= x < self.width and 0 <= y < self.height:
                        samples.append(self.gray[y, x])
                if samples:
                    intensities.append(np.median(samples))
                else:
                    intensities.append(None)
            else:
                intensities.append(None)
        
        # Expected pattern: Core (bright) > Ferrule (medium) > Cladding (dark)
        # If cladding is not darkest, we might need to adjust
        if len(intensities) == 3 and all(i is not None for i in intensities):
            core_int = intensities[0] if intensities[0] else 0
            clad_int = intensities[1] if intensities[1] else 0
            ferr_int = intensities[2] if intensities[2] else 0
            
            if core_int > 0 and clad_int > 0 and ferr_int > 0:
                print("  Intensity validation:")
                print(f"    Core region: {core_int:.1f}")
                print(f"    Cladding region: {clad_int:.1f}")
                print(f"    Ferrule region: {ferr_int:.1f}")
                
                # Check if pattern matches expected
                if not (clad_int < ferr_int < core_int):
                    print("    ⚠ Warning: Unexpected intensity pattern")
    
    def _calculate_boundary_confidence(self, all_boundaries):
        """Calculate confidence based on clustering tightness"""
        if len(all_boundaries) < 2:
            return 0.0
        
        # Check how tightly clustered the boundaries are around consensus
        tolerance = 5  # pixels
        
        near_boundary1 = np.sum(np.abs(all_boundaries - self.consensus_boundaries[0]) <= tolerance)
        near_boundary2 = np.sum(np.abs(all_boundaries - self.consensus_boundaries[1]) <= tolerance)
        
        confidence = (near_boundary1 + near_boundary2) / (2 * len(all_boundaries))
        return min(confidence, 1.0)
    
    def refine_center_using_boundaries(self):
        """Refine center estimate using detected boundaries"""
        print("\nRefining center using detected boundaries...")
        
        if not hasattr(self, 'consensus_boundaries') or len(self.consensus_boundaries) < 2:
            print("  No valid boundaries for center refinement")
            return False
        
        # Method 1: Edge-based center finding
        # Find actual edge points near the consensus boundaries
        edges = cv2.Canny(self.gray, 50, 150)
        edge_points = []
        
        # Sample in a narrow band around each boundary
        for boundary_radius in self.consensus_boundaries:
            # Create annular mask
            Y, X = np.ogrid[:self.height, :self.width]
            dist_from_center = np.sqrt((X - self.center_x)**2 + (Y - self.center_y)**2)
            
            # Band around boundary (±5 pixels)
            band_mask = (np.abs(dist_from_center - boundary_radius) <= 5) & (edges > 0)
            
            # Get edge points in this band
            edge_y, edge_x = np.where(band_mask)
            if len(edge_y) > 20:
                edge_points.extend(zip(edge_x, edge_y))
        
        if len(edge_points) < 50:
            print("  Insufficient edge points, using radial sampling instead")
            # Fallback to radial sampling
            num_angles = 72
            angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
            
            edge_points = []
            for boundary_radius in self.consensus_boundaries:
                for angle in angles:
                    # Sample along radial line
                    for r_offset in range(-5, 6):
                        r = boundary_radius + r_offset
                        x = self.center_x + r * np.cos(angle)
                        y = self.center_y + r * np.sin(angle)
                        if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                            if edges[int(y), int(x)] > 0:
                                edge_points.append([x, y])
                                break
        
        if len(edge_points) < 20:
            print("  Still insufficient edge points for refinement")
            return False
        
        edge_points = np.array(edge_points)
        
        # Method 2: Fit circles to edge points and find common center
        def fit_circle_ransac(points, max_iterations=100):
            """Fit circle using RANSAC for robustness"""
            best_center = None
            best_radius = None
            best_inliers = 0
            
            n_points = len(points)
            if n_points < 3:
                return None, None
            
            for _ in range(max_iterations):
                # Randomly select 3 points
                idx = np.random.choice(n_points, 3, replace=False)
                sample_points = points[idx]
                
                # Fit circle to these 3 points
                center, radius = fit_circle_3points(sample_points)
                
                if center is None or radius is None or radius <= 0 or radius > min(self.width, self.height):
                    continue
                
                # Count inliers
                distances = np.sqrt((points[:, 0] - center[0])**2 + 
                                  (points[:, 1] - center[1])**2)
                inliers = np.sum(np.abs(distances - radius) < 3)  # 3 pixel tolerance
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_center = center
                    best_radius = radius
            
            return best_center, best_radius
        
        def fit_circle_3points(points):
            """Fit circle through 3 points"""
            if len(points) != 3:
                return None, None
            
            # Convert to homogeneous coordinates
            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]
            
            # Check if points are collinear
            det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if abs(det) < 1e-6:
                return None, None
            
            # Calculate circle center
            a = x1**2 + y1**2
            b = x2**2 + y2**2
            c = x3**2 + y3**2
            
            cx = 0.5 * ((a * (y2 - y3) + b * (y3 - y1) + c * (y1 - y2)) / det)
            cy = 0.5 * ((a * (x3 - x2) + b * (x1 - x3) + c * (x2 - x1)) / det)
            
            # Calculate radius
            radius = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
            
            return [cx, cy], radius
        
        # Separate edge points by approximate boundary
        centers = []
        weights = []
        
        for boundary_radius in self.consensus_boundaries:
            # Get points near this boundary
            distances = np.sqrt((edge_points[:, 0] - self.center_x)**2 + 
                              (edge_points[:, 1] - self.center_y)**2)
            boundary_points = edge_points[np.abs(distances - boundary_radius) < 10]
            
            if len(boundary_points) > 10:
                center, radius = fit_circle_ransac(boundary_points)
                if center is not None:
                    centers.append(center)
                    # Weight by number of points and how close radius is to expected
                    weight = len(boundary_points) * np.exp(-abs(radius - boundary_radius) / boundary_radius)
                    weights.append(weight)
        
        if len(centers) == 0:
            print("  Circle fitting failed")
            return False
        
        # Weighted average of centers
        centers = np.array(centers)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        new_center_x = np.sum(centers[:, 0] * weights)
        new_center_y = np.sum(centers[:, 1] * weights)
        
        # Limit the maximum shift to prevent instability
        max_shift = 5.0
        shift_x = new_center_x - self.center_x
        shift_y = new_center_y - self.center_y
        shift_magnitude = np.sqrt(shift_x**2 + shift_y**2)
        
        if shift_magnitude > max_shift:
            # Scale down the shift
            scale = max_shift / shift_magnitude
            new_center_x = self.center_x + shift_x * scale
            new_center_y = self.center_y + shift_y * scale
        
        # Apply smoothing factor to prevent oscillation
        smoothing = 0.7  # Move only 70% of the way to the new center
        final_x = self.center_x + smoothing * (new_center_x - self.center_x)
        final_y = self.center_y + smoothing * (new_center_y - self.center_y)
        
        # Update center
        old_x, old_y = self.center_x, self.center_y
        self.center_x = final_x
        self.center_y = final_y
        
        print(f"  Old center: ({old_x:.2f}, {old_y:.2f})")
        print(f"  New center: ({self.center_x:.2f}, {self.center_y:.2f})")
        print(f"  Shift: {np.sqrt((self.center_x - old_x)**2 + (self.center_y - old_y)**2):.3f} pixels")
        
        # Verify improvement
        print(f"  Circle fitting used {len(centers)} boundaries")
        
        return True
    
    def validate_with_geometry(self):
        """Optional geometric validation - only to confirm data-driven boundaries"""
        print("\nStage 5: Geometric Validation (Optional)")
        print("-" * 40)
        
        # Only proceed if confidence is low
        if self.boundary_confidence > 0.7:
            print("  High confidence in data-driven boundaries - skipping geometric validation")
            return
        
        print("  Low confidence - attempting geometric validation...")
        
        # Get edge points
        edge_points = self._get_simple_edge_points()
        
        if len(edge_points) < 100:
            print("  Insufficient edge points for geometric validation")
            return
        
        # Try to fit circles at the consensus boundaries
        # This is ONLY for validation, not for finding boundaries
        validation_score = self._validate_boundaries_geometrically(edge_points)
        
        if validation_score < 10:  # Good alignment
            print(f"  Geometric validation successful (score: {validation_score:.1f})")
            self.boundary_confidence = min(self.boundary_confidence * 1.2, 1.0)
        else:
            print(f"  Geometric validation weak (score: {validation_score:.1f})")
    
    def _get_simple_edge_points(self):
        """Get edge points using gradient threshold"""
        # Simple gradient
        grad_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold
        threshold = np.percentile(grad_mag, 85)
        edge_mask = grad_mag > threshold
        
        # Get points
        points = np.column_stack(np.where(edge_mask))
        return points[:, [1, 0]]  # Convert to (x, y)
    
    def _validate_boundaries_geometrically(self, edge_points):
        """Check if edge points align with consensus boundaries"""
        # Calculate distances from center for all edge points
        distances = np.sqrt((edge_points[:, 0] - self.center_x)**2 + 
                           (edge_points[:, 1] - self.center_y)**2)
        
        # Count points near each boundary
        tolerance = 5
        near_boundary1 = np.sum(np.abs(distances - self.consensus_boundaries[0]) < tolerance)
        near_boundary2 = np.sum(np.abs(distances - self.consensus_boundaries[1]) < tolerance)
        
        # Score is inverse of alignment (lower is better)
        total_near = near_boundary1 + near_boundary2
        score = 100 * (1 - total_near / len(edge_points))
        
        return score
    
    def create_masks_and_refine(self):
        """Create masks and apply binary refinement"""
        print("\nStage 6: Creating Masks and Applying Refinement")
        print("-" * 40)
        
        # Create distance map
        Y, X = np.ogrid[:self.height, :self.width]
        distance_map = np.sqrt((X - self.center_x)**2 + (Y - self.center_y)**2)
        
        # Create initial masks based on consensus boundaries
        self.masks = {}
        
        # Core mask
        self.masks['core'] = (distance_map <= self.consensus_boundaries[0]).astype(np.uint8) * 255
        
        # Cladding mask
        self.masks['cladding'] = ((distance_map > self.consensus_boundaries[0]) & 
                                 (distance_map <= self.consensus_boundaries[1])).astype(np.uint8) * 255
        
        # Ferrule mask
        self.masks['ferrule'] = (distance_map > self.consensus_boundaries[1]).astype(np.uint8) * 255
        
        # Apply binary refinement
        self._apply_binary_refinement()
        
        print("  Masks created and refined")
    
    def _apply_binary_refinement(self):
        """Apply binary filter to remove artifacts"""
        # Extract initial regions
        initial_regions = {}
        for name, mask in self.masks.items():
            initial_regions[name] = cv2.bitwise_and(self.original, self.original, mask=mask)
        
        # Refine core: keep bright pixels
        if 'core' in initial_regions:
            core_gray = cv2.cvtColor(initial_regions['core'], cv2.COLOR_BGR2GRAY)
            
            if np.any(self.masks['core']):
                # Threshold using Otsu on valid pixels only
                valid_pixels = core_gray[self.masks['core'] > 0]
                if len(valid_pixels) > 0:
                    threshold = max(np.percentile(valid_pixels, 50), 1)
                    _, binary = cv2.threshold(core_gray, threshold, 255, cv2.THRESH_BINARY)
                    
                    # Morphological cleaning
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    
                    # Keep only largest component
                    num_labels, labels = cv2.connectedComponents(binary)
                    if num_labels > 1:
                        # Find largest component (excluding background)
                        largest_label = 1 + np.argmax([np.sum(labels == i) for i in range(1, num_labels)])
                        binary = (labels == largest_label).astype(np.uint8) * 255
                    
                    # Update mask
                    self.masks['core'] = cv2.bitwise_and(self.masks['core'], binary)
        
        # Refine cladding: remove bright spots
        if 'cladding' in initial_regions:
            clad_gray = cv2.cvtColor(initial_regions['cladding'], cv2.COLOR_BGR2GRAY)
            
            if np.any(self.masks['cladding']):
                valid_pixels = clad_gray[self.masks['cladding'] > 0]
                if len(valid_pixels) > 0:
                    # Find bright outliers
                    threshold = np.percentile(valid_pixels, 75)
                    _, bright_spots = cv2.threshold(clad_gray, threshold, 255, cv2.THRESH_BINARY)
                    
                    # Remove bright spots from cladding
                    self.masks['cladding'] = cv2.bitwise_and(
                        self.masks['cladding'],
                        cv2.bitwise_not(bright_spots)
                    )
    
    def extract_regions_and_output(self):
        """Extract final regions and generate output"""
        print("\nStage 7: Extracting Regions and Generating Output")
        print("-" * 40)
        
        # Verify final alignment quality
        self._verify_alignment_quality()
        
        # Extract regions
        self.regions = {}
        region_stats = {}
        
        for region_name, mask in self.masks.items():
            self.regions[region_name] = cv2.bitwise_and(
                self.original, self.original, mask=mask
            )
            
            # Calculate statistics - Convert numpy int64 to Python int
            pixel_count = int(np.sum(mask > 0))  # Fix: Convert to Python int
            region_stats[region_name] = {
                'pixel_count': pixel_count,
                'percentage': float(100 * pixel_count / (self.width * self.height))  # Fix: Convert to Python float
            }
        
        # Generate report BEFORE visualization
        self.results = {
            'image_info': {
                'width': int(self.width),  # Fix: Convert to Python int
                'height': int(self.height),  # Fix: Convert to Python int
                'center': {
                    'x': float(self.center_x),
                    'y': float(self.center_y)
                }
            },
            'consensus_boundaries': [int(b) for b in self.consensus_boundaries],
            'boundary_confidence': float(self.boundary_confidence),
            'alignment_quality': self.alignment_metrics,
            'regions': region_stats,
            'method_summary': {
                'total_detections': len(self.data_driven_boundaries),
                '2nd_derivative': len([d for d in self.data_driven_boundaries 
                                     if d['method'] == '2nd_derivative']),
                'gradient': len([d for d in self.data_driven_boundaries 
                               if d['method'] == 'gradient'])
            }
        }
        
        # Create output directory
        output_dir = 'ultimate_fiber_separation_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save regions
        for region_name, region in self.regions.items():
            cv2.imwrite(os.path.join(output_dir, f'{region_name}.png'), region)
            print(f"  Saved {region_name}: {region_stats[region_name]['pixel_count']} pixels "
                  f"({region_stats[region_name]['percentage']:.1f}%)")
        
        # Save masks
        for region_name, mask in self.masks.items():
            cv2.imwrite(os.path.join(output_dir, f'{region_name}_mask.png'), mask)
        
        # Create visualization (now self.results exists)
        self._create_final_visualization(output_dir)
        
        # Save report
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"\nResults saved to: {output_dir}/")
    
    def _verify_alignment_quality(self):
        """Verify the quality of the final center alignment"""
        print("\nVerifying alignment quality...")
        
        # Compute radial variance at each boundary
        num_angles = 360
        angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        self.alignment_metrics = {}
        
        for i, boundary_radius in enumerate(self.consensus_boundaries):
            # Sample intensities at this radius
            intensities = []
            actual_radii = []
            
            for angle in angles:
                x = self.center_x + boundary_radius * np.cos(angle)
                y = self.center_y + boundary_radius * np.sin(angle)
                
                if 0 <= int(x) < self.width and 0 <= int(y) < self.height:
                    # Get intensity
                    intensities.append(self.gray[int(y), int(x)])
                    
                    # Find actual edge location near this radius
                    # Sample along a radial line
                    search_range = 10
                    radial_profile = []
                    radial_positions = []
                    
                    for r in range(max(0, boundary_radius - search_range), 
                                 min(boundary_radius + search_range, int(min(self.width, self.height)/2))):
                        rx = self.center_x + r * np.cos(angle)
                        ry = self.center_y + r * np.sin(angle)
                        if 0 <= int(rx) < self.width and 0 <= int(ry) < self.height:
                            radial_profile.append(self.gray[int(ry), int(rx)])
                            radial_positions.append(r)
                    
                    if len(radial_profile) > 5:
                        # Find the steepest gradient
                        gradients = np.abs(np.gradient(radial_profile))
                        max_grad_idx = np.argmax(gradients)
                        actual_radius = radial_positions[max_grad_idx]
                        actual_radii.append(actual_radius)
            
            if len(intensities) > num_angles * 0.8:
                # Calculate metrics
                intensity_std = np.std(intensities)
                intensity_cv = intensity_std / np.mean(intensities) * 100  # Coefficient of variation
                
                if actual_radii:
                    radius_std = np.std(actual_radii)
                    radius_mean = np.mean(actual_radii)
                    circularity = 1.0 - (radius_std / radius_mean)  # 1.0 = perfect circle
                else:
                    radius_std = 0
                    circularity = 0
                
                self.alignment_metrics[f'boundary_{i+1}'] = {
                    'intensity_cv': float(intensity_cv),
                    'radius_std': float(radius_std),
                    'circularity': float(circularity),
                    'nominal_radius': int(boundary_radius)
                }
                
                print(f"  Boundary {i+1} ({boundary_radius}px):")
                print(f"    Intensity CV: {intensity_cv:.1f}%")
                print(f"    Radius std dev: {radius_std:.2f}px")
                print(f"    Circularity: {circularity:.3f}")
                
                # Perfect alignment criteria
                if intensity_cv < 5.0 and radius_std < 2.0:
                    print("    ✓ Excellent alignment!")
                elif intensity_cv < 10.0 and radius_std < 4.0:
                    print("    ✓ Good alignment")
                else:
                    print("    ⚠ Alignment could be improved")
    
    def _create_final_visualization(self, output_dir):
        """Create comprehensive visualization"""
        # Find best detection for visualization
        best_detection = max(self.data_driven_boundaries, 
                           key=lambda x: x['confidence'])
        profiles = best_detection['profiles']
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Original image with boundaries
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        ax1.plot(self.center_x, self.center_y, 'r+', markersize=15, markeredgewidth=2)
        # Draw circles at boundaries
        circle1 = Circle((self.center_x, self.center_y), self.consensus_boundaries[0], 
                        fill=False, color='green', linewidth=2, linestyle='--')
        circle2 = Circle((self.center_x, self.center_y), self.consensus_boundaries[1], 
                        fill=False, color='red', linewidth=2, linestyle='--')
        ax1.add_patch(circle1)
        ax1.add_patch(circle2)
        ax1.set_title('Original Image with Boundaries')
        ax1.axis('off')
        
        # Region masks
        ax2 = fig.add_subplot(gs[0, 1])
        mask_overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask_overlay[self.masks['core'] > 0] = [255, 0, 0]
        mask_overlay[self.masks['cladding'] > 0] = [0, 255, 0]
        mask_overlay[self.masks['ferrule'] > 0] = [0, 0, 255]
        ax2.imshow(mask_overlay)
        ax2.set_title('Region Masks (R=Core, G=Cladding, B=Ferrule)')
        ax2.axis('off')
        
        # Separated regions
        ax3 = fig.add_subplot(gs[0, 2])
        composite = np.zeros_like(self.original)
        for name, region in self.regions.items():
            mask = self.masks[name] > 0
            composite[mask] = region[mask]
        ax3.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        ax3.set_title('Separated Regions')
        ax3.axis('off')
        
        # Intensity profile
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(profiles['radii'], profiles['intensity'], 'b-', linewidth=2, label='Intensity')
        ax4.plot(profiles['radii'], profiles['intensity_smooth'], 'b--', alpha=0.5, 
                label='Smoothed', linewidth=1)
        ax4.set_xlabel('Radius (pixels)')
        ax4.set_ylabel('Intensity')
        ax4.set_title('Radial Intensity Profile')
        ax4.grid(True, alpha=0.3)
        
        # Mark boundaries with intensity values
        for i, b in enumerate(self.consensus_boundaries):
            if b < len(profiles['intensity']):
                intensity_at_boundary = profiles['intensity'][b]
                ax4.axvline(x=b, color=['g', 'r'][i], linestyle='--', linewidth=2,
                           label=f'Boundary {i+1} ({b}px, I={intensity_at_boundary:.0f})')
        ax4.legend()
        
        # Gradient profile
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(profiles['radii'], profiles['gradient'], 'orange', linewidth=2)
        ax5.set_xlabel('Radius (pixels)')
        ax5.set_ylabel('|dI/dr|')
        ax5.set_title('Gradient Magnitude')
        ax5.grid(True, alpha=0.3)
        for b in self.consensus_boundaries:
            ax5.axvline(x=b, color='r', linestyle='--', linewidth=1)
        
        # Second derivative
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(profiles['radii'], profiles['second_derivative'], 'green', linewidth=2)
        ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax6.set_xlabel('Radius (pixels)')
        ax6.set_ylabel('d²I/dr²')
        ax6.set_title('Second Derivative (Minima = Inflection Points)')
        ax6.grid(True, alpha=0.3)
        
        # Mark minima
        for b in self.consensus_boundaries:
            ax6.axvline(x=b, color='r', linestyle='--', linewidth=1)
            if b < len(profiles['second_derivative']):
                ax6.plot(b, profiles['second_derivative'][b], 'ro', markersize=8)
        
        # Method statistics
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        stats_text = f"Analysis Summary:\n\n"
        stats_text += f"Total detections: {self.results['method_summary']['total_detections']}\n"
        stats_text += f"2nd derivative: {self.results['method_summary']['2nd_derivative']}\n"
        stats_text += f"Gradient: {self.results['method_summary']['gradient']}\n\n"
        stats_text += f"Boundary confidence: {self.boundary_confidence:.1%}\n"
        stats_text += f"Center: ({self.center_x:.1f}, {self.center_y:.1f})\n"
        stats_text += f"Boundaries: {self.consensus_boundaries}\n\n"
        
        # Add alignment quality
        if hasattr(self, 'alignment_metrics') and self.alignment_metrics:
            stats_text += "Alignment Quality:\n"
            for boundary_name, metrics in self.alignment_metrics.items():
                stats_text += f"  {boundary_name}: {metrics['circularity']:.3f}\n"
            stats_text += "\n"
        
        stats_text += "Region sizes:\n"
        for region, data in self.results['regions'].items():
            stats_text += f"  {region}: {data['percentage']:.1f}%\n"
        
        ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Data-Driven Fiber Optic Analysis Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'analysis_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close()


# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Create test image
        print("No image provided. Creating test fiber image...")
        
        image_path = 'test_fiber.jpg'
        size = 500
        img = np.zeros((size, size, 3), dtype=np.uint8)
        center = size // 2
        
        # Create realistic pattern
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - center)**2 + (Y - center)**2)
        
        # Base intensity
        intensity = np.ones((size, size)) * 140  # Ferrule
        
        # Core (brightest)
        core_mask = dist < 60
        intensity[core_mask] = 180
        
        # Cladding (darkest)
        cladding_mask = (dist >= 60) & (dist < 120)
        intensity[cladding_mask] = 100
        
        # Smooth transitions
        from scipy.ndimage import gaussian_filter
        intensity = gaussian_filter(intensity, sigma=3)
        
        # Add artifacts
        # Bright spots in cladding
        np.random.seed(42)
        for _ in range(5):
            x = np.random.randint(100, size-100)
            y = np.random.randint(100, size-100)
            if cladding_mask[y, x]:
                cv2.circle(intensity, (x, y), 3, 200, -1)
        
        # Dark spots in core
        for _ in range(3):
            x = np.random.randint(center-30, center+30)
            y = np.random.randint(center-30, center+30)
            if core_mask[y, x]:
                cv2.circle(intensity, (x, y), 2, 50, -1)
        
        # Apply to all channels
        for i in range(3):
            img[:, :, i] = intensity.astype(np.uint8)
        
        # Add noise
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(image_path, img)
        print(f"Test image saved as {image_path}")
    
    # Run analysis
    try:
        separator = UltimateFiberSeparator(image_path)
        results = separator.analyze()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE - SUMMARY:")
        print("="*60)
        print(f"Center: ({results['image_info']['center']['x']:.1f}, "
              f"{results['image_info']['center']['y']:.1f})")
        print(f"Boundaries: {results['consensus_boundaries']} pixels")
        print(f"Confidence: {results['boundary_confidence']:.1%}")
        print("\nRegion sizes:")
        for region, data in results['regions'].items():
            print(f"  {region}: {data['pixel_count']} pixels ({data['percentage']:.1f}%)")
        print(f"\nMethod breakdown:")
        print(f"  2nd derivative detections: {results['method_summary']['2nd_derivative']}")
        print(f"  Gradient detections: {results['method_summary']['gradient']}")
        print("\nNote: If regions seem small, the fiber may be small relative to the image.")
        print("Check the visualization for boundary placement accuracy.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()