#!/usr/bin/env python3
"""
Integrated Fiber Optic Analysis Pipeline
Demonstrates usage of all modular functions together

This script shows how to combine all the extracted functions 
into a complete fiber optic analysis pipeline.
"""

import cv2
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple, Optional

# Import all modular functions
try:
    from image_filtering import (
        apply_clahe_enhancement, gaussian_blur_adaptive, 
        denoise_bilateral, homomorphic_filter
    )
    from center_detection import (
        multi_method_center_fusion, validate_center,
        detect_center_hough_circles
    )
    from edge_detection_ransac import (
        extract_edge_points, ransac_two_circles_fitting,
        adaptive_canny_thresholds
    )
    from radial_profile_analysis import (
        compute_radial_intensity_profile, 
        compute_radial_gradient_profile,
        find_fiber_boundaries_from_profile,
        analyze_multi_scale_profiles
    )
    from mask_creation import (
        create_fiber_masks, validate_mask_geometry,
        apply_morphological_operations
    )
    from peak_detection import (
        analyze_signal_quality, consensus_peak_detection,
        multi_scale_peak_detection
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all module files are in the same directory")
    sys.exit(1)


class IntegratedFiberAnalyzer:
    """
    Integrated fiber optic analysis using all modular functions.
    Combines the best functions from all legacy scripts.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.results = {}
        
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Complete fiber optic image analysis pipeline.
        
        Args:
            image_path: Path to fiber optic image
            
        Returns:
            Dictionary containing all analysis results
        """
        print(f"\n{'='*60}")
        print("INTEGRATED FIBER OPTIC ANALYSIS")
        print(f"{'='*60}")
        
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        print(f"Loaded image: {original.shape}")
        
        # Convert to grayscale
        if len(original.shape) == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray = original.copy()
            original = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        self.results = {
            'image_path': image_path,
            'image_shape': original.shape,
            'success': False
        }
        
        try:
            # Stage 1: Image Preprocessing
            self._stage1_preprocessing(gray)
            
            # Stage 2: Center Detection
            self._stage2_center_detection()
            
            # Stage 3: Edge Analysis
            self._stage3_edge_analysis()
            
            # Stage 4: Radial Profile Analysis
            self._stage4_radial_analysis()
            
            # Stage 5: Boundary Detection
            self._stage5_boundary_detection()
            
            # Stage 6: Mask Creation
            self._stage6_mask_creation()
            
            # Stage 7: Validation and Quality Assessment
            self._stage7_validation()
            
            self.results['success'] = True
            print("\n✓ Analysis completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Analysis failed: {e}")
            self.results['error'] = str(e)
            
        return self.results
    
    def _stage1_preprocessing(self, gray: np.ndarray):
        """Stage 1: Advanced image preprocessing"""
        print("\nStage 1: Image Preprocessing")
        print("-" * 30)
        
        # CLAHE enhancement
        enhanced = apply_clahe_enhancement(gray, clip_limit=3.0)
        print("✓ CLAHE enhancement applied")
        
        # Homomorphic filtering for illumination correction
        homo_filtered = homomorphic_filter(enhanced)
        print("✓ Homomorphic filtering applied")
        
        # Bilateral denoising
        denoised = denoise_bilateral(homo_filtered)
        print("✓ Bilateral denoising applied")
        
        # Adaptive Gaussian blur
        blurred = gaussian_blur_adaptive(denoised)
        print("✓ Adaptive Gaussian blur applied")
        
        self.results['preprocessing'] = {
            'original': gray,
            'enhanced': enhanced,
            'homo_filtered': homo_filtered,
            'denoised': denoised,
            'blurred': blurred
        }
    
    def _stage2_center_detection(self):
        """Stage 2: Multi-method center detection"""
        print("\nStage 2: Center Detection")
        print("-" * 30)
        
        processed_image = self.results['preprocessing']['blurred']
        
        # Multi-method center fusion
        center = multi_method_center_fusion(processed_image)
        print(f"✓ Multi-method center: {center}")
        
        # Validate center
        is_valid = validate_center(processed_image, center)
        print(f"✓ Center validation: {'PASS' if is_valid else 'FAIL'}")
        
        # Try Hough circles as backup
        hough_center = detect_center_hough_circles(processed_image)
        print(f"✓ Hough center backup: {hough_center}")
        
        # Use best center
        final_center = center if is_valid else (hough_center if hough_center else center)
        
        self.results['center_detection'] = {
            'multi_method_center': center,
            'hough_center': hough_center,
            'final_center': final_center,
            'validation_passed': is_valid
        }
    
    def _stage3_edge_analysis(self):
        """Stage 3: Edge detection and geometric fitting"""
        print("\nStage 3: Edge Analysis")
        print("-" * 30)
        
        processed_image = self.results['preprocessing']['blurred']
        
        # Adaptive edge detection
        low_thresh, high_thresh = adaptive_canny_thresholds(processed_image)
        edge_points = extract_edge_points(processed_image, 
                                        low_threshold=low_thresh,
                                        high_threshold=high_thresh)
        print(f"✓ Extracted {len(edge_points)} edge points")
        
        # RANSAC circle fitting
        ransac_params = None
        if len(edge_points) > 100:
            ransac_params = ransac_two_circles_fitting(edge_points)
            if ransac_params:
                print(f"✓ RANSAC circles: center({ransac_params[0]:.1f}, {ransac_params[1]:.1f}), radii({ransac_params[2]:.1f}, {ransac_params[3]:.1f})")
            else:
                print("✗ RANSAC fitting failed")
        else:
            print("✗ Insufficient edge points for RANSAC")
        
        self.results['edge_analysis'] = {
            'edge_points': edge_points,
            'num_edge_points': len(edge_points),
            'canny_thresholds': (low_thresh, high_thresh),
            'ransac_params': ransac_params
        }
    
    def _stage4_radial_analysis(self):
        """Stage 4: Radial profile analysis"""
        print("\nStage 4: Radial Profile Analysis")
        print("-" * 30)
        
        processed_image = self.results['preprocessing']['blurred']
        center = self.results['center_detection']['final_center']
        
        # Radial intensity profile
        radii, intensity_profile = compute_radial_intensity_profile(processed_image, center)
        print(f"✓ Intensity profile: {len(intensity_profile)} points")
        
        # Radial gradient profile
        _, gradient_profile = compute_radial_gradient_profile(processed_image, center)
        print(f"✓ Gradient profile: {len(gradient_profile)} points")
        
        # Multi-scale analysis
        multi_scale_results = analyze_multi_scale_profiles(processed_image, center)
        print(f"✓ Multi-scale analysis: {len(multi_scale_results)} scales")
        
        # Signal quality analysis
        intensity_quality = analyze_signal_quality(intensity_profile)
        gradient_quality = analyze_signal_quality(gradient_profile)
        print(f"✓ Signal quality - Intensity SNR: {intensity_quality.get('snr', 0):.2f}, Gradient SNR: {gradient_quality.get('snr', 0):.2f}")
        
        self.results['radial_analysis'] = {
            'radii': radii,
            'intensity_profile': intensity_profile,
            'gradient_profile': gradient_profile,
            'multi_scale_results': multi_scale_results,
            'intensity_quality': intensity_quality,
            'gradient_quality': gradient_quality
        }
    
    def _stage5_boundary_detection(self):
        """Stage 5: Boundary detection from profiles"""
        print("\nStage 5: Boundary Detection")
        print("-" * 30)
        
        radial_data = self.results['radial_analysis']
        radii = radial_data['radii']
        intensity_profile = radial_data['intensity_profile']
        gradient_profile = radial_data['gradient_profile']
        
        # Profile-based boundary detection
        core_radius, cladding_radius = find_fiber_boundaries_from_profile(
            radii, intensity_profile, gradient_profile)
        print(f"✓ Profile boundaries - Core: {core_radius}, Cladding: {cladding_radius}")
        
        # Multi-scale peak detection for validation
        intensity_peaks = multi_scale_peak_detection(intensity_profile)
        gradient_peaks = multi_scale_peak_detection(gradient_profile)
        
        # Consensus boundary detection
        all_boundaries = []
        for scale_peaks in intensity_peaks.values():
            all_boundaries.extend([radii[p] for p in scale_peaks if p < len(radii)])
        for scale_peaks in gradient_peaks.values():
            all_boundaries.extend([radii[p] for p in scale_peaks if p < len(radii)])
        
        # Use RANSAC parameters if available and reasonable
        ransac_params = self.results['edge_analysis']['ransac_params']
        final_core_radius = core_radius
        final_cladding_radius = cladding_radius
        
        if ransac_params and len(ransac_params) >= 4:
            ransac_core = min(ransac_params[2], ransac_params[3])
            ransac_cladding = max(ransac_params[2], ransac_params[3])
            
            # Use RANSAC if close to profile-based detection
            if abs(ransac_core - core_radius) < 20:
                final_core_radius = int((ransac_core + core_radius) / 2)
            if abs(ransac_cladding - cladding_radius) < 30:
                final_cladding_radius = int((ransac_cladding + cladding_radius) / 2)
        
        print(f"✓ Final boundaries - Core: {final_core_radius}, Cladding: {final_cladding_radius}")
        
        self.results['boundary_detection'] = {
            'profile_core_radius': core_radius,
            'profile_cladding_radius': cladding_radius,
            'final_core_radius': final_core_radius,
            'final_cladding_radius': final_cladding_radius,
            'all_detected_boundaries': sorted(set(all_boundaries))
        }
    
    def _stage6_mask_creation(self):
        """Stage 6: Create segmentation masks"""
        print("\nStage 6: Mask Creation")
        print("-" * 30)
        
        image_shape = self.results['preprocessing']['original'].shape
        center = self.results['center_detection']['final_center']
        core_radius = self.results['boundary_detection']['final_core_radius']
        cladding_radius = self.results['boundary_detection']['final_cladding_radius']
        
        # Create fiber masks
        masks = create_fiber_masks(image_shape, center, core_radius, cladding_radius)
        print("✓ Created core, cladding, and ferrule masks")
        
        # Validate mask geometry
        validations = {}
        for region, mask in masks.items():
            is_valid = validate_mask_geometry(mask, min_area=50)
            validations[region] = is_valid
            print(f"✓ {region} mask validation: {'PASS' if is_valid else 'FAIL'}")
        
        # Clean up masks with morphological operations
        cleaned_masks = {}
        for region, mask in masks.items():
            cleaned = apply_morphological_operations(mask, 'close', kernel_size=3)
            cleaned_masks[region] = cleaned
        
        print("✓ Applied morphological cleaning to masks")
        
        self.results['masks'] = {
            'raw_masks': masks,
            'cleaned_masks': cleaned_masks,
            'validations': validations
        }
    
    def _stage7_validation(self):
        """Stage 7: Overall validation and quality assessment"""
        print("\nStage 7: Validation & Quality Assessment")
        print("-" * 30)
        
        # Overall quality score
        quality_factors = []
        
        # Center validation
        if self.results['center_detection']['validation_passed']:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.5)
        
        # Edge points quality
        num_edges = self.results['edge_analysis']['num_edge_points']
        edge_quality = min(1.0, num_edges / 1000)  # Normalize to 1000 points
        quality_factors.append(edge_quality)
        
        # Signal quality
        intensity_snr = self.results['radial_analysis']['intensity_quality'].get('snr', 0)
        signal_quality = min(1.0, intensity_snr / 10)  # Normalize to SNR of 10
        quality_factors.append(signal_quality)
        
        # Mask validation
        mask_validations = self.results['masks']['validations']
        mask_quality = sum(mask_validations.values()) / len(mask_validations)
        quality_factors.append(mask_quality)
        
        # RANSAC success
        ransac_quality = 1.0 if self.results['edge_analysis']['ransac_params'] else 0.3
        quality_factors.append(ransac_quality)
        
        overall_quality = np.mean(quality_factors)
        
        print(f"✓ Overall quality score: {overall_quality:.2f}")
        print(f"  - Center detection: {quality_factors[0]:.2f}")
        print(f"  - Edge quality: {quality_factors[1]:.2f}")
        print(f"  - Signal quality: {quality_factors[2]:.2f}")
        print(f"  - Mask quality: {quality_factors[3]:.2f}")
        print(f"  - RANSAC quality: {quality_factors[4]:.2f}")
        
        self.results['validation'] = {
            'overall_quality': overall_quality,
            'quality_factors': {
                'center': quality_factors[0],
                'edges': quality_factors[1],
                'signal': quality_factors[2],
                'masks': quality_factors[3],
                'ransac': quality_factors[4]
            }
        }
    
    def save_results(self, output_dir: str = "analysis_output"):
        """Save analysis results and visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save numerical results (excluding image arrays)
        import json
        from datetime import datetime
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to Python types for JSON serialization"""
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            else:
                return obj
        
        results_for_json = {
            'timestamp': str(datetime.now()),
            'image_path': self.results['image_path'],
            'image_shape': self.results['image_shape'],
            'success': self.results['success'],
            'center': convert_numpy_types(self.results['center_detection']['final_center']),
            'core_radius': convert_numpy_types(self.results['boundary_detection']['final_core_radius']),
            'cladding_radius': convert_numpy_types(self.results['boundary_detection']['final_cladding_radius']),
            'quality_score': convert_numpy_types(self.results['validation']['overall_quality']),
            'edge_points_count': convert_numpy_types(self.results['edge_analysis']['num_edge_points'])
        }
        
        with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        # Save masks
        masks = self.results['masks']['cleaned_masks']
        for region, mask in masks.items():
            cv2.imwrite(os.path.join(output_dir, f'{region}_mask.png'), mask)
        
        print(f"✓ Results saved to {output_dir}/")


def main():
    """Test the integrated analysis pipeline"""
    # Create a test image if no argument provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Create synthetic test image
        print("Creating synthetic test image...")
        test_image = np.zeros((400, 400, 3), dtype=np.uint8)
        test_image[:] = 100  # Gray background
        
        center = (200, 200)
        # Add circular structures
        cv2.circle(test_image, center, 60, (200, 200, 200), -1)   # Bright core
        cv2.circle(test_image, center, 100, (150, 150, 150), 20)  # Cladding ring
        cv2.circle(test_image, center, 140, (80, 80, 80), -1)     # Outer region
        
        # Add some noise
        noise = np.random.randint(-20, 20, test_image.shape).astype(np.int16)
        test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        image_path = "test_fiber_image.png"
        cv2.imwrite(image_path, test_image)
        print(f"✓ Test image saved as {image_path}")
    
    # Run analysis
    analyzer = IntegratedFiberAnalyzer(debug=True)
    results = analyzer.analyze_image(image_path)
    
    # Save results
    analyzer.save_results()
    
    # Print summary
    if results['success']:
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Image: {results['image_path']}")
        print(f"Center: {results['center_detection']['final_center']}")
        print(f"Core Radius: {results['boundary_detection']['final_core_radius']}")
        print(f"Cladding Radius: {results['boundary_detection']['final_cladding_radius']}")
        print(f"Quality Score: {results['validation']['overall_quality']:.2f}")
        print(f"Edge Points: {results['edge_analysis']['num_edge_points']}")
    else:
        print(f"\n✗ Analysis failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
