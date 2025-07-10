#!/usr/bin/env python3
"""
Modular Fiber Optic Defect Detection Library - Index and Documentation
=====================================================================

This file provides an index and documentation for all modularized functions
extracted from the original complex fiber optic defect detection scripts.

Author: Modular Analysis Team
Version: 1.0
Created: December 2024

OVERVIEW
========
This library contains 18 standalone, self-contained modules extracted from
multiple legacy fiber optic analysis scripts. Each module implements specific
advanced computer vision and machine learning techniques for defect detection
and analysis.

MODULES AVAILABLE
================

1. Statistical Analysis
   --------------------
   - statistical_outlier_detection.py: Z-score, IQR, MAD, Grubbs test, etc.
   - lof_detection.py: Local Outlier Factor anomaly detection

2. Image Enhancement
   ------------------
   - adaptive_thresholding.py: Multiple adaptive thresholding techniques
   - anisotropic_diffusion.py: Edge-preserving smoothing
   - illumination_correction.py: Homomorphic filtering, Retinex, rolling ball

3. Morphological Analysis
   ----------------------
   - morphological_analysis.py: Advanced morphological operations
   - blob_detection.py: Multi-scale blob detection (LoG, DoG, DoH)

4. Pattern Detection
   -----------------
   - scratch_detection.py: Multiple scratch detection algorithms
   - do2mr_detection.py: Difference of Min-Max Ranking detection
   - lei_scratch_detection.py: Linear Enhancement Inspector

5. Texture and Frequency Analysis
   ------------------------------
   - texture_analysis.py: GLCM, LBP, and advanced texture features
   - frequency_domain_analysis.py: FFT, DCT, spectral analysis
   - wavelet_analysis.py: Multi-wavelet decomposition and analysis
   - gabor_filter_bank.py: Oriented texture detection using Gabor filters

6. Machine Learning
   ----------------
   - ml_anomaly_detection.py: Isolation Forest, One-Class SVM, DBSCAN

7. Region Detection
   ----------------
   - robust_mask_generation.py: Multi-method fiber region detection

8. Ensemble Methods
   ----------------
   - ensemble_detection.py: Intelligent fusion of multiple detection methods

9. Quality Assessment
   ------------------
   - quality_metrics.py: Comprehensive quality metrics (SSIM, contrast, etc.)

USAGE EXAMPLES
==============
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def demonstrate_all_modules():
    """
    Demonstrate usage of all modules with a synthetic test image.
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("Modular Fiber Optic Defect Detection Library")
    print("=" * 50)
    
    # Create a comprehensive test image
    def create_comprehensive_test_image(size=512):
        """Create a test image with various defects and features."""
        image = np.ones((size, size), dtype=np.uint8) * 128
        
        # Add fiber structure
        center = (size // 2, size // 2)
        cv2.circle(image, center, size // 3, 160, -1)  # Cladding
        cv2.circle(image, center, size // 20, 100, -1)  # Core
        
        # Add various defects
        # Scratches
        cv2.line(image, (100, 100), (400, 150), 80, 3)
        cv2.line(image, (150, 300), (200, 450), 85, 2)
        
        # Pits and digs
        cv2.circle(image, (200, 200), 8, 60, -1)
        cv2.circle(image, (350, 180), 12, 65, -1)
        cv2.circle(image, (280, 350), 6, 70, -1)
        
        # Contamination spots
        for i in range(5):
            center = (np.random.randint(150, 350), np.random.randint(150, 350))
            radius = np.random.randint(8, 15)
            cv2.circle(image, center, radius, 90, -1)
        
        # Add noise and texture
        noise = np.random.normal(0, 8, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
    
    test_image = create_comprehensive_test_image()
    
    print("Testing individual modules:\\n")
    
    # 1. Test Statistical Outlier Detection
    try:
        from statistical_outlier_detection import StatisticalOutlierDetector
        detector = StatisticalOutlierDetector()
        result = detector.detect_outliers(test_image)
        print("✓ Statistical Outlier Detection: SUCCESS")
        print(f"  Found {np.sum(result['combined_outliers']) // 255} outlier pixels")
    except Exception as e:
        print(f"✗ Statistical Outlier Detection: FAILED - {e}")
    
    # 2. Test Adaptive Thresholding
    try:
        from adaptive_thresholding import AdaptiveThresholder
        thresholder = AdaptiveThresholder()
        result = thresholder.apply_all_methods(test_image)
        print("✓ Adaptive Thresholding: SUCCESS")
        print(f"  Applied {len(result)} different thresholding methods")
    except Exception as e:
        print(f"✗ Adaptive Thresholding: FAILED - {e}")
    
    # 3. Test Illumination Correction
    try:
        from illumination_correction import AdvancedIlluminationCorrector
        corrector = AdvancedIlluminationCorrector()
        result = corrector.correct_illumination(test_image, ['rolling_ball', 'clahe'])
        print("✓ Illumination Correction: SUCCESS")
        print(f"  Applied {len(result)} correction methods")
    except Exception as e:
        print(f"✗ Illumination Correction: FAILED - {e}")
    
    # 4. Test Gabor Filter Bank
    try:
        from gabor_filter_bank import GaborFilterBank
        gabor = GaborFilterBank()
        result = gabor.apply_filter_bank(test_image)
        print("✓ Gabor Filter Bank: SUCCESS")
        print(f"  Generated {len(gabor.filters)} Gabor filters")
    except Exception as e:
        print(f"✗ Gabor Filter Bank: FAILED - {e}")
    
    # 5. Test Ensemble Detection
    try:
        from ensemble_detection import EnsembleDetector
        detector = EnsembleDetector()
        result = detector.detect_ensemble(test_image)
        print("✓ Ensemble Detection: SUCCESS")
        print(f"  Combined {len(result['methods_used'])} detection methods")
    except Exception as e:
        print(f"✗ Ensemble Detection: FAILED - {e}")
    
    # 6. Test Robust Mask Generation
    try:
        from robust_mask_generation import RobustMaskGenerator
        generator = RobustMaskGenerator()
        masks, localization = generator.generate_masks(test_image)
        if masks and localization:
            print("✓ Robust Mask Generation: SUCCESS")
            print(f"  Method used: {localization['method']}")
        else:
            print("✗ Robust Mask Generation: FAILED - No masks generated")
    except Exception as e:
        print(f"✗ Robust Mask Generation: FAILED - {e}")
    
    # 7. Test more modules...
    module_tests = [
        ("DO2MR Detection", "do2mr_detection", "DO2MRDetector"),
        ("LEI Scratch Detection", "lei_scratch_detection", "LEIScratchDetector"),
        ("Blob Detection", "blob_detection", "BlobDetector"),
        ("Quality Metrics", "quality_metrics", "QualityMetrics"),
        ("ML Anomaly Detection", "ml_anomaly_detection", "MLAnomalyDetector"),
        ("Texture Analysis", "texture_analysis", "TextureAnalyzer"),
        ("Morphological Analysis", "morphological_analysis", "MorphologicalAnalyzer"),
        ("Wavelet Analysis", "wavelet_analysis", "WaveletAnalyzer"),
        ("Frequency Domain Analysis", "frequency_domain_analysis", "FrequencyDomainAnalyzer"),
        ("Anisotropic Diffusion", "anisotropic_diffusion", "AnisotropicDiffusion"),
        ("Scratch Detection", "scratch_detection", "ScratchDetector"),
        ("LOF Detection", "lof_detection", "LOFDetector"),
    ]
    
    for name, module_name, class_name in module_tests:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            instance = cls()
            print(f"✓ {name}: MODULE AVAILABLE")
        except Exception as e:
            print(f"✗ {name}: UNAVAILABLE - {e}")
    
    print(f"\\nTotal modules tested: {len(module_tests) + 6}")
    print("\\nFor detailed usage examples, run individual module demo functions.")


def list_all_modules():
    """List all available modules with descriptions."""
    modules_info = [
        {
            "name": "statistical_outlier_detection.py",
            "description": "Multiple statistical outlier detection methods",
            "key_functions": ["detect_outliers", "z_score_outliers", "iqr_outliers", "grubbs_test"],
            "use_case": "Finding statistically abnormal pixels or regions"
        },
        {
            "name": "adaptive_thresholding.py", 
            "description": "Various adaptive thresholding techniques",
            "key_functions": ["apply_all_methods", "otsu_threshold", "adaptive_gaussian"],
            "use_case": "Segmenting features under varying illumination"
        },
        {
            "name": "illumination_correction.py",
            "description": "Advanced illumination correction methods",
            "key_functions": ["rolling_ball_correction", "homomorphic_filtering", "multi_scale_retinex"],
            "use_case": "Normalizing uneven lighting conditions"
        },
        {
            "name": "gabor_filter_bank.py",
            "description": "Oriented texture detection using Gabor filters",
            "key_functions": ["apply_filter_bank", "detect_oriented_features"],
            "use_case": "Detecting scratches and oriented patterns"
        },
        {
            "name": "ensemble_detection.py",
            "description": "Intelligent fusion of multiple detection methods",
            "key_functions": ["detect_ensemble", "fuse_detections"],
            "use_case": "Robust detection by combining multiple algorithms"
        },
        {
            "name": "robust_mask_generation.py",
            "description": "Multi-method fiber region detection",
            "key_functions": ["generate_masks", "adaptive_threshold_method", "hough_circles_method"],
            "use_case": "Reliable fiber core/cladding/ferrule segmentation"
        },
        {
            "name": "do2mr_detection.py",
            "description": "Difference of Min-Max Ranking defect detection",
            "key_functions": ["detect_defects", "apply_do2mr_filter"],
            "use_case": "Detecting local intensity variations"
        },
        {
            "name": "lei_scratch_detection.py",
            "description": "Linear Enhancement Inspector for scratch detection",
            "key_functions": ["detect_scratches", "apply_lei_filter"],
            "use_case": "Specifically detecting linear scratches"
        },
        {
            "name": "blob_detection.py",
            "description": "Multi-scale blob detection (LoG, DoG, DoH)",
            "key_functions": ["detect_blobs", "log_detection", "dog_detection"],
            "use_case": "Finding circular defects like pits and digs"
        },
        {
            "name": "quality_metrics.py",
            "description": "Comprehensive image quality assessment",
            "key_functions": ["calculate_all_metrics", "ssim", "contrast_metrics"],
            "use_case": "Quantitative quality evaluation"
        },
        {
            "name": "ml_anomaly_detection.py",
            "description": "Machine learning based anomaly detection",
            "key_functions": ["detect_anomalies", "isolation_forest", "one_class_svm"],
            "use_case": "ML-based defect detection"
        },
        {
            "name": "texture_analysis.py",
            "description": "Advanced texture feature extraction",
            "key_functions": ["analyze_texture", "glcm_features", "lbp_features"],
            "use_case": "Characterizing surface texture properties"
        },
        {
            "name": "morphological_analysis.py",
            "description": "Advanced morphological operations",
            "key_functions": ["detect_defects", "top_hat_detection", "watershed_segmentation"],
            "use_case": "Shape-based defect detection"
        },
        {
            "name": "wavelet_analysis.py",
            "description": "Multi-wavelet decomposition and analysis",
            "key_functions": ["wavelet_decomposition", "detect_anomalies"],
            "use_case": "Multi-resolution defect analysis"
        },
        {
            "name": "frequency_domain_analysis.py",
            "description": "FFT, DCT, and spectral analysis",
            "key_functions": ["fft_analysis", "dct_analysis", "spectral_features"],
            "use_case": "Frequency-based pattern detection"
        },
        {
            "name": "anisotropic_diffusion.py",
            "description": "Edge-preserving smoothing filter",
            "key_functions": ["apply_diffusion", "perona_malik_diffusion"],
            "use_case": "Noise reduction while preserving edges"
        },
        {
            "name": "scratch_detection.py",
            "description": "Multiple scratch detection algorithms",
            "key_functions": ["detect_scratches", "hessian_detection", "ridge_detection"],
            "use_case": "Comprehensive scratch detection"
        },
        {
            "name": "lof_detection.py",
            "description": "Local Outlier Factor anomaly detection",
            "key_functions": ["detect_outliers", "calculate_lof_scores"],
            "use_case": "Density-based anomaly detection"
        }
    ]
    
    print("\\nAVAILABLE MODULES")
    print("=" * 50)
    
    for i, module in enumerate(modules_info, 1):
        print(f"{i:2d}. {module['name']}")
        print(f"    Description: {module['description']}")
        print(f"    Key Functions: {', '.join(module['key_functions'])}")
        print(f"    Use Case: {module['use_case']}")
        print()


def generate_quick_start_guide():
    """Generate a quick start guide for using the modules."""
    guide = '''
QUICK START GUIDE
================

1. Basic Usage Pattern:
   ```python
   # Import the module
   from module_name import ClassName
   
   # Create instance
   detector = ClassName()
   
   # Apply to image
   result = detector.main_function(image)
   
   # Visualize (most modules include visualization)
   detector.visualize_results(result, "output.png")
   ```

2. Example Workflow for Comprehensive Analysis:
   ```python
   import cv2
   import numpy as np
   
   # Load image
   image = cv2.imread("fiber_image.png", cv2.IMREAD_GRAYSCALE)
   
   # Step 1: Correct illumination
   from illumination_correction import AdvancedIlluminationCorrector
   corrector = AdvancedIlluminationCorrector()
   corrected = corrector.correct_illumination(image, ['clahe', 'retinex'])
   
   # Step 2: Generate fiber masks
   from robust_mask_generation import RobustMaskGenerator
   generator = RobustMaskGenerator()
   masks, localization = generator.generate_masks(corrected['clahe'])
   
   # Step 3: Apply ensemble detection
   from ensemble_detection import EnsembleDetector
   detector = EnsembleDetector()
   detections = detector.detect_ensemble(corrected['clahe'], mask=masks['Fiber'])
   
   # Step 4: Calculate quality metrics
   from quality_metrics import QualityMetrics
   quality = QualityMetrics()
   metrics = quality.calculate_all_metrics(image, detections['ensemble_mask'])
   ```

3. Running Module Demos:
   Each module includes a demo function. Run any module directly:
   ```bash
   python module_name.py
   ```

4. Common Parameters:
   - Most detection functions accept threshold parameters
   - Visualization functions accept save_path parameter
   - Many modules support region masking
   - Advanced modules offer multiple algorithm variants

5. Dependencies:
   - Core: numpy, opencv-python, matplotlib
   - ML modules: scikit-learn
   - Some modules: scipy, skimage (noted in module headers)
'''
    print(guide)


def main():
    """Main function to display library information."""
    print(__doc__)
    
    # List all modules
    list_all_modules()
    
    # Generate quick start guide
    generate_quick_start_guide()
    
    # Demonstrate modules (if requested)
    print("\\nTo test all modules, uncomment the following line and run:")
    print("# demonstrate_all_modules()")
    
    print(f"\\nLibrary location: {Path(__file__).parent}")
    print("\\nFor individual module documentation, see the docstrings in each file.")


if __name__ == "__main__":
    main()
