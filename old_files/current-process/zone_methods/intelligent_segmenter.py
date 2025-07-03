#!/usr/bin/env python3
"""
Intelligent Segmenter - Fixed version with fallback for missing correlation data
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class IntelligentSegmenter:
    """
    An advanced segmentation system that uses correlation data when available,
    but falls back to robust default methods when data is missing.
    """

    def __init__(self, data_path: str = ".", knowledge_base_name: str = "segmentation_knowledge.json"):
        """
        Initializes the IntelligentSegmenter.
        """
        self.data_path = Path(data_path)
        self.knowledge_base_path = self.data_path / knowledge_base_name

        # Configuration
        self.contrast_threshold_low = 35
        self.contrast_threshold_high = 40

        # Features used for finding similar images
        self.similarity_features = [
            'mean', 'std_dev', 'entropy', 'fractal_dimension',
            'contrast_d1', 'homogeneity_d1', 'energy_d1'
        ]

        # Data Storage
        self.features_df = None
        self.correlations = {}
        self.knowledge_base = {}
        self.data_available = False

        # Try to load data, but don't fail if missing
        self._load_data()
        self._load_knowledge_base()

    def _load_data(self):
        """
        Attempts to load correlation matrices and features, but continues if missing.
        """
        try:
            # Try to load feature matrix
            features_path = self.data_path / "extracted_features.csv"
            if features_path.exists():
                import pandas as pd
                self.features_df = pd.read_csv(features_path, index_col=0)
                self.data_available = True
                print(f"  ✓ Loaded features for {len(self.features_df)} images.")
            else:
                print("  ! Feature data not found. Using standalone mode.")
        except Exception as e:
            print(f"  ! Could not load feature data: {e}. Using standalone mode.")
            self.features_df = None

    def _load_knowledge_base(self):
        """Loads the segmentation knowledge base if available."""
        try:
            if self.knowledge_base_path.exists():
                with open(self.knowledge_base_path, 'r') as f:
                    self.knowledge_base = json.load(f)
                print("  ✓ Knowledge base loaded successfully.")
            else:
                print("  ! Knowledge base not found. Using default parameters.")
        except Exception as e:
            print(f"  ! Error loading knowledge base: {e}")
            self.knowledge_base = {}

    def classify_image(self, image: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """
        Classifies the image based on contrast and spatial trends.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Calculate basic statistics
        mean_val = float(np.mean(gray_image))
        std_val = float(np.std(gray_image))
        entropy = self._calculate_entropy(gray_image)

        # Analyze spatial trends
        spatial_trends = self._analyze_spatial_trends(gray_image)

        # Primary classification based on contrast
        if std_val < self.contrast_threshold_low:
            classification = "Textural/Unified"
        elif std_val > self.contrast_threshold_high:
            classification = "Structured/Outlier"
        else:
            # Use vertical trend for tie-breaking
            vertical_b = spatial_trends['vertical']['b']
            classification = "Textural/Unified" if vertical_b < 0 else "Structured/Outlier"

        profile = {
            'classification': classification,
            'mean_grayscale': mean_val,
            'std_deviation': std_val,
            'entropy': entropy,
            'spatial_trends': spatial_trends
        }

        return classification, profile

    def find_similar_images(self, image_profile: Dict[str, Any], n_similar: int = 5) -> List[Tuple[str, float]]:
        """
        Finds similar images if database is available, otherwise returns empty list.
        """
        if self.features_df is None or self.features_df.empty or not self.data_available:
            return []

        try:
            import pandas as pd
            current_features = self._extract_feature_vector(image_profile)
            current_features_s = pd.Series(current_features)

            similarities = []
            for img_name, db_row in self.features_df.iterrows():
                try:
                    # Only use features that exist in both
                    common_features = [f for f in self.similarity_features if f in db_row.index and f in current_features_s.index]
                    if not common_features:
                        continue
                        
                    db_features = db_row[common_features].values
                    curr_subset = current_features_s[common_features].values
                    
                    distance = euclidean(
                        StandardScaler().fit_transform([curr_subset])[0],
                        StandardScaler().fit_transform([db_features])[0]
                    )
                    similarities.append((img_name, distance))
                except:
                    continue

            similarities.sort(key=lambda x: x[1])
            return similarities[:n_similar]
        except:
            return []

    def get_segmentation_parameters(self, classification: str, similar_images: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Determines optimal segmentation parameters based on classification.
        """
        if classification == "Textural/Unified":
            params = {
                'segmentation_method': self._segment_textural_unified,
                'preprocessing': {'blur_kernel': 5},
            }
        else:  # Structured/Outlier
            params = {
                'segmentation_method': self._segment_structured_outlier,
                'preprocessing': {'blur_kernel': 3},
            }
        
        return params
        
    def apply_spatial_trend_correction(self, image: np.ndarray, spatial_trends: Dict) -> np.ndarray:
        """
        Applies spatial trend correction to normalize brightness variations.
        """
        h, w = image.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Calculate inverse correction map
        h_trend = spatial_trends['horizontal']
        v_trend = spatial_trends['vertical']
        h_correction = -(h_trend['a'] * x_coords**2 + h_trend['b'] * x_coords)
        v_correction = -(v_trend['a'] * y_coords**2 + v_trend['b'] * y_coords)

        # Combine corrections
        total_correction = 0.5 * h_correction + 0.5 * v_correction
        corrected_image = np.clip(image.astype(float) + total_correction, 0, 255).astype(np.uint8)

        return corrected_image

    def segment_image(self, image: np.ndarray, image_name: str) -> Dict[str, Any]:
        """
        Main segmentation pipeline.
        """
        print(f"\nProcessing: {os.path.basename(image_name)}")

        # Step 1: Classify the image
        classification, profile = self.classify_image(image)
        print(f"  Classification: {classification}")

        # Step 2: Find similar images (if data available)
        similar_images = self.find_similar_images(profile)
        
        # Step 3: Get optimal parameters
        params = self.get_segmentation_parameters(classification, similar_images)

        # Step 4: Preprocess the image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        corrected_image = self.apply_spatial_trend_correction(gray_image, profile['spatial_trends'])
        blurred_image = cv2.GaussianBlur(
            corrected_image,
            (params['preprocessing']['blur_kernel'],) * 2, 0
        )

        # Step 5: Perform segmentation
        result = params['segmentation_method'](blurred_image, params)
        
        # Step 6: Validate and refine
        result = self._validate_and_refine(result, image.shape[:2], classification)
        
        return {
            'success': True,
            'classification': classification,
            'masks': result['masks'],
            'center': result['center'],
            'core_radius': result['core_radius'],
            'cladding_radius': result['cladding_radius'],
            'confidence': result['confidence'],
            'spatial_trends': profile['spatial_trends'],
            'similar_images': similar_images[:3]
        }

    def _segment_textural_unified(self, image: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Segmentation strategy for textural/unified images."""
        h, w = image.shape
        center = self._find_fiber_center_weighted(image)
        
        # Analyze radial intensity profile
        radial_profile = self._compute_radial_profile(image, center)
        gradients = np.gradient(gaussian_filter(radial_profile, sigma=2))
        
        # Find transitions
        core_radius = self._find_transition_radius(gradients, mode='negative')
        cladding_radius = self._find_transition_radius(
            gradients[int(core_radius):], mode='negative', offset=core_radius
        )
        
        # Ensure valid radii
        if cladding_radius <= core_radius:
            cladding_radius = core_radius * 2.5
            
        masks = self._create_circular_masks((h, w), center, core_radius, cladding_radius)
        return {
            'masks': masks, 
            'center': center, 
            'core_radius': core_radius, 
            'cladding_radius': cladding_radius, 
            'confidence': 0.85
        }

    def _segment_structured_outlier(self, image: np.ndarray, params: Dict) -> Dict[str, Any]:
        """Segmentation strategy for structured/outlier images."""
        h, w = image.shape
        
        # Use Hough Circle Transform
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=10, maxRadius=int(min(h, w) * 0.45)
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Sort by radius
            circles = sorted(circles, key=lambda x: x[2])
            
            if len(circles) >= 2:
                center = (int(np.mean([c[0] for c in circles])), int(np.mean([c[1] for c in circles])))
                core_radius = circles[0][2]
                cladding_radius = circles[-1][2]
            else:
                # Single circle found
                center = (circles[0][0], circles[0][1])
                cladding_radius = circles[0][2]
                core_radius = int(cladding_radius * 0.3)
        else:
            # Fallback
            center = self._find_fiber_center_weighted(image)
            core_radius = min(h, w) * 0.1
            cladding_radius = min(h, w) * 0.35
        
        masks = self._create_circular_masks((h, w), center, core_radius, cladding_radius)
        return {
            'masks': masks,
            'center': center,
            'core_radius': core_radius,
            'cladding_radius': cladding_radius,
            'confidence': 0.75
        }

    def _validate_and_refine(self, result: Dict, shape: Tuple[int, int], classification: str) -> Dict[str, Any]:
        """Validates and refines segmentation results."""
        h, w = shape
        max_dim = min(h, w)
        
        # Define constraints
        constraints = {
            'core_ratio': (0.05, 0.25),
            'cladding_ratio': (0.15, 0.45),
            'defaults': {'core': max_dim * 0.15, 'cladding': max_dim * 0.35}
        }
        
        # Adjust for structured images
        if classification == "Structured/Outlier":
            constraints['core_ratio'] = (0.05, 0.35)
            constraints['cladding_ratio'] = (0.15, 0.55)

        # Validate core radius
        core_ratio = result['core_radius'] / max_dim
        if not (constraints['core_ratio'][0] < core_ratio < constraints['core_ratio'][1]):
            result['core_radius'] = constraints['defaults']['core']
            result['confidence'] *= 0.8
            
        # Validate cladding radius
        cladding_ratio = result['cladding_radius'] / max_dim
        if not (constraints['cladding_ratio'][0] < cladding_ratio < constraints['cladding_ratio'][1]):
            result['cladding_radius'] = constraints['defaults']['cladding']
            result['confidence'] *= 0.8
        
        # Ensure proper ordering
        if result['core_radius'] >= result['cladding_radius']:
            result['cladding_radius'] = result['core_radius'] * 2.5
        
        # Recreate masks
        result['masks'] = self._create_circular_masks(shape, result['center'], result['core_radius'], result['cladding_radius'])
        return result

    # Helper Methods
    @staticmethod
    def _analyze_spatial_trends(image: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Fits quadratic models to brightness profiles."""
        h, w = image.shape
        h_profile = np.mean(image, axis=0)
        v_profile = np.mean(image, axis=1)
        
        h_coeffs = np.polyfit(np.arange(w), h_profile, 2)
        v_coeffs = np.polyfit(np.arange(h), v_profile, 2)
        
        return {
            'horizontal': {'a': h_coeffs[0], 'b': h_coeffs[1], 'c': h_coeffs[2]},
            'vertical': {'a': v_coeffs[0], 'b': v_coeffs[1], 'c': v_coeffs[2]}
        }

    @staticmethod
    def _calculate_entropy(image: np.ndarray) -> float:
        """Calculates Shannon entropy."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10))

    def _extract_feature_vector(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Creates a feature vector from profile."""
        return {
            'mean': profile['mean_grayscale'],
            'std_dev': profile['std_deviation'],
            'entropy': profile['entropy'],
            'fractal_dimension': 1.5,  # Placeholder
            'contrast_d1': profile['std_deviation'] * 0.8,
            'homogeneity_d1': 1.0 / (1.0 + profile['std_deviation'] * 0.1),
            'energy_d1': 1.0 / (1.0 + profile['entropy'])
        }

    @staticmethod
    def _find_fiber_center_weighted(image: np.ndarray) -> Tuple[int, int]:
        """Finds center using intensity-weighted centroid."""
        h, w = image.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        total_weight = np.sum(image)
        if total_weight == 0:
            return (w // 2, h // 2)
        
        cy = int(np.sum(y_coords * image) / total_weight)
        cx = int(np.sum(x_coords * image) / total_weight)
        return (cx, cy)

    @staticmethod
    def _compute_radial_profile(image: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Computes radial intensity profile."""
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        max_radius = int(dist_from_center.max())
        profile = np.zeros(max_radius)
        
        for r in range(max_radius):
            mask = (dist_from_center >= r) & (dist_from_center < r + 1)
            if np.any(mask):
                profile[r] = np.mean(image[mask])
        
        return gaussian_filter(profile, sigma=2)

    @staticmethod
    def _find_transition_radius(gradients: np.ndarray, mode: str, offset: int = 0) -> float:
        """Finds transition radius from gradient."""
        if gradients.size == 0:
            return offset + 10
        
        if mode == 'negative':
            threshold = np.mean(gradients) - 0.5 * np.std(gradients)
            candidates = np.where(gradients < threshold)[0]
        else:
            threshold = np.mean(gradients) + 0.5 * np.std(gradients)
            candidates = np.where(gradients > threshold)[0]
        
        return (candidates[0] + offset) if len(candidates) > 0 else (len(gradients) // 2 + offset)

    @staticmethod
    def _create_circular_masks(shape: Tuple[int, int], center: Tuple[int, int], 
                              core_r: float, clad_r: float) -> Dict[str, np.ndarray]:
        """Creates circular binary masks."""
        h, w = shape
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        core_mask = (dist <= core_r).astype(np.uint8)
        cladding_mask = ((dist > core_r) & (dist <= clad_r)).astype(np.uint8)
        ferrule_mask = (dist > clad_r).astype(np.uint8)
        
        return {
            'core': binary_fill_holes(core_mask).astype(np.uint8),
            'cladding': cladding_mask,
            'ferrule': ferrule_mask
        }


def run_intelligent_segmentation(image_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Wrapper function for UnifiedSegmentationSystem compatibility.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': f'Could not load image: {image_path}',
                'center': None,
                'core_radius': None,
                'cladding_radius': None,
                'confidence': 0.0
            }

        # Initialize segmenter
        segmenter = IntelligentSegmenter(data_path=".")
        result = segmenter.segment_image(image, image_path)

        # Convert to expected format
        return {
            'success': result.get('success', False),
            'center': result.get('center'),
            'core_radius': result.get('core_radius'),
            'cladding_radius': result.get('cladding_radius'),
            'confidence': result.get('confidence', 0.5),
            'error': result.get('error')
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'center': None,
            'core_radius': None,
            'cladding_radius': None,
            'confidence': 0.0
        }


def main():
    """Main function for standalone testing."""
    if len(sys.argv) < 3:
        print("Usage: python intelligent_segmenter.py <image_path> <output_dir>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run segmentation
    result = run_intelligent_segmentation(image_path, str(output_dir))
    
    if result['success']:
        print(f"✓ Segmentation complete.")
        print(f"  Center: {result['center']}")
        print(f"  Core radius: {result['core_radius']:.2f}")
        print(f"  Cladding radius: {result['cladding_radius']:.2f}")
        print(f"  Confidence: {result['confidence']:.2f}")
    else:
        print(f"✗ Segmentation failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()