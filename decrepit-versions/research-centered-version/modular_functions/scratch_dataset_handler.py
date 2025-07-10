#!/usr/bin/env python3
"""
Scratch Dataset Handler Module
==============================
Standalone module for handling external scratch datasets for validation and augmentation.

Usage:
    python scratch_dataset_handler.py --dataset path/to/scratch_dataset --image test_image.jpg --validate
    python scratch_dataset_handler.py --dataset path/to/scratch_dataset --image test_image.jpg --augment --output probability_map.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScratchDatasetHandler:
    """
    Manages scratch dataset for validation and augmentation of scratch detection.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset handler.
        
        Args:
            dataset_path: Path to the scratch dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.scratch_samples = []
        self.scratch_features = []
        
        if self.dataset_path.exists():
            self._load_dataset()
        else:
            logger.warning(f"Scratch dataset path not found: {dataset_path}")
    
    def _load_dataset(self) -> None:
        """Load scratch samples from the dataset."""
        # Look for common scratch image patterns
        patterns = ["*_s.bmp", "*_scratch.png", "*_scratch.jpg", "scratch_*.png", "scratch_*.jpg"]
        scratch_files = []
        
        for pattern in patterns:
            scratch_files.extend(self.dataset_path.glob(pattern))
        
        # Remove duplicates and sort
        scratch_files = sorted(set(scratch_files))
        
        for file_path in scratch_files:
            try:
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    self.scratch_samples.append({
                        'image': img,
                        'path': file_path,
                        'id': file_path.stem
                    })
                    
                    # Extract features for matching
                    features = self._extract_scratch_features(img)
                    self.scratch_features.append(features)
                    
            except Exception as e:
                logger.warning(f"Failed to load scratch sample {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.scratch_samples)} scratch samples from dataset")
    
    def _extract_scratch_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from a scratch image for matching.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        try:
            # Basic image statistics
            features['mean_intensity'] = float(np.mean(image))
            features['std_intensity'] = float(np.std(image))
            features['image_size'] = image.shape
            
            # Edge detection
            edges = cv2.Canny(image, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / edges.size)
            
            # Hough line detection for linear scratches
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            features['line_count'] = len(lines) if lines is not None else 0
            
            # Aspect ratio analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 10:
                    rect = cv2.minAreaRect(largest_contour)
                    w, h = rect[1]
                    if h > 0:
                        features['aspect_ratio'] = float(w / h)
                    else:
                        features['aspect_ratio'] = 1.0
                else:
                    features['aspect_ratio'] = 1.0
            else:
                features['aspect_ratio'] = 1.0
            
            # Texture features using Gabor filters
            gabor_features = []
            for theta in np.arange(0, np.pi, np.pi / 4):
                try:
                    kernel = cv2.getGaborKernel((21, 21), 4, theta, 10, 0.5, 0)
                    filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                    gabor_features.append(float(np.mean(np.abs(filtered))))
                except:
                    gabor_features.append(0.0)
            features['gabor'] = gabor_features
            
            # Morphological features for scratch-like structures
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            features['blackhat_mean'] = float(np.mean(blackhat))
            
            # Directional variance (scratches often have preferred orientation)
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            features['horizontal_edge_strength'] = float(np.mean(np.abs(sobel_x)))
            features['vertical_edge_strength'] = float(np.mean(np.abs(sobel_y)))
            
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            # Return minimal features to avoid crashes
            features = {
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'edge_density': 0.0,
                'line_count': 0,
                'aspect_ratio': 1.0,
                'gabor': [0.0] * 4,
                'blackhat_mean': 0.0,
                'horizontal_edge_strength': 0.0,
                'vertical_edge_strength': 0.0
            }
        
        return features
    
    def _compute_similarity(self, f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
        """
        Compute similarity score between two feature dictionaries.
        
        Args:
            f1: First feature dictionary
            f2: Second feature dictionary
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Extract numerical features
            vals1 = [
                f1.get("edge_density", 0),
                f1.get("line_count", 0),
                f1.get("blackhat_mean", 0),
                f1.get("aspect_ratio", 1.0),
                f1.get("horizontal_edge_strength", 0),
                f1.get("vertical_edge_strength", 0)
            ] + f1.get("gabor", [])
            
            vals2 = [
                f2.get("edge_density", 0),
                f2.get("line_count", 0),
                f2.get("blackhat_mean", 0),
                f2.get("aspect_ratio", 1.0),
                f2.get("horizontal_edge_strength", 0),
                f2.get("vertical_edge_strength", 0)
            ] + f2.get("gabor", [])
            
            # Ensure same length
            min_len = min(len(vals1), len(vals2))
            if min_len == 0:
                return 0.0
            
            a = np.array(vals1[:min_len], dtype=np.float32)
            b = np.array(vals2[:min_len], dtype=np.float32)
            
            # Normalize to prevent large values from dominating
            a_norm = a / (np.linalg.norm(a) + 1e-8)
            b_norm = b / (np.linalg.norm(b) + 1e-8)
            
            # Compute cosine similarity
            similarity = np.dot(a_norm, b_norm)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            return 0.0
    
    def augment_scratch_detection(
        self, 
        image: np.ndarray, 
        threshold: float = 0.7,
        max_matches: int = 5
    ) -> np.ndarray:
        """
        Augment scratch detection using the dataset.
        
        Args:
            image: Input image to analyze
            threshold: Similarity threshold for matching
            max_matches: Maximum number of dataset matches to use
            
        Returns:
            Probability map of scratch locations
        """
        if not self.scratch_samples:
            logger.warning("No scratch samples available for augmentation")
            return np.zeros_like(image, dtype=np.float32)
        
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        
        scratch_probability = np.zeros((h, w), dtype=np.float32)
        
        try:
            # Extract features from input image
            input_features = self._extract_scratch_features(image_gray)
            
            # Find similar scratches in dataset
            similar_scratches = []
            for i, features in enumerate(self.scratch_features):
                similarity = self._compute_similarity(input_features, features)
                if similarity > threshold:
                    similar_scratches.append((i, similarity))
            
            # Sort by similarity and take top matches
            similar_scratches.sort(key=lambda x: x[1], reverse=True)
            similar_scratches = similar_scratches[:max_matches]
            
            if similar_scratches:
                logger.info(f"Found {len(similar_scratches)} similar scratches in dataset")
                
                # Use template matching with similar scratches
                for idx, similarity in similar_scratches:
                    scratch_img = self.scratch_samples[idx]['image']
                    
                    # Resize scratch image if needed
                    if scratch_img.shape != image_gray.shape:
                        scratch_img_resized = cv2.resize(scratch_img, (w, h), 
                                                       interpolation=cv2.INTER_NEAREST)
                    else:
                        scratch_img_resized = scratch_img
                    
                    # Apply template matching
                    try:
                        result = cv2.matchTemplate(image_gray, scratch_img_resized, cv2.TM_CCOEFF_NORMED)
                        
                        # Resize result to match image dimensions if needed
                        if result.shape != (h, w):
                            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # Add weighted result to probability map
                        scratch_probability += result * similarity * 0.5
                    except Exception as e:
                        logger.warning(f"Template matching failed for sample {idx}: {e}")
                        continue
            else:
                logger.info("No similar scratches found in dataset")
            
            # Normalize probability map
            if np.max(scratch_probability) > 0:
                scratch_probability = scratch_probability / np.max(scratch_probability)
        
        except Exception as e:
            logger.error(f"Error in scratch augmentation: {e}")
            return np.zeros((h, w), dtype=np.float32)
        
        return scratch_probability
    
    def validate_detection(
        self, 
        detected_scratches: List[Dict], 
        image: np.ndarray
    ) -> List[Dict]:
        """
        Validate detected scratches against the dataset.
        
        Args:
            detected_scratches: List of detected scratch dictionaries
            image: Original image
            
        Returns:
            Updated list with validation scores
        """
        if not self.scratch_samples:
            logger.warning("No scratch samples available for validation")
            return detected_scratches
        
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        
        for scratch in detected_scratches:
            try:
                # Extract region around scratch
                bbox_x = scratch.get('bbox_x_px', scratch.get('centroid_x_px', 0))
                bbox_y = scratch.get('bbox_y_px', scratch.get('centroid_y_px', 0))
                bbox_w = scratch.get('bbox_w_px', 10)
                bbox_h = scratch.get('bbox_h_px', 10)
                
                x, y, w, h = int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h)
                
                # Ensure valid coordinates
                x = max(0, min(x, image_gray.shape[1] - 1))
                y = max(0, min(y, image_gray.shape[0] - 1))
                w = max(1, min(w, image_gray.shape[1] - x))
                h = max(1, min(h, image_gray.shape[0] - y))
                
                scratch_region = image_gray[y:y+h, x:x+w]
                
                if scratch_region.size > 0:
                    # Compare with dataset
                    scratch_features = self._extract_scratch_features(scratch_region)
                    
                    max_similarity = 0
                    for features in self.scratch_features:
                        similarity = self._compute_similarity(scratch_features, features)
                        max_similarity = max(max_similarity, similarity)
                    
                    # Add validation score
                    scratch['dataset_validation_score'] = float(max_similarity)
                    
                    # Boost confidence if high similarity
                    if max_similarity > 0.8:
                        current_confidence = scratch.get('confidence_score', 0.5)
                        scratch['confidence_score'] = min(1.0, current_confidence * 1.2)
                        logger.info(f"Boosted confidence for scratch {scratch.get('defect_id', 'unknown')} "
                                  f"due to high dataset similarity: {max_similarity:.3f}")
                else:
                    scratch['dataset_validation_score'] = 0.0
                    
            except Exception as e:
                logger.warning(f"Error validating scratch {scratch.get('defect_id', 'unknown')}: {e}")
                scratch['dataset_validation_score'] = 0.0
        
        return detected_scratches
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if not self.scratch_samples:
            return {"total_samples": 0, "status": "No samples loaded"}
        
        # Calculate statistics
        sizes = [sample['image'].shape for sample in self.scratch_samples]
        widths = [s[1] for s in sizes]
        heights = [s[0] for s in sizes]
        
        feature_stats = {}
        if self.scratch_features:
            # Aggregate feature statistics
            all_edge_densities = [f.get('edge_density', 0) for f in self.scratch_features]
            all_line_counts = [f.get('line_count', 0) for f in self.scratch_features]
            all_aspect_ratios = [f.get('aspect_ratio', 1.0) for f in self.scratch_features]
            
            feature_stats = {
                "edge_density": {
                    "mean": float(np.mean(all_edge_densities)),
                    "std": float(np.std(all_edge_densities))
                },
                "line_count": {
                    "mean": float(np.mean(all_line_counts)),
                    "std": float(np.std(all_line_counts))
                },
                "aspect_ratio": {
                    "mean": float(np.mean(all_aspect_ratios)),
                    "std": float(np.std(all_aspect_ratios))
                }
            }
        
        return {
            "total_samples": len(self.scratch_samples),
            "dataset_path": str(self.dataset_path),
            "image_dimensions": {
                "width": {"min": min(widths), "max": max(widths), "mean": float(np.mean(widths))},
                "height": {"min": min(heights), "max": max(heights), "mean": float(np.mean(heights))}
            },
            "feature_statistics": feature_stats,
            "status": "Loaded successfully"
        }
    
    def save_augmentation_result(
        self, 
        probability_map: np.ndarray, 
        output_path: Path,
        original_image: Optional[np.ndarray] = None
    ) -> bool:
        """
        Save augmentation probability map.
        
        Args:
            probability_map: Probability map from augmentation
            output_path: Path to save the result
            original_image: Optional original image for overlay
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Normalize and convert to 8-bit
            prob_8bit = (probability_map * 255).astype(np.uint8)
            
            # Save probability map
            cv2.imwrite(str(output_path), prob_8bit)
            
            # Save overlay if original image provided
            if original_image is not None:
                overlay_path = output_path.parent / f"{output_path.stem}_overlay{output_path.suffix}"
                
                if len(original_image.shape) == 3:
                    overlay = original_image.copy()
                else:
                    overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                
                # Create red overlay for high probability areas
                high_prob_mask = probability_map > 0.5
                overlay[high_prob_mask] = [0, 0, 255]  # Red overlay
                
                cv2.imwrite(str(overlay_path), overlay)
                logger.info(f"Saved overlay to {overlay_path}")
            
            logger.info(f"Saved augmentation result to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save augmentation result: {e}")
            return False


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Scratch Dataset Handler for Fiber Optic Inspection")
    parser.add_argument("--dataset", required=True, help="Path to scratch dataset directory")
    parser.add_argument("--image", help="Path to input image for processing")
    parser.add_argument("--detections", help="Path to JSON file with detected scratches")
    parser.add_argument("--output", help="Path for output results")
    parser.add_argument("--validate", action="store_true", help="Validate detections against dataset")
    parser.add_argument("--augment", action="store_true", help="Generate augmentation probability map")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    
    args = parser.parse_args()
    
    # Initialize handler
    handler = ScratchDatasetHandler(args.dataset)
    
    # Show statistics if requested
    if args.stats:
        stats = handler.get_dataset_statistics()
        print(json.dumps(stats, indent=2))
        return
    
    # Process image if provided
    if args.image:
        if not Path(args.image).exists():
            logger.error(f"Input image does not exist: {args.image}")
            sys.exit(1)
        
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"Could not load image from {args.image}")
            sys.exit(1)
        
        # Augmentation mode
        if args.augment:
            probability_map = handler.augment_scratch_detection(image, args.threshold)
            
            if args.output:
                output_path = Path(args.output)
                success = handler.save_augmentation_result(probability_map, output_path, image)
                if success:
                    logger.info(f"Augmentation results saved to {output_path}")
                else:
                    sys.exit(1)
            else:
                logger.info(f"Generated probability map with max value: {np.max(probability_map):.3f}")
        
        # Validation mode
        if args.validate:
            if not args.detections:
                logger.error("--detections is required for validation mode")
                sys.exit(1)
            
            try:
                with open(args.detections, 'r') as f:
                    detections_data = json.load(f)
                
                # Extract scratches from various possible formats
                if "characterized_defects" in detections_data:
                    scratches = [d for d in detections_data["characterized_defects"] 
                               if d.get("classification") == "Scratch"]
                elif "detections" in detections_data:
                    scratches = detections_data["detections"]
                else:
                    scratches = detections_data if isinstance(detections_data, list) else []
                
                logger.info(f"Validating {len(scratches)} scratch detections")
                validated_scratches = handler.validate_detection(scratches, image)
                
                # Save results
                if args.output:
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_path, 'w') as f:
                        json.dump(validated_scratches, f, indent=2)
                    
                    logger.info(f"Validation results saved to {output_path}")
                
                # Print summary
                validation_scores = [s.get('dataset_validation_score', 0) for s in validated_scratches]
                if validation_scores:
                    avg_score = np.mean(validation_scores)
                    max_score = np.max(validation_scores)
                    logger.info(f"Validation summary - Average score: {avg_score:.3f}, Max score: {max_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error loading detections: {e}")
                sys.exit(1)
    
    else:
        logger.info("No image provided. Use --stats to see dataset information.")


if __name__ == "__main__":
    main()
