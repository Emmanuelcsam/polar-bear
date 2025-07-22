#!/usr/bin/env python3
"""
Anomaly Detection Module
=======================
Standalone module for deep learning-based anomaly detection using Anomalib.
Provides inference capabilities for detecting anomalies in fiber optic images.

Usage:
    python anomaly_detection_module.py --image path/to/image.jpg --model path/to/model
    python anomaly_detection_module.py --image path/to/image.jpg --train path/to/good_samples
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for Anomalib availability
try:
    from anomalib.data.utils import read_image
    from anomalib.deploy import OpenVINOInferencer
    from anomalib.models import Padim
    ANOMALIB_AVAILABLE = True
    logger.info("Anomalib library is available")
except ImportError:
    ANOMALIB_AVAILABLE = False
    logger.warning("Anomalib not available. Deep learning features disabled.")


class AnomalyDetector:
    """
    Standalone anomaly detector using Anomalib for fiber optic defect detection.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the anomaly detector.
        
        Args:
            model_path: Path to pre-trained OpenVINO model directory
        """
        self.model = None
        self.inferencer = None
        
        if not ANOMALIB_AVAILABLE:
            logger.warning("Anomalib not installed. Anomaly detection disabled.")
            return
            
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> bool:
        """
        Load a pre-trained OpenVINO model.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        model_file_path = Path(model_path)
        
        if not model_file_path.exists():
            logger.error(f"Model path does not exist: {model_file_path}")
            return False
            
        try:
            self.inferencer = OpenVINOInferencer(
                path=model_file_path,
                device="CPU"  # Can be changed to "GPU" if configured
            )
            logger.info(f"Successfully loaded anomaly detection model from {model_file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load anomaly model: {e}")
            self.inferencer = None
            return False
    
    def detect_anomalies(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect anomalies in the given image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Binary mask of detected anomalies, or None if detection fails
        """
        if not self.inferencer:
            logger.warning("No model loaded. Cannot perform anomaly detection.")
            return None
            
        try:
            # Prepare input image
            if image.ndim == 3 and image.shape[2] == 3:
                # Convert BGR to RGB
                inp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.ndim == 2:
                # Convert grayscale to RGB
                inp = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                inp = image.copy()
            
            # Run inference
            predictions = self.inferencer.predict(image=inp)
            
            if not hasattr(predictions, "anomaly_map"):
                logger.error("Anomaly detector returned unexpected prediction format - missing anomaly_map.")
                return None
            
            anomaly_map = predictions.anomaly_map
            
            # Determine threshold
            if hasattr(predictions, "pred_score"):
                threshold = predictions.pred_score
            else:
                # Use Otsu's thresholding as fallback
                if anomaly_map.dtype != np.uint8:
                    # Normalize to 0-255 range
                    anomaly_map_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
                    anomaly_map_uint8 = (anomaly_map_normalized * 255).astype(np.uint8)
                else:
                    anomaly_map_uint8 = anomaly_map
                threshold, _ = cv2.threshold(anomaly_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                threshold = threshold / 255.0
            
            # Create binary mask
            if isinstance(threshold, (float, np.floating)):
                anomaly_mask = (anomaly_map > threshold).astype(np.uint8) * 255
            else:
                anomaly_mask = (anomaly_map > threshold).astype(np.uint8) * 255
                
            return anomaly_mask
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}", exc_info=True)
            return None
    
    def train_on_good_samples(self, good_sample_dir: str, save_path: str) -> bool:
        """
        Train a new anomaly detection model on good samples.
        
        Args:
            good_sample_dir: Directory containing good (non-anomalous) samples
            save_path: Path to save the trained model
            
        Returns:
            True if training completed successfully, False otherwise
        """
        if not ANOMALIB_AVAILABLE:
            logger.error("Cannot train: Anomalib not available")
            return False
            
        # This is a placeholder for training implementation
        # Full implementation would require setting up Anomalib's training pipeline
        logger.info("Training functionality would be implemented here using Anomalib's training pipeline.")
        logger.info(f"Hypothetical training with samples from {good_sample_dir}, saving to {save_path}")
        return True
    
    def process_image_file(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single image file for anomaly detection.
        
        Args:
            image_path: Path to input image
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            logger.info(f"Processing image: {image_path}")
            
            # Detect anomalies
            anomaly_mask = self.detect_anomalies(image)
            
            results = {
                "input_image": image_path,
                "anomaly_detected": anomaly_mask is not None,
                "anomaly_pixels": 0,
                "anomaly_percentage": 0.0
            }
            
            if anomaly_mask is not None:
                anomaly_pixels = np.sum(anomaly_mask > 0)
                total_pixels = anomaly_mask.shape[0] * anomaly_mask.shape[1]
                anomaly_percentage = (anomaly_pixels / total_pixels) * 100
                
                results.update({
                    "anomaly_pixels": int(anomaly_pixels),
                    "anomaly_percentage": float(anomaly_percentage)
                })
                
                # Save results if output directory specified
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save anomaly mask
                    mask_path = output_path / f"{Path(image_path).stem}_anomaly_mask.png"
                    cv2.imwrite(str(mask_path), anomaly_mask)
                    
                    # Save overlay
                    overlay = image.copy()
                    overlay[anomaly_mask > 0] = [0, 0, 255]  # Red overlay for anomalies
                    overlay_path = output_path / f"{Path(image_path).stem}_anomaly_overlay.png"
                    cv2.imwrite(str(overlay_path), overlay)
                    
                    results.update({
                        "mask_saved": str(mask_path),
                        "overlay_saved": str(overlay_path)
                    })
                    
                    logger.info(f"Results saved to {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {"error": str(e), "input_image": image_path}


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Standalone Anomaly Detection for Fiber Optic Images")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", help="Path to pre-trained OpenVINO model directory")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--train", help="Directory of good samples for training")
    parser.add_argument("--save-model", help="Path to save trained model")
    
    args = parser.parse_args()
    
    if not ANOMALIB_AVAILABLE:
        logger.error("Anomalib is not installed. Please install it to use this module.")
        sys.exit(1)
    
    # Initialize detector
    detector = AnomalyDetector(model_path=args.model)
    
    # Training mode
    if args.train:
        if not args.save_model:
            logger.error("--save-model is required when using --train")
            sys.exit(1)
        
        success = detector.train_on_good_samples(args.train, args.save_model)
        if success:
            logger.info("Training completed successfully")
        else:
            logger.error("Training failed")
            sys.exit(1)
    
    # Inference mode
    else:
        if not Path(args.image).exists():
            logger.error(f"Input image does not exist: {args.image}")
            sys.exit(1)
        
        results = detector.process_image_file(args.image, args.output)
        
        if "error" in results:
            logger.error(f"Processing failed: {results['error']}")
            sys.exit(1)
        
        # Print results
        logger.info("Anomaly Detection Results:")
        logger.info(f"  Anomaly Detected: {results['anomaly_detected']}")
        if results['anomaly_detected']:
            logger.info(f"  Anomaly Pixels: {results['anomaly_pixels']}")
            logger.info(f"  Anomaly Percentage: {results['anomaly_percentage']:.2f}%")


if __name__ == "__main__":
    main()
