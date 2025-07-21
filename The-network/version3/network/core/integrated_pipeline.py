import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

from ..utils.logger import get_logger
from ..config.config import get_config
from ..models.integrated_fiber_nn import IntegratedFiberOpticsNN
from ..data_loaders.tensor_loader import TensorDataLoader
from ..utils.results_exporter import ResultsExporter


class IntegratedPipeline:
    """
    Complete pipeline using the integrated neural network.
    "the neural network does the segmentation and reference comparison and 
    anomaly detection internally"
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.config = get_config()
        self.logger = get_logger()
        self.device = self.config.get_device()
        
        # Initialize model
        self.model = IntegratedFiberOpticsNN().to(self.device)
        
        # Load pretrained weights if available
        if model_path and model_path.exists():
            self.logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        
        # Initialize components
        self.data_loader = TensorDataLoader()
        self.results_exporter = ResultsExporter()
        
        self.logger.info("Initialized IntegratedPipeline")
    
    def process_image(self, image_tensor: torch.Tensor) -> Dict:
        """
        Process a single image through the integrated pipeline.
        "an image will be selected from a dataset folder... the image will then be 
        tensorized so it can better be computed by pytorch"
        """
        start_time = time.time()
        
        # Ensure correct shape and device
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Log processing start
        self.logger.info("Processing image through integrated neural network...")
        
        # Forward pass through the integrated model
        # "that tensor will be compared to other tensors in the correlational data 
        # and at the same time it will go through the neural network for classification"
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Extract results
        results = self._extract_results(outputs, image_tensor)
        
        # Check similarity threshold
        # "if the entire program does not achieve over .7 whenever referring to the 
        # reference no matter if its trying to find a feature"
        if not outputs['meets_threshold'][0]:
            self.logger.warning(
                f"Similarity {outputs['best_similarity'][0]:.4f} below threshold "
                f"{self.config.SIMILARITY_THRESHOLD}. Investigating anomalies..."
            )
            # "if it doesn't it will detail why it didn't and try to locate the anomalies"
            results['anomaly_analysis'] = self._analyze_low_similarity(outputs, image_tensor)
        
        # Log completion
        processing_time = time.time() - start_time
        self.logger.info(f"Processing completed in {processing_time:.3f}s")
        results['processing_time'] = processing_time
        
        return results
    
    def process_batch(self, image_tensors: torch.Tensor) -> List[Dict]:
        """
        Process multiple images.
        "or multiple images for batch processing or realtime processing"
        """
        self.logger.info(f"Processing batch of {image_tensors.shape[0]} images...")
        
        batch_results = []
        
        # Process entire batch at once for efficiency
        with torch.no_grad():
            outputs = self.model(image_tensors.to(self.device))
        
        # Extract results for each image
        for i in range(image_tensors.shape[0]):
            # Extract single image outputs
            single_outputs = {
                key: value[i:i+1] if isinstance(value, torch.Tensor) else value
                for key, value in outputs.items()
            }
            
            results = self._extract_results(single_outputs, image_tensors[i:i+1])
            batch_results.append(results)
        
        return batch_results
    
    def _extract_results(self, outputs: Dict[str, torch.Tensor], 
                        original_image: torch.Tensor) -> Dict:
        """
        Extract and format results from model outputs.
        "the for every image the program will spit out an anomaly or defect map"
        """
        results = {}
        
        # Segmentation results
        # "after the network converges the features of the image to either core 
        # cladding and ferrule"
        segmentation = outputs['segmentation'][0]  # Remove batch dimension
        results['segmentation'] = {
            'probabilities': segmentation.cpu().numpy(),
            'regions': {
                'core': outputs['region_masks']['core'][0].cpu().numpy(),
                'cladding': outputs['region_masks']['cladding'][0].cpu().numpy(),
                'ferrule': outputs['region_masks']['ferrule'][0].cpu().numpy()
            },
            'dominant_region': ['core', 'cladding', 'ferrule'][segmentation.argmax(dim=0)[64, 64].item()]
        }
        
        # Reference matching results
        # "now that the process has either the three regions or a most similar image 
        # from the reference bank"
        results['reference_matching'] = {
            'best_reference_idx': outputs['best_reference_idx'][0].item(),
            'similarity_score': outputs['best_similarity'][0].item(),
            'meets_threshold': outputs['meets_threshold'][0].item()
        }
        
        # Anomaly detection results
        # "the process will take the input tensorized region or image and the 
        # reference image and take the absolute value of the difference"
        anomaly_map = outputs['anomaly_map'][0].cpu().numpy()
        difference_map = outputs['difference_map'][0].cpu().numpy()
        
        # Create defect overlay
        # "a matrix of the pixel intensities of the defects in a heightened contrast 
        # on top of the matrix of the input image"
        defect_overlay = self._create_defect_overlay(
            original_image[0].cpu().numpy(),
            anomaly_map
        )
        
        results['anomaly_detection'] = {
            'anomaly_map': anomaly_map,
            'difference_map': difference_map,
            'defect_overlay': defect_overlay,
            'num_anomalies': int((anomaly_map > self.config.ANOMALY_HEAT_MAP_THRESHOLD).sum()),
            'max_anomaly_score': float(anomaly_map.max()),
            'mean_anomaly_score': float(anomaly_map.mean())
        }
        
        # Feature analysis
        results['feature_analysis'] = {
            'num_features_extracted': len(outputs['all_features']),
            'similarity_patterns': [s.mean(dim=[2,3]).cpu().numpy() 
                                   for s in outputs['all_similarities']]
        }
        
        # Equation parameters at inference time
        results['equation_parameters'] = self.model.get_equation_parameters()
        
        return results
    
    def _create_defect_overlay(self, original_image: np.ndarray, 
                              anomaly_map: np.ndarray) -> np.ndarray:
        """Create visualization with heightened contrast for defects"""
        # Ensure correct shape
        if original_image.shape[0] == 3:  # CHW to HWC
            original_image = original_image.transpose(1, 2, 0)
        
        # Normalize to [0, 1]
        if original_image.max() > 1:
            original_image = original_image / 255.0
        
        # Create RGB overlay
        overlay = original_image.copy()
        if overlay.ndim == 2:
            overlay = np.stack([overlay] * 3, axis=2)
        
        # Apply heightened contrast to anomaly regions
        contrast_factor = 2.5
        anomaly_mask = anomaly_map > self.config.ANOMALY_HEAT_MAP_THRESHOLD
        
        # Enhance contrast in anomaly regions
        for c in range(3):
            channel = overlay[:, :, c]
            enhanced = channel * (1 + contrast_factor * anomaly_mask)
            overlay[:, :, c] = np.clip(enhanced, 0, 1)
        
        # Add red tint to anomalies
        overlay[:, :, 0] = np.maximum(overlay[:, :, 0], anomaly_map * 0.8)
        
        return overlay
    
    def _analyze_low_similarity(self, outputs: Dict[str, torch.Tensor], 
                              original_image: torch.Tensor) -> Dict:
        """
        Detailed analysis when similarity is below threshold.
        "if it doesn't it will detail why it didn't and try to locate the anomalies"
        """
        analysis = {}
        
        # Identify which regions have the most anomalies
        anomaly_map = outputs['anomaly_map'][0]
        region_masks = outputs['region_masks']
        
        region_anomalies = {}
        for region_name, mask in region_masks.items():
            mask = mask[0]
            region_anomaly_score = (anomaly_map * mask).sum() / (mask.sum() + 1e-8)
            region_anomalies[region_name] = region_anomaly_score.item()
        
        analysis['region_anomaly_scores'] = region_anomalies
        analysis['most_affected_region'] = max(region_anomalies, key=region_anomalies.get)
        
        # Identify specific anomaly locations
        anomaly_threshold = anomaly_map.mean() + 2 * anomaly_map.std()
        anomaly_locations = torch.where(anomaly_map > anomaly_threshold)
        
        analysis['anomaly_locations'] = {
            'count': len(anomaly_locations[0]),
            'coordinates': [(int(y), int(x)) for y, x in zip(anomaly_locations[0], anomaly_locations[1])][:10]  # Top 10
        }
        
        # Analyze why similarity is low
        similarity_score = outputs['best_similarity'][0].item()
        analysis['similarity_breakdown'] = {
            'overall_score': similarity_score,
            'below_by': self.config.SIMILARITY_THRESHOLD - similarity_score,
            'likely_cause': 'defects' if analysis['anomaly_locations']['count'] > 10 else 'poor_match'
        }
        
        # Check if it's a region classification issue
        segmentation_confidence = outputs['segmentation'][0].max(dim=0)[0].mean().item()
        if segmentation_confidence < 0.6:
            analysis['classification_issue'] = True
            analysis['segmentation_confidence'] = segmentation_confidence
        
        return analysis
    
    def export_results(self, image_id: str, results: Dict, 
                      original_tensor: torch.Tensor) -> Path:
        """
        Export results in minimal format.
        "it will export this into a results folder and each result matrix will be 
        the smallest files type and size possible"
        """
        # Create AnomalyResult-like object for compatibility
        from ..core.anomaly_detection import AnomalyResult
        
        # Convert anomaly map to binary defect map
        anomaly_map = results['anomaly_detection']['anomaly_map']
        defect_map = anomaly_map > self.config.ANOMALY_HEAT_MAP_THRESHOLD
        
        # Find defect locations (simplified)
        defect_locations = []
        if defect_map.any():
            from scipy import ndimage
            labeled, num_features = ndimage.label(defect_map)
            for i in range(1, num_features + 1):
                coords = np.where(labeled == i)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    defect_locations.append((int(x_min), int(y_min), 
                                           int(x_max - x_min), int(y_max - y_min)))
        
        anomaly_result = AnomalyResult(
            defect_map=torch.from_numpy(defect_map),
            anomaly_heatmap=torch.from_numpy(anomaly_map),
            defect_locations=defect_locations,
            defect_types=['anomaly'] * len(defect_locations),
            confidence_scores=[results['anomaly_detection']['mean_anomaly_score']] * len(defect_locations),
            combined_anomaly_score=results['anomaly_detection']['mean_anomaly_score']
        )
        
        # Export using results exporter
        output_path = self.results_exporter.export_anomaly_result(
            image_id,
            anomaly_result,
            original_tensor.squeeze(0) if original_tensor.dim() == 4 else original_tensor,
            additional_data={
                'segmentation': results['segmentation'],
                'reference_matching': results['reference_matching'],
                'equation_parameters': results['equation_parameters'],
                'anomaly_analysis': results.get('anomaly_analysis', {})
            }
        )
        
        return output_path
    
    def continuous_processing(self, image_source):
        """
        Process images continuously in real-time.
        "or realtime processing"
        """
        self.logger.info("Starting continuous processing mode...")
        
        frame_count = 0
        total_time = 0
        
        try:
            while True:
                # Get next image (implementation depends on source)
                image_tensor = self._get_next_image(image_source)
                
                if image_tensor is None:
                    break
                
                # Process image
                start_time = time.time()
                results = self.process_image(image_tensor)
                processing_time = time.time() - start_time
                
                frame_count += 1
                total_time += processing_time
                
                # Log performance
                if frame_count % 30 == 0:
                    avg_time = total_time / frame_count
                    fps = 1 / avg_time
                    self.logger.info(f"Processed {frame_count} frames. Avg time: {avg_time:.3f}s, FPS: {fps:.1f}")
                
                # Export if anomalies found
                if results['anomaly_detection']['num_anomalies'] > 0:
                    self.export_results(f"frame_{frame_count}", results, image_tensor)
                
        except KeyboardInterrupt:
            self.logger.info("Continuous processing stopped by user")
        
        self.logger.info(f"Processed {frame_count} frames total")
    
    def _get_next_image(self, image_source):
        """Get next image from source (placeholder)"""
        # This would be implemented based on the actual image source
        # (camera, video file, image directory, etc.)
        return None