#!/usr/bin/env python3
"""
Integrated Neural Network module for Fiber Optics Analysis
"the neural network does the segmentation and reference comparison and anomaly detection internally"
"it separates, does anomaly detection and comparison, and then also does reconstruction"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from fiber_config import get_config
from fiber_logger import get_logger
from fiber_tensor_processor import TensorProcessor
from fiber_feature_extractor import MultiScaleFeatureExtractor, TrendAnalyzer
from fiber_reference_comparator import SimilarityCalculator
from fiber_anomaly_detector import AnomalyDetector

class FiberOpticsIntegratedNetwork(nn.Module):
    """
    Complete integrated neural network that performs all operations internally
    "the neural network itself in order to classify the image will separate the 
    image into its features (or edges)"
    """
    
    def __init__(self):
        super().__init__()
        print(f"[{datetime.now()}] Initializing FiberOpticsIntegratedNetwork")
        print(f"[{datetime.now()}] Previous script: anomaly_detector.py")
        
        self.config = get_config()
        self.logger = get_logger("FiberOpticsIntegratedNetwork")
        
        self.logger.log_class_init("FiberOpticsIntegratedNetwork")
        
        # Initialize tensor processor
        self.tensor_processor = TensorProcessor()
        
        # Multi-scale feature extraction with simultaneous analysis
        self.feature_extractor = MultiScaleFeatureExtractor()
        
        # Trend analyzer for region classification
        self.trend_analyzer = TrendAnalyzer()
        
        # Integrated segmentation network
        self.segmentation_net = nn.Sequential(
            nn.Conv2d(sum(self.config.FEATURE_CHANNELS), 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1)  # core, cladding, ferrule
        )
        
        # Reference comparison network
        self.reference_encoder = nn.Sequential(
            nn.Conv2d(sum(self.config.FEATURE_CHANNELS), 512, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256)
        )
        
        # Learnable reference embeddings
        # "try to see which of the reference images in the reference folder"
        self.num_references = 1000
        self.reference_embeddings = nn.Parameter(torch.randn(self.num_references, 256))
        
        # Anomaly detection network (integrated)
        self.anomaly_detector = AnomalyDetector()
        
        # Reconstruction decoder
        # "and then also does reconstruction"
        decoder_channels = [512, 256, 128, 64, 32, 3]
        self.decoder = nn.ModuleList()
        
        in_ch = sum(self.config.FEATURE_CHANNELS)
        for out_ch in decoder_channels:
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True) if out_ch != 3 else nn.Sigmoid()
            ))
            in_ch = out_ch
        
        # Equation parameter networks
        # "I=Ax1+Bx2+Cx3... =S(R)"
        self.equation_adjuster = nn.Sequential(
            nn.Linear(5, 16),  # 5 coefficients
            nn.ReLU(),
            nn.Linear(16, 5),
            nn.Sigmoid()
        )
        
        self.logger.info("FiberOpticsIntegratedNetwork initialized")
        print(f"[{datetime.now()}] FiberOpticsIntegratedNetwork initialized successfully")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with integrated processing
        "I want each feature to not only look for comparisons but also look for 
        anomalies while comparing"
        """
        batch_size = x.shape[0]
        
        # Calculate gradient and position information
        gradient_info = self.tensor_processor.calculate_gradient_intensity(x)
        position_info = self.tensor_processor.calculate_pixel_positions(x.shape)
        
        # Multi-scale feature extraction with simultaneous analysis
        extraction_results = self.feature_extractor(x, gradient_info, position_info)
        
        # Combine multi-scale features
        # Resize all to same size and concatenate
        target_size = extraction_results['features'][0].shape[-2:]
        combined_features = []
        
        for features in extraction_results['features']:
            if features.shape[-2:] != target_size:
                features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
            combined_features.append(features)
        
        combined_features = torch.cat(combined_features, dim=1)
        
        # Segmentation
        segmentation_logits = self.segmentation_net(combined_features)
        region_probs = F.softmax(segmentation_logits, dim=1)
        
        # Trend analysis for region validation
        trend_results = self.trend_analyzer(
            gradient_info['gradient_map'],
            position_info['radial_positions'],
            region_probs
        )
        
        # Reference comparison
        feature_embedding = self.reference_encoder(combined_features)
        
        # Compare to all reference embeddings
        similarities = F.cosine_similarity(
            feature_embedding.unsqueeze(1),
            self.reference_embeddings.unsqueeze(0),
            dim=2
        )
        
        # Find best matching reference
        best_similarities, best_indices = similarities.max(dim=1)
        
        # Check threshold
        # "the program must achieve over .7"
        meets_threshold = best_similarities > self.config.SIMILARITY_THRESHOLD
        
        # Reconstruction
        reconstructed = self._reconstruct(combined_features)
        
        # Anomaly detection using reconstruction
        # "Im subtracting the resulting classification with the original input to find anomalies"
        reconstruction_diff = torch.abs(reconstructed - x)
        
        # Integrated anomaly detection
        anomaly_results = self.anomaly_detector(
            x,
            reconstructed,
            gradient_info['gradient_map'],
            self._get_region_boundaries(region_probs)
        )
        
        # Combine all anomaly sources
        combined_anomaly_map = torch.max(
            anomaly_results['anomaly_map'],
            reconstruction_diff.mean(dim=1)
        )
        
        # Apply trend-based suppression
        # "when a feature of an image follows this line its classified as region when it doesn't its classified as a defect"
        final_anomaly_map = combined_anomaly_map * (1 - trend_results['trend_adherence'].squeeze(1))
        
        # Adjust equation coefficients based on performance
        current_coeffs = torch.tensor(list(self.config.EQUATION_COEFFICIENTS.values()), 
                                    dtype=torch.float32, device=x.device)
        adjusted_coeffs = self.equation_adjuster(current_coeffs)
        
        # Calculate final similarity score using equation
        # "I=Ax1+Bx2+Cx3... =S(R)"
        similarity_components = torch.stack([
            best_similarities,  # x1: reference similarity
            trend_results['trend_adherence'].mean(dim=[1, 2]),  # x2: trend adherence
            1 - final_anomaly_map.mean(dim=[1, 2]),  # x3: inverse anomaly score
            region_probs.max(dim=1)[0].mean(dim=[1, 2]),  # x4: segmentation confidence
            F.cosine_similarity(x.view(batch_size, -1), 
                              reconstructed.view(batch_size, -1), dim=1)  # x5: reconstruction similarity
        ], dim=1)
        
        final_similarity = (adjusted_coeffs * similarity_components).sum(dim=1)
        
        return {
            # Segmentation results
            'segmentation': segmentation_logits,
            'region_probs': region_probs,
            
            # Reference comparison
            'best_reference_indices': best_indices,
            'reference_similarities': best_similarities,
            'meets_threshold': meets_threshold,
            
            # Anomaly detection
            'anomaly_map': final_anomaly_map,
            'defect_type_probs': anomaly_results['defect_type_probs'],
            
            # Reconstruction
            'reconstruction': reconstructed,
            'reconstruction_error': reconstruction_diff.mean(dim=1),
            
            # Trend analysis
            'trend_adherence': trend_results['trend_adherence'],
            
            # Final similarity
            'final_similarity': final_similarity,
            'similarity_components': similarity_components,
            'adjusted_coefficients': adjusted_coeffs,
            
            # Multi-scale features for analysis
            'multi_scale_features': extraction_results['features'],
            'scale_weights': extraction_results['scale_weights']
        }
    
    def _reconstruct(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from features"""
        x = features
        
        # Progressively upsample
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            
            # Skip connections could be added here if we stored encoder features
        
        return x
    
    def _get_region_boundaries(self, region_probs: torch.Tensor) -> torch.Tensor:
        """Extract region boundaries from probability maps"""
        # Get hard predictions
        regions = region_probs.argmax(dim=1)
        
        # Simple edge detection
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                            dtype=torch.float32, device=regions.device).view(1, 1, 3, 3)
        
        boundaries = F.conv2d(
            regions.float().unsqueeze(1),
            kernel,
            padding=1
        )
        
        boundaries = torch.abs(boundaries)
        boundaries = (boundaries > 0).float()
        
        return boundaries

class IntegratedAnalysisPipeline:
    """
    Complete analysis pipeline using the integrated network
    """
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing IntegratedAnalysisPipeline")
        
        self.config = get_config()
        self.logger = get_logger("IntegratedAnalysisPipeline")
        self.tensor_processor = TensorProcessor()
        
        self.logger.log_class_init("IntegratedAnalysisPipeline")
        
        # Initialize integrated network
        self.network = FiberOpticsIntegratedNetwork()
        
        # Move to device
        self.device = self.config.get_device()
        self.network = self.network.to(self.device)
        
        # Set to evaluation mode by default
        self.network.eval()
        
        self.logger.info("IntegratedAnalysisPipeline initialized")
        print(f"[{datetime.now()}] IntegratedAnalysisPipeline initialized successfully")
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze a single image using the integrated network
        """
        self.logger.log_process_start(f"Analyzing image: {image_path}")
        
        # Load and tensorize image
        image_tensor = self.tensor_processor.image_to_tensor(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Run through integrated network
        with torch.no_grad():
            results = self.network(image_tensor)
        
        # Post-process results
        processed_results = self._post_process_results(results)
        
        # Log key metrics
        self.logger.log_similarity_check(
            results['final_similarity'][0].item(),
            self.config.SIMILARITY_THRESHOLD,
            f"ref_{results['best_reference_indices'][0].item()}"
        )
        
        # Count defects
        anomaly_threshold = self.config.ANOMALY_THRESHOLD
        num_anomaly_pixels = (results['anomaly_map'] > anomaly_threshold).sum().item()
        
        if num_anomaly_pixels > 0:
            self.logger.log_anomaly_detection(
                num_anomaly_pixels,
                self._get_anomaly_locations(results['anomaly_map'][0], anomaly_threshold)
            )
        
        self.logger.log_region_classification({
            'core': results['region_probs'][0, 0].mean().item(),
            'cladding': results['region_probs'][0, 1].mean().item(),
            'ferrule': results['region_probs'][0, 2].mean().item()
        })
        
        self.logger.log_process_end(f"Analyzing image: {image_path}")
        
        return processed_results
    
    def _post_process_results(self, results: Dict) -> Dict:
        """Post-process network outputs"""
        processed = {}
        
        # Convert tensors to numpy for export
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.cpu().numpy()
            else:
                processed[key] = value
        
        # Add summary statistics
        processed['summary'] = {
            'final_similarity_score': float(results['final_similarity'][0].item()),
            'meets_threshold': bool(results['meets_threshold'][0].item()),
            'primary_region': ['core', 'cladding', 'ferrule'][results['region_probs'][0].argmax(dim=0)[128, 128].item()],
            'anomaly_score': float(results['anomaly_map'][0].mean().item()),
            'max_anomaly_score': float(results['anomaly_map'][0].max().item()),
            'reconstruction_error': float(results['reconstruction_error'][0].mean().item())
        }
        
        # Equation coefficients
        processed['equation_info'] = {
            'coefficients': {
                'A': float(results['adjusted_coefficients'][0].item()),
                'B': float(results['adjusted_coefficients'][1].item()),
                'C': float(results['adjusted_coefficients'][2].item()),
                'D': float(results['adjusted_coefficients'][3].item()),
                'E': float(results['adjusted_coefficients'][4].item())
            },
            'components': {
                'reference_similarity': float(results['similarity_components'][0, 0].item()),
                'trend_adherence': float(results['similarity_components'][0, 1].item()),
                'anomaly_inverse': float(results['similarity_components'][0, 2].item()),
                'segmentation_confidence': float(results['similarity_components'][0, 3].item()),
                'reconstruction_similarity': float(results['similarity_components'][0, 4].item())
            }
        }
        
        return processed
    
    def _get_anomaly_locations(self, anomaly_map: torch.Tensor, threshold: float) -> List[Tuple[int, int]]:
        """Get locations of anomalies above threshold"""
        anomaly_mask = anomaly_map > threshold
        locations = torch.where(anomaly_mask)
        
        # Return first 10 locations
        return list(zip(locations[0].tolist()[:10], locations[1].tolist()[:10]))
    
    def export_results(self, results: Dict, output_path: str):
        """
        Export results to file
        "the program will spit out an anomaly or defect map which will just be a 
        matrix of the pixel intensities of the defects"
        """
        self.logger.log_function_entry("export_results", output_path=output_path)
        
        # Create results matrix
        anomaly_matrix = results['anomaly_map'][0]
        
        # Save as minimal format
        np.savetxt(output_path, anomaly_matrix, fmt='%.4f')
        
        # Also save complete results as .npz
        complete_path = output_path.replace('.txt', '_complete.npz')
        np.savez_compressed(complete_path, **results)
        
        self.logger.info(f"Results exported to: {output_path}")
        self.logger.info(f"Complete results saved to: {complete_path}")
        
        self.logger.log_function_exit("export_results")

# Test the integrated network
if __name__ == "__main__":
    pipeline = IntegratedAnalysisPipeline()
    logger = get_logger("IntegratedNetworkTest")
    
    logger.log_process_start("Integrated Network Test")
    
    # Create test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_path = "test_image.png"
    cv2.imwrite(test_path, test_image)
    
    # Analyze
    results = pipeline.analyze_image(test_path)
    
    # Log summary
    logger.info("\nAnalysis Summary:")
    for key, value in results['summary'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nEquation Components:")
    for key, value in results['equation_info']['components'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Export results
    pipeline.export_results(results, "test_results.txt")
    
    # Clean up
    import os
    os.remove(test_path)
    
    logger.log_process_end("Integrated Network Test")
    logger.log_script_transition("integrated_network.py", "trainer.py")
    
    print(f"[{datetime.now()}] Integrated network test completed")
    print(f"[{datetime.now()}] Next script: trainer.py")
