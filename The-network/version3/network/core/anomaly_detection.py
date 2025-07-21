import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import cv2

from ..utils.logger import get_logger
from ..config.config import get_config
from .similarity import SimilarityCalculator


@dataclass
class AnomalyResult:
    """Container for anomaly detection results"""
    defect_map: torch.Tensor  # Binary mask of defects
    anomaly_heatmap: torch.Tensor  # Continuous anomaly scores
    defect_locations: List[Tuple[int, int, int, int]]  # Bounding boxes
    defect_types: List[str]
    confidence_scores: List[float]
    combined_anomaly_score: float


class AnomalyDetector:
    """
    Detects anomalies and defects in fiber optic images.
    "I've made code and scripts for detecting anomalies, I need help in integrating 
    the logic and processes into the neural network as well"
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.device = self.config.get_device()
        self.similarity_calculator = SimilarityCalculator()
        
        # Defect type patterns
        self.defect_patterns = {
            'scratch': {
                'gradient_threshold': 0.5,
                'aspect_ratio_range': (3, 20),  # Scratches are elongated
                'min_area': 50
            },
            'contamination': {
                'intensity_deviation': 0.3,
                'circularity_range': (0.4, 1.0),
                'min_area': 100
            },
            'pit': {
                'depth_threshold': 0.4,
                'circularity_range': (0.7, 1.0),
                'min_area': 20
            },
            'crack': {
                'gradient_threshold': 0.6,
                'linearity_threshold': 0.8,
                'min_length': 30
            }
        }
        
        self.logger.info("Initialized AnomalyDetector")
    
    def detect_anomalies(self, input_tensor: torch.Tensor,
                        reference_tensor: torch.Tensor,
                        segmentation_masks: Optional[Dict[str, torch.Tensor]] = None) -> AnomalyResult:
        """
        Main anomaly detection function.
        "take the absolute value of the difference between the two and it will also 
        look at the structural similarity index between the two to find the defects 
        or anomalies"
        """
        self.logger.log_image_processing("anomaly_detection", "detection", "started")
        
        # Calculate difference map
        difference_map = self.similarity_calculator.calculate_difference_map(
            input_tensor, reference_tensor
        )
        
        # Calculate structural similarity map
        ssim_map = self._calculate_ssim_map(input_tensor, reference_tensor)
        
        # Calculate local anomaly heatmap
        local_anomaly_map = self._calculate_local_anomaly_heatmap(
            input_tensor, reference_tensor
        )
        
        # Combine anomaly indicators
        # "it will compare the local anomaly heatmap with the structural similarity 
        # index and combine them to find the total anomalies or defects"
        combined_anomaly_map = self._combine_anomaly_indicators(
            difference_map, ssim_map, local_anomaly_map
        )
        
        # Apply region-aware filtering if segmentation is provided
        if segmentation_masks:
            combined_anomaly_map = self._apply_region_aware_filtering(
                combined_anomaly_map, segmentation_masks
            )
        
        # Threshold to get binary defect map
        defect_map = self._threshold_anomaly_map(combined_anomaly_map)
        
        # Identify individual defects and their properties
        defect_locations, defect_types, confidence_scores = self._identify_defects(
            defect_map, combined_anomaly_map, input_tensor
        )
        
        # Calculate overall anomaly score
        combined_score = combined_anomaly_map.mean().item()
        
        # Log results
        self.logger.log_defect_detection(
            "current_image",
            f"{len(defect_locations)} defects",
            str(defect_locations),
            combined_score
        )
        
        result = AnomalyResult(
            defect_map=defect_map,
            anomaly_heatmap=combined_anomaly_map,
            defect_locations=defect_locations,
            defect_types=defect_types,
            confidence_scores=confidence_scores,
            combined_anomaly_score=combined_score
        )
        
        return result
    
    def _calculate_ssim_map(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Calculate pixel-wise SSIM map"""
        # Convert to grayscale for SSIM
        if tensor1.dim() == 3 and tensor1.shape[0] == 3:
            gray1 = 0.299 * tensor1[0] + 0.587 * tensor1[1] + 0.114 * tensor1[2]
            gray2 = 0.299 * tensor2[0] + 0.587 * tensor2[1] + 0.114 * tensor2[2]
        else:
            gray1 = tensor1.squeeze()
            gray2 = tensor2.squeeze()
        
        # Parameters for SSIM
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Calculate local means
        kernel_size = 11
        kernel = self._gaussian_kernel(kernel_size).to(self.device)
        
        mu1 = F.conv2d(gray1.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2)
        mu2 = F.conv2d(gray2.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate local variances and covariance
        sigma1_sq = F.conv2d((gray1**2).unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2) - mu1_sq
        sigma2_sq = F.conv2d((gray2**2).unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2) - mu2_sq
        sigma12 = F.conv2d((gray1*gray2).unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Convert to anomaly score (1 - SSIM)
        anomaly_map = 1 - ssim_map.squeeze()
        
        return anomaly_map
    
    def _calculate_local_anomaly_heatmap(self, input_tensor: torch.Tensor,
                                       reference_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate local anomaly heatmap.
        "it will also look at the local anomaly heat map"
        """
        # Calculate local statistics for both images
        window_size = 15
        
        input_stats = self._calculate_local_statistics(input_tensor, window_size)
        reference_stats = self._calculate_local_statistics(reference_tensor, window_size)
        
        # Compare local statistics
        mean_diff = torch.abs(input_stats['mean'] - reference_stats['mean'])
        std_diff = torch.abs(input_stats['std'] - reference_stats['std'])
        gradient_diff = torch.abs(input_stats['gradient'] - reference_stats['gradient'])
        
        # Normalize differences
        mean_diff = mean_diff / (reference_stats['mean'] + 1e-8)
        std_diff = std_diff / (reference_stats['std'] + 1e-8)
        gradient_diff = gradient_diff / (reference_stats['gradient'] + 1e-8)
        
        # Combine into anomaly heatmap
        anomaly_heatmap = (mean_diff + std_diff + gradient_diff) / 3.0
        
        return anomaly_heatmap
    
    def _calculate_local_statistics(self, tensor: torch.Tensor, 
                                   window_size: int) -> Dict[str, torch.Tensor]:
        """Calculate local statistics using sliding window"""
        # Convert to grayscale
        if tensor.dim() == 3 and tensor.shape[0] == 3:
            gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
        else:
            gray = tensor.squeeze()
        
        # Pad the image
        pad = window_size // 2
        padded = F.pad(gray.unsqueeze(0).unsqueeze(0), [pad, pad, pad, pad], mode='reflect')
        
        # Calculate local statistics
        kernel = torch.ones(1, 1, window_size, window_size).to(self.device) / (window_size ** 2)
        
        # Local mean
        local_mean = F.conv2d(padded, kernel)
        
        # Local standard deviation
        local_sq_mean = F.conv2d(padded ** 2, kernel)
        local_std = torch.sqrt(local_sq_mean - local_mean ** 2 + 1e-8)
        
        # Local gradient magnitude
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        
        grad_x = F.conv2d(padded, sobel_x, padding=1)
        grad_y = F.conv2d(padded, sobel_y, padding=1)
        local_gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return {
            'mean': local_mean.squeeze(),
            'std': local_std.squeeze(),
            'gradient': local_gradient.squeeze()
        }
    
    def _combine_anomaly_indicators(self, difference_map: torch.Tensor,
                                  ssim_map: torch.Tensor,
                                  local_anomaly_map: torch.Tensor) -> torch.Tensor:
        """
        Combine multiple anomaly indicators into a single map.
        "compare the local anomaly heatmap with the structural similarity index 
        and combine them to find the total anomalies"
        """
        # Normalize each map to [0, 1]
        diff_norm = (difference_map - difference_map.min()) / (difference_map.max() - difference_map.min() + 1e-8)
        ssim_norm = (ssim_map - ssim_map.min()) / (ssim_map.max() - ssim_map.min() + 1e-8)
        local_norm = (local_anomaly_map - local_anomaly_map.min()) / (local_anomaly_map.max() - local_anomaly_map.min() + 1e-8)
        
        # Weighted combination based on config
        combined = (
            0.3 * diff_norm.mean(dim=0) +  # Average across channels if needed
            self.config.STRUCTURAL_SIMILARITY_WEIGHT * ssim_norm +
            self.config.LOCAL_ANOMALY_WEIGHT * local_norm
        )
        
        return combined
    
    def _apply_region_aware_filtering(self, anomaly_map: torch.Tensor,
                                    segmentation_masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Filter anomalies based on region boundaries.
        "it will understand that the regions themselves or the change in regions 
        within an image such as the change in pixels between the core to cladding 
        or cladding to ferrule are not anomalies"
        """
        filtered_map = anomaly_map.clone()
        
        # Create region boundary mask
        boundary_mask = self._create_boundary_mask(segmentation_masks)
        
        # Reduce anomaly scores at boundaries
        filtered_map = filtered_map * (1 - 0.8 * boundary_mask)
        
        # Apply region-specific thresholds
        for region, mask in segmentation_masks.items():
            if region in ['core', 'cladding', 'ferrule']:
                # Different sensitivity for different regions
                region_weight = {
                    'core': 1.2,      # More sensitive in core
                    'cladding': 1.0,  # Normal sensitivity
                    'ferrule': 0.8    # Less sensitive in ferrule
                }.get(region, 1.0)
                
                filtered_map = torch.where(mask, filtered_map * region_weight, filtered_map)
        
        return filtered_map
    
    def _create_boundary_mask(self, segmentation_masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create mask for region boundaries"""
        # Combine all masks
        all_masks = torch.stack(list(segmentation_masks.values()))
        
        # Find boundaries using morphological operations
        kernel = torch.ones(1, 1, 3, 3).to(self.device)
        
        boundaries = torch.zeros_like(segmentation_masks['core'])
        
        for mask in segmentation_masks.values():
            # Dilate and subtract to find boundaries
            dilated = F.conv2d(mask.float().unsqueeze(0).unsqueeze(0), 
                              kernel, padding=1).squeeze() > 0
            boundary = dilated.float() - mask.float()
            boundaries = boundaries + boundary
        
        return (boundaries > 0).float()
    
    def _threshold_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """
        Threshold anomaly map to create binary defect map.
        "if the entire program does not achieve over .7 whenever referring to the 
        reference no matter if its trying to find a feature"
        """
        # Adaptive thresholding based on statistics
        mean_val = anomaly_map.mean()
        std_val = anomaly_map.std()
        
        # Use configurable threshold
        threshold = mean_val + 2 * std_val
        threshold = max(threshold, self.config.ANOMALY_HEAT_MAP_THRESHOLD)
        
        defect_map = anomaly_map > threshold
        
        # Morphological operations to clean up
        defect_map = defect_map.float()
        
        # Remove small noise
        kernel = torch.ones(1, 1, 3, 3).to(self.device)
        defect_map = F.conv2d(defect_map.unsqueeze(0).unsqueeze(0), 
                             kernel, padding=1).squeeze() > 4
        
        return defect_map.float()
    
    def _identify_defects(self, defect_map: torch.Tensor,
                         anomaly_heatmap: torch.Tensor,
                         input_tensor: torch.Tensor) -> Tuple[List, List, List]:
        """Identify individual defects and classify them"""
        # Convert to numpy for connected component analysis
        defect_np = defect_map.cpu().numpy().astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(defect_np)
        
        defect_locations = []
        defect_types = []
        confidence_scores = []
        
        for i in range(1, num_labels):  # Skip background (0)
            # Get bounding box
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Skip very small defects
            if area < 10:
                continue
            
            defect_locations.append((x, y, w, h))
            
            # Extract defect region
            defect_region = anomaly_heatmap[y:y+h, x:x+w]
            
            # Classify defect type
            defect_type = self._classify_defect(defect_region, w, h, area)
            defect_types.append(defect_type)
            
            # Calculate confidence
            confidence = defect_region.mean().item()
            confidence_scores.append(confidence)
            
            self.logger.log_defect_detection(
                "current_image",
                defect_type,
                f"({x}, {y}, {w}, {h})",
                confidence
            )
        
        return defect_locations, defect_types, confidence_scores
    
    def _classify_defect(self, defect_region: torch.Tensor, 
                        width: int, height: int, area: int) -> str:
        """Classify defect based on its characteristics"""
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        
        # Calculate circularity
        perimeter = 2 * (width + height)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Check patterns
        if aspect_ratio > 3 and area > 50:
            return 'scratch'
        elif circularity > 0.7 and area > 20:
            return 'pit'
        elif aspect_ratio > 5:
            return 'crack'
        else:
            return 'contamination'
    
    def _gaussian_kernel(self, kernel_size: int) -> torch.Tensor:
        """Create Gaussian kernel for convolution"""
        sigma = kernel_size / 3.0
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def generate_defect_report(self, anomaly_result: AnomalyResult) -> Dict:
        """
        Generate detailed defect report.
        "if it doesn't it will detail why it didn't and try to locate the anomalies"
        """
        report = {
            'total_defects': len(anomaly_result.defect_locations),
            'defect_types': {},
            'severity_score': anomaly_result.combined_anomaly_score,
            'affected_area_percentage': 0.0,
            'defect_details': []
        }
        
        # Count defect types
        for defect_type in anomaly_result.defect_types:
            report['defect_types'][defect_type] = report['defect_types'].get(defect_type, 0) + 1
        
        # Calculate affected area
        total_pixels = anomaly_result.defect_map.numel()
        defect_pixels = anomaly_result.defect_map.sum().item()
        report['affected_area_percentage'] = (defect_pixels / total_pixels) * 100
        
        # Add details for each defect
        for i, (loc, dtype, conf) in enumerate(zip(
            anomaly_result.defect_locations,
            anomaly_result.defect_types,
            anomaly_result.confidence_scores
        )):
            report['defect_details'].append({
                'id': i,
                'type': dtype,
                'location': loc,
                'confidence': conf,
                'severity': 'high' if conf > 0.8 else 'medium' if conf > 0.5 else 'low'
            })
        
        return report