import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from skimage.metrics import structural_similarity as ssim
import cv2

from ..utils.logger import get_logger
from ..config.config import get_config


class SimilarityCalculator:
    """
    Calculates similarity between input images and reference images.
    "it will take those three features of an image and try to see which of the 
    reference images in the reference folder of the database that the regions 
    most specifically represent"
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.device = self.config.get_device()
        
        # Cache for computed features
        self.feature_cache = {}
        
        self.logger.info("Initialized SimilarityCalculator")
    
    def calculate_similarity(self, input_tensor: torch.Tensor, 
                           reference_tensor: torch.Tensor,
                           method: str = 'combined') -> float:
        """
        Calculate similarity between two tensors.
        "the program must achieve over .7(S is always fractional to R)"
        """
        if method == 'cosine':
            similarity = self._cosine_similarity(input_tensor, reference_tensor)
        elif method == 'ssim':
            similarity = self._structural_similarity(input_tensor, reference_tensor)
        elif method == 'correlation':
            similarity = self._correlation_similarity(input_tensor, reference_tensor)
        elif method == 'combined':
            # Combine multiple similarity metrics
            cosine_sim = self._cosine_similarity(input_tensor, reference_tensor)
            ssim_sim = self._structural_similarity(input_tensor, reference_tensor)
            corr_sim = self._correlation_similarity(input_tensor, reference_tensor)
            
            # Weighted combination
            similarity = (0.4 * cosine_sim + 0.4 * ssim_sim + 0.2 * corr_sim)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        # Log if similarity is below threshold
        if similarity < self.config.SIMILARITY_THRESHOLD:
            self.logger.warning(f"Low similarity score: {similarity:.4f} (threshold: {self.config.SIMILARITY_THRESHOLD})")
        
        return similarity
    
    def _cosine_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Calculate cosine similarity between two tensors"""
        # Ensure same shape
        if tensor1.shape != tensor2.shape:
            tensor2 = F.interpolate(
                tensor2.unsqueeze(0) if tensor2.dim() == 3 else tensor2,
                size=tensor1.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            if tensor2.dim() == 4:
                tensor2 = tensor2.squeeze(0)
        
        # Flatten tensors
        flat1 = tensor1.flatten()
        flat2 = tensor2.flatten()
        
        # Normalize
        flat1_norm = F.normalize(flat1.unsqueeze(0), p=2, dim=1)
        flat2_norm = F.normalize(flat2.unsqueeze(0), p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = torch.mm(flat1_norm, flat2_norm.t()).item()
        
        return similarity
    
    def _structural_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """
        Calculate structural similarity index (SSIM).
        "it will also look at the structural similarity index between the two"
        """
        # Convert to numpy arrays
        img1 = tensor1.cpu().numpy()
        img2 = tensor2.cpu().numpy()
        
        # Ensure same shape
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2.transpose(1, 2, 0), 
                            (img1.shape[2], img1.shape[1]),
                            interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        
        # Convert to grayscale for SSIM calculation
        if img1.shape[0] == 3:
            gray1 = 0.299 * img1[0] + 0.587 * img1[1] + 0.114 * img1[2]
            gray2 = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]
        else:
            gray1 = img1.squeeze()
            gray2 = img2.squeeze()
        
        # Calculate SSIM
        similarity = ssim(gray1, gray2, data_range=1.0)
        
        return similarity
    
    def _correlation_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Calculate normalized cross-correlation"""
        # Ensure same shape
        if tensor1.shape != tensor2.shape:
            tensor2 = F.interpolate(
                tensor2.unsqueeze(0) if tensor2.dim() == 3 else tensor2,
                size=tensor1.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            if tensor2.dim() == 4:
                tensor2 = tensor2.squeeze(0)
        
        # Flatten and normalize
        flat1 = tensor1.flatten()
        flat2 = tensor2.flatten()
        
        # Remove mean
        flat1_centered = flat1 - flat1.mean()
        flat2_centered = flat2 - flat2.mean()
        
        # Calculate correlation
        correlation = torch.sum(flat1_centered * flat2_centered)
        normalization = torch.sqrt(torch.sum(flat1_centered**2) * torch.sum(flat2_centered**2))
        
        if normalization > 0:
            similarity = (correlation / normalization).item()
        else:
            similarity = 0.0
        
        return similarity
    
    def find_best_reference_match(self, input_features: Dict[str, torch.Tensor],
                                reference_features_list: List[Dict[str, torch.Tensor]],
                                region: str) -> Tuple[int, float, Dict[str, float]]:
        """
        Find the best matching reference for a given input.
        "try to see which of the reference images in the reference folder of the 
        database that the regions most specifically represent"
        """
        best_idx = -1
        best_similarity = -1
        best_scores = {}
        
        region_key = f'{region}_features'
        
        for idx, ref_features in enumerate(reference_features_list):
            if region_key not in input_features or region_key not in ref_features:
                continue
            
            # Calculate similarity for this region
            similarity = self.calculate_similarity(
                input_features[region_key],
                ref_features[region_key],
                method='combined'
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
                best_scores = {
                    'overall': similarity,
                    'cosine': self._cosine_similarity(
                        input_features[region_key],
                        ref_features[region_key]
                    ),
                    'ssim': self._structural_similarity(
                        input_features[region_key], 
                        ref_features[region_key]
                    ),
                    'correlation': self._correlation_similarity(
                        input_features[region_key],
                        ref_features[region_key]
                    )
                }
        
        self.logger.info(f"Best match for {region}: index {best_idx} with similarity {best_similarity:.4f}")
        
        return best_idx, best_similarity, best_scores
    
    def calculate_difference_map(self, input_tensor: torch.Tensor,
                               reference_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate pixel-wise difference between input and reference.
        "the process will take the input tensorized region or image and the reference 
        image and take the absolute value of the difference between the two"
        """
        # Ensure same shape
        if input_tensor.shape != reference_tensor.shape:
            reference_tensor = F.interpolate(
                reference_tensor.unsqueeze(0) if reference_tensor.dim() == 3 else reference_tensor,
                size=input_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            if reference_tensor.dim() == 4:
                reference_tensor = reference_tensor.squeeze(0)
        
        # Calculate absolute difference
        difference_map = torch.abs(input_tensor - reference_tensor)
        
        self.logger.log_tensor_operation("difference_map_calculated", 
                                       difference_map.shape,
                                       f"max_diff={difference_map.max().item():.4f}")
        
        return difference_map
    
    def compute_local_similarity_map(self, input_tensor: torch.Tensor,
                                   reference_tensor: torch.Tensor,
                                   window_size: int = 11) -> torch.Tensor:
        """
        Compute local similarity map using sliding window.
        "it will also look at the local anomaly heat map"
        """
        # Ensure same shape
        if input_tensor.shape != reference_tensor.shape:
            reference_tensor = F.interpolate(
                reference_tensor.unsqueeze(0) if reference_tensor.dim() == 3 else reference_tensor,
                size=input_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            if reference_tensor.dim() == 4:
                reference_tensor = reference_tensor.squeeze(0)
        
        # Convert to grayscale for local similarity
        if input_tensor.dim() == 3 and input_tensor.shape[0] == 3:
            gray_input = 0.299 * input_tensor[0] + 0.587 * input_tensor[1] + 0.114 * input_tensor[2]
            gray_ref = 0.299 * reference_tensor[0] + 0.587 * reference_tensor[1] + 0.114 * reference_tensor[2]
        else:
            gray_input = input_tensor.squeeze()
            gray_ref = reference_tensor.squeeze()
        
        # Pad tensors
        pad = window_size // 2
        padded_input = F.pad(gray_input, [pad, pad, pad, pad], mode='reflect')
        padded_ref = F.pad(gray_ref, [pad, pad, pad, pad], mode='reflect')
        
        h, w = gray_input.shape
        similarity_map = torch.zeros(h, w, device=input_tensor.device)
        
        # Compute local similarity for each position
        for i in range(h):
            for j in range(w):
                # Extract local windows
                window_input = padded_input[i:i+window_size, j:j+window_size]
                window_ref = padded_ref[i:i+window_size, j:j+window_size]
                
                # Calculate local correlation
                local_sim = self._correlation_similarity(
                    window_input.flatten(),
                    window_ref.flatten()
                )
                
                similarity_map[i, j] = local_sim
        
        return similarity_map
    
    def match_to_reference_or_no_match(self, input_tensor: torch.Tensor,
                                     reference_tensors: Dict[str, torch.Tensor],
                                     min_similarity: float = 0.7) -> Tuple[Optional[str], float]:
        """
        Try to match input to reference, return None if no good match.
        "this might not be possible if there is no image in the reference bank that 
        represent any of the regions, if that is the case it will not converge the 
        core cladding and ferrule instead it will use those regions to compare to 
        the most likely regions in the reference bank"
        """
        best_match_key = None
        best_similarity = -1
        
        for ref_key, ref_tensor in reference_tensors.items():
            similarity = self.calculate_similarity(input_tensor, ref_tensor, method='combined')
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_key = ref_key
        
        # Check if similarity meets threshold
        if best_similarity >= min_similarity:
            self.logger.info(f"Found reference match: {best_match_key} with similarity {best_similarity:.4f}")
            return best_match_key, best_similarity
        else:
            self.logger.warning(f"No reference match found. Best similarity: {best_similarity:.4f}")
            return None, best_similarity
    
    def calculate_feature_correlation(self, features1: torch.Tensor,
                                    features2: torch.Tensor) -> Dict[str, float]:
        """
        Calculate detailed correlation statistics between features.
        "each segment will be analyzed by multiple comparisons, statistics, and 
        correlational data from the multiple sources"
        """
        correlations = {}
        
        # Pearson correlation
        if features1.dim() == 1 and features2.dim() == 1:
            pearson_corr = torch.corrcoef(torch.stack([features1, features2]))[0, 1].item()
            correlations['pearson'] = pearson_corr
        
        # Spearman rank correlation (approximate)
        if features1.numel() == features2.numel():
            rank1 = torch.argsort(features1.flatten())
            rank2 = torch.argsort(features2.flatten())
            rank_corr = torch.corrcoef(torch.stack([rank1.float(), rank2.float()]))[0, 1].item()
            correlations['spearman'] = rank_corr
        
        # Distance metrics
        correlations['euclidean_distance'] = torch.norm(features1 - features2).item()
        correlations['manhattan_distance'] = torch.norm(features1 - features2, p=1).item()
        
        # Statistical measures
        correlations['mean_diff'] = (features1.mean() - features2.mean()).abs().item()
        correlations['std_diff'] = (features1.std() - features2.std()).abs().item()
        
        return correlations