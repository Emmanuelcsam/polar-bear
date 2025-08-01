#!/usr/bin/env python3
"""
Reference Comparator module for Fiber Optics Neural Network
"try to see which of the reference images in the reference folder of the database 
that the regions most specifically represent"
"the program must achieve over .7"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

from core.config_loader import get_config
from core.logger import get_logger
from data.tensor_processor import TensorProcessor

class ReferenceDatabase:
    """
    Manages the reference image database
    "I have a folder with a large database of .pt files of different images used as reference"
    """
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing ReferenceDatabase")
        print(f"[{datetime.now()}] Previous script: feature_extractor.py")
        
        self.config = get_config()
        self.logger = get_logger("ReferenceDatabase")
        self.tensor_processor = TensorProcessor()
        
        self.logger.log_class_init("ReferenceDatabase")
        
        # Storage for reference data
        self.reference_tensors = {
            'core': {},
            'cladding': {},
            'ferrule': {},
            'full': {}  # Full images for overall comparison
        }
        
        self.reference_features = {
            'core': {},
            'cladding': {},
            'ferrule': {},
            'full': {}
        }
        
        self.reference_metadata = {}
        
        # Load database index if exists
        self.index_file = self.config.REFERENCE_PATH / "reference_index.json"
        self.load_index()
        
        self.logger.info("ReferenceDatabase initialized")
        print(f"[{datetime.now()}] ReferenceDatabase initialized successfully")
    
    def load_index(self):
        """Load reference database index"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.reference_metadata = json.load(f)
            self.logger.info(f"Loaded reference index with {len(self.reference_metadata)} entries")
    
    def save_index(self):
        """Save reference database index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.reference_metadata, f, indent=2)
        self.logger.debug("Saved reference index")
    
    def scan_reference_folders(self):
        """
        Scan reference folders and build database
        "the files are separated by folder name"
        """
        print(f"[{datetime.now()}] ReferenceDatabase.scan_reference_folders: Starting reference database scan")
        self.logger.log_process_start("Reference Database Scan")
        
        total_references = 0
        
        # Scan each region category
        for region, folders in self.config.REGION_CATEGORIES.items():
            if region == 'defects':
                continue  # Skip defect folders for reference
            
            self.logger.info(f"Scanning {region} folders: {folders}")
            
            for folder_name in folders:
                folder_path = self.config.TENSORIZED_DATA_PATH / folder_name
                
                if not folder_path.exists():
                    self.logger.warning(f"Folder not found: {folder_path}")
                    continue
                
                # Find all .pt files
                pt_files = list(folder_path.glob("*.pt"))
                self.logger.info(f"Found {len(pt_files)} files in {folder_name}")
                
                for pt_file in pt_files[:100]:  # Limit for memory
                    try:
                        # Load tensor
                        tensor_data = self.tensor_processor.load_tensor_file(pt_file)
                        tensor = tensor_data.get('tensor')
                        
                        if tensor is not None:
                            ref_id = f"{region}_{folder_name}_{pt_file.stem}"
                            
                            # Store tensor reference (path only to save memory)
                            self.reference_tensors[region][ref_id] = pt_file
                            
                            # Store metadata
                            self.reference_metadata[ref_id] = {
                                'region': region,
                                'folder': folder_name,
                                'file': pt_file.name,
                                'shape': list(tensor.shape) if hasattr(tensor, 'shape') else None,
                                'path': str(pt_file)
                            }
                            
                            total_references += 1
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to load {pt_file}: {e}")
        
        self.save_index()
        self.logger.info(f"Total references loaded: {total_references}")
        self.logger.log_process_end("Reference Database Scan")
    
    def get_reference_tensor(self, ref_id: str) -> Optional[torch.Tensor]:
        """Get a reference tensor by ID"""
        # Find which region it belongs to
        metadata = self.reference_metadata.get(ref_id)
        if not metadata:
            return None
        
        region = metadata['region']
        
        # Check if we have the path
        if ref_id in self.reference_tensors.get(region, {}):
            ref_path = self.reference_tensors[region][ref_id]
            
            # Load tensor
            if isinstance(ref_path, Path):
                tensor_data = self.tensor_processor.load_tensor_file(ref_path)
                return tensor_data.get('tensor')
            else:
                return ref_path  # Already a tensor
        
        return None
    
    def get_references_for_region(self, region: str, limit: int = 50) -> Dict[str, torch.Tensor]:
        """Get reference tensors for a specific region"""
        references = {}
        
        region_refs = self.reference_tensors.get(region, {})
        
        for i, (ref_id, ref_path) in enumerate(region_refs.items()):
            if i >= limit:
                break
            
            tensor = self.get_reference_tensor(ref_id)
            if tensor is not None:
                references[ref_id] = tensor
        
        return references

class SimilarityCalculator(nn.Module):
    """
    Calculates similarity between input and reference images
    "S is the similarity coefficient the percentage of how similar the input image is from the reference"
    """
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Feature encoder for similarity calculation
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim // 2)
        )
        
        # Learnable similarity metric
        self.similarity_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_tensor: torch.Tensor, reference_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate similarity between input and reference"""
        # Encode both tensors
        input_features = self.encoder(input_tensor)
        reference_features = self.encoder(reference_tensor)
        
        # Calculate different similarity metrics
        
        # 1. Cosine similarity
        cosine_sim = F.cosine_similarity(input_features, reference_features, dim=1)
        
        # 2. L2 distance (converted to similarity)
        l2_distance = torch.norm(input_features - reference_features, p=2, dim=1)
        l2_similarity = torch.exp(-l2_distance / self.feature_dim)
        
        # 3. Learned similarity
        combined_features = torch.cat([input_features, reference_features], dim=1)
        learned_similarity = self.similarity_net(combined_features).squeeze()
        
        # 4. Structural similarity (simplified)
        ssim = self._calculate_ssim(input_tensor, reference_tensor)
        
        # Combine similarities
        # "I=Ax1+Bx2+Cx3... =S(R)"
        combined_similarity = (
            0.3 * cosine_sim + 
            0.2 * l2_similarity + 
            0.3 * learned_similarity + 
            0.2 * ssim
        )
        
        return {
            'similarity': combined_similarity,
            'cosine_similarity': cosine_sim,
            'l2_similarity': l2_similarity,
            'learned_similarity': learned_similarity,
            'structural_similarity': ssim,
            'input_features': input_features,
            'reference_features': reference_features
        }
    
    def _calculate_ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate structural similarity index"""
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Use grayscale for SSIM
        x_gray = x.mean(dim=1, keepdim=True)
        y_gray = y.mean(dim=1, keepdim=True)
        
        # Local means
        mu_x = F.avg_pool2d(x_gray, 3, 1, 1)
        mu_y = F.avg_pool2d(y_gray, 3, 1, 1)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Local variances
        sigma_x_sq = F.avg_pool2d(x_gray ** 2, 3, 1, 1) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y_gray ** 2, 3, 1, 1) - mu_y_sq
        sigma_xy = F.avg_pool2d(x_gray * y_gray, 3, 1, 1) - mu_xy
        
        # SSIM
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return ssim_map.mean(dim=[1, 2, 3])

class ReferenceComparator:
    """
    Main reference comparison system
    "which of the reference images in the reference folder of the database that the regions most specifically represent"
    """
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing ReferenceComparator")
        
        self.config = get_config()
        self.logger = get_logger("ReferenceComparator")
        
        self.logger.log_class_init("ReferenceComparator")
        
        # Initialize components
        self.database = ReferenceDatabase()
        self.similarity_calculator = SimilarityCalculator()
        
        # Move to device
        self.device = self.config.get_device()
        self.similarity_calculator = self.similarity_calculator.to(self.device)
        
        # Cache for reference features
        self.reference_feature_cache = {}
        
        self.logger.info("ReferenceComparator initialized")
        print(f"[{datetime.now()}] ReferenceComparator initialized successfully")
    
    def find_best_references(self, input_tensor: torch.Tensor, 
                           region_masks: Dict[str, torch.Tensor],
                           top_k: int = 5) -> Dict[str, Dict]:
        """
        Find best matching references for input
        "if the entire program does not achieve over .7 whenever refering to the reference... it must achieve over .7"
        """
        self.logger.log_process_start("Reference Comparison")
        
        results = {}
        
        # Compare full image
        full_results = self._compare_to_region_references(input_tensor, 'full', top_k)
        results['full'] = full_results
        
        # Compare each region
        for region, mask in region_masks.items():
            # Apply mask to input
            masked_input = input_tensor * mask.unsqueeze(1)
            
            # Compare to region-specific references
            region_results = self._compare_to_region_references(masked_input, region, top_k)
            results[region] = region_results
            
            # Check if similarity meets threshold
            best_similarity = region_results['best_similarity']
            self.logger.log_similarity_check(
                best_similarity,
                self.config.SIMILARITY_THRESHOLD,
                region_results['best_reference']
            )
            
            if best_similarity < self.config.SIMILARITY_THRESHOLD:
                self.logger.warning(f"Low similarity for {region}: {best_similarity:.4f}")
                # "if it doesn't it will detail why it didn't and try to locate the anomalies"
                results[region]['low_similarity_reason'] = self._analyze_low_similarity(
                    masked_input, region_results['best_reference_tensor']
                )
        
        self.logger.log_process_end("Reference Comparison")
        
        return results
    
    def _compare_to_region_references(self, input_tensor: torch.Tensor, 
                                    region: str, top_k: int) -> Dict:
        """Compare input to references for a specific region"""
        # Get references for region
        references = self.database.get_references_for_region(region, limit=50)
        
        if not references:
            self.logger.warning(f"No references found for region: {region}")
            return {
                'best_similarity': 0.0,
                'best_reference': None,
                'top_k_references': []
            }
        
        similarities = []
        
        # Compare to each reference
        for ref_id, ref_tensor in references.items():
            # Ensure same device and shape
            ref_tensor = ref_tensor.to(self.device)
            
            # Resize if needed
            if ref_tensor.shape[-2:] != input_tensor.shape[-2:]:
                ref_tensor = F.interpolate(
                    ref_tensor.unsqueeze(0) if ref_tensor.dim() == 3 else ref_tensor,
                    size=input_tensor.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                if input_tensor.dim() == 3:
                    ref_tensor = ref_tensor.squeeze(0)
            
            # Calculate similarity
            sim_results = self.similarity_calculator(
                input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor,
                ref_tensor.unsqueeze(0) if ref_tensor.dim() == 3 else ref_tensor
            )
            
            similarity = sim_results['similarity'].item()
            
            similarities.append({
                'reference_id': ref_id,
                'similarity': similarity,
                'tensor': ref_tensor,
                'details': {
                    'cosine': sim_results['cosine_similarity'].item(),
                    'l2': sim_results['l2_similarity'].item(),
                    'learned': sim_results['learned_similarity'].item(),
                    'ssim': sim_results['structural_similarity'].item()
                }
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top k
        top_k_refs = similarities[:top_k]
        
        return {
            'best_similarity': similarities[0]['similarity'] if similarities else 0.0,
            'best_reference': similarities[0]['reference_id'] if similarities else None,
            'best_reference_tensor': similarities[0]['tensor'] if similarities else None,
            'top_k_references': top_k_refs,
            'all_similarities': similarities
        }
    
    def _analyze_low_similarity(self, input_tensor: torch.Tensor, 
                              reference_tensor: torch.Tensor) -> Dict:
        """
        Analyze why similarity is low
        "it will detail why it didn't and try to locate the anomalies"
        """
        if reference_tensor is None:
            return {'reason': 'No reference found'}
        
        # Calculate pixel-wise difference
        diff = torch.abs(input_tensor - reference_tensor)
        
        # Find high difference regions
        threshold = diff.mean() + 2 * diff.std()
        anomaly_mask = diff.mean(dim=0 if input_tensor.dim() == 3 else 1) > threshold
        
        # Get anomaly locations
        anomaly_locations = torch.where(anomaly_mask)
        
        return {
            'reason': 'High pixel differences detected',
            'mean_difference': diff.mean().item(),
            'max_difference': diff.max().item(),
            'anomaly_pixel_count': anomaly_mask.sum().item(),
            'anomaly_locations': list(zip(
                anomaly_locations[0].tolist()[:10],
                anomaly_locations[1].tolist()[:10]
            ))  # First 10 locations
        }
    
    def calculate_reconstruction_similarity(self, original: torch.Tensor, 
                                          reconstructed: torch.Tensor) -> Dict:
        """
        Calculate similarity between original and reconstructed image
        "Im subtracting the resulting classification with the original input to find anomalies"
        """
        # Direct similarity calculation
        sim_results = self.similarity_calculator(original, reconstructed)
        
        # Pixel-wise difference
        pixel_diff = torch.abs(original - reconstructed)
        
        # Local anomaly heatmap
        # "it will also look at the local anomaly heat map"
        kernel_size = 5
        local_mean_diff = F.avg_pool2d(
            pixel_diff.mean(dim=1, keepdim=True),
            kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        
        # Normalize to [0, 1]
        anomaly_heatmap = (local_mean_diff - local_mean_diff.min()) / \
                         (local_mean_diff.max() - local_mean_diff.min() + 1e-8)
        
        return {
            'similarity': sim_results['similarity'],
            'pixel_difference': pixel_diff,
            'anomaly_heatmap': anomaly_heatmap.squeeze(1),
            'similarity_details': sim_results
        }

# Test the reference comparator
if __name__ == "__main__":
    comparator = ReferenceComparator()
    logger = get_logger("ReferenceComparatorTest")
    
    logger.log_process_start("Reference Comparator Test")
    
    # Scan reference database
    comparator.database.scan_reference_folders()
    
    # Create test tensors
    test_input = torch.randn(3, 256, 256).to(comparator.device)
    test_masks = {
        'core': torch.zeros(256, 256).to(comparator.device),
        'cladding': torch.zeros(256, 256).to(comparator.device),
        'ferrule': torch.zeros(256, 256).to(comparator.device)
    }
    
    # Create simple circular masks
    center = 128
    for i in range(256):
        for j in range(256):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < 50:
                test_masks['core'][i, j] = 1
            elif dist < 100:
                test_masks['cladding'][i, j] = 1
            else:
                test_masks['ferrule'][i, j] = 1
    
    # Find best references
    results = comparator.find_best_references(test_input, test_masks, top_k=3)
    
    # Log results
    for region, region_results in results.items():
        logger.info(f"\n{region.upper()} region results:")
        logger.info(f"  Best similarity: {region_results['best_similarity']:.4f}")
        logger.info(f"  Best reference: {region_results['best_reference']}")
        
        if 'low_similarity_reason' in region_results:
            logger.warning(f"  Low similarity reason: {region_results['low_similarity_reason']['reason']}")
    
    logger.log_process_end("Reference Comparator Test")
    logger.log_script_transition("reference_comparator.py", "anomaly_detector.py")
    
    print(f"[{datetime.now()}] Reference comparator test completed")
    print(f"[{datetime.now()}] Next script: anomaly_detector.py")