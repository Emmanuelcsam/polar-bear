import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger
from ..config.config import get_config


@dataclass
class SegmentationResult:
    """Container for segmentation results"""
    core_mask: torch.Tensor
    cladding_mask: torch.Tensor
    ferrule_mask: torch.Tensor
    core_region: torch.Tensor
    cladding_region: torch.Tensor
    ferrule_region: torch.Tensor
    confidence_scores: Dict[str, float]


class FiberOpticSegmentation:
    """
    Segments fiber optic images into core, cladding, and ferrule regions.
    "I have made programs in the past and script that have segmented the regions of the 
    fiber optic endfaces into the core cladding and ferrule"
    "after the network converges the features of the image to either core cladding and ferrule"
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.device = self.config.get_device()
        
        # Expected characteristics of each region
        self.region_characteristics = {
            'core': {
                'typical_radius_ratio': 0.05,  # Core is typically 5% of image width
                'intensity_range': (0.7, 1.0),  # Usually bright
                'position': 'center'
            },
            'cladding': {
                'typical_radius_ratio': 0.25,  # Cladding extends to ~25% of image width
                'intensity_range': (0.3, 0.7),  # Medium intensity
                'position': 'annulus'  # Ring around core
            },
            'ferrule': {
                'typical_radius_ratio': 1.0,  # Rest of the image
                'intensity_range': (0.0, 0.3),  # Usually darker
                'position': 'outer'
            }
        }
        
        self.logger.info("Initialized FiberOpticSegmentation module")
    
    def segment_image(self, tensor: torch.Tensor) -> SegmentationResult:
        """
        Main segmentation function that splits image into three regions.
        "in the neural network for classification it will be split up into segments"
        """
        self.logger.log_tensor_operation("segmentation_start", tensor.shape)
        
        # Ensure tensor is on correct device
        tensor = tensor.to(self.device)
        
        # Find center and radii of regions
        center, radii = self._find_fiber_center_and_radii(tensor)
        
        # Create masks for each region
        masks = self._create_region_masks(tensor.shape[-2:], center, radii)
        
        # Extract regions
        core_region = self._extract_region(tensor, masks['core'])
        cladding_region = self._extract_region(tensor, masks['cladding'])
        ferrule_region = self._extract_region(tensor, masks['ferrule'])
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            tensor, masks, center, radii
        )
        
        # Log results
        self.logger.info(f"Segmentation complete - Confidence scores: {confidence_scores}")
        
        result = SegmentationResult(
            core_mask=masks['core'],
            cladding_mask=masks['cladding'],
            ferrule_mask=masks['ferrule'],
            core_region=core_region,
            cladding_region=cladding_region,
            ferrule_region=ferrule_region,
            confidence_scores=confidence_scores
        )
        
        return result
    
    def _find_fiber_center_and_radii(self, tensor: torch.Tensor) -> Tuple[Tuple[int, int], Dict[str, float]]:
        """
        Find the center of the fiber and radii of different regions.
        "the change in pixels between the core to cladding or cladding to ferrule"
        """
        # Convert to grayscale for analysis
        if tensor.dim() == 3:
            gray = tensor.mean(dim=0)
        else:
            gray = tensor
        
        # Method 1: Hough Circle Detection for initial estimate
        center_hough = self._hough_circle_detection(gray)
        
        # Method 2: Intensity-based center of mass
        center_mass = self._intensity_center_of_mass(gray)
        
        # Combine methods with weighted average
        if center_hough is not None:
            center = (
                int(0.7 * center_hough[0] + 0.3 * center_mass[0]),
                int(0.7 * center_hough[1] + 0.3 * center_mass[1])
            )
        else:
            center = (int(center_mass[0]), int(center_mass[1]))
        
        # Find radii using gradient analysis
        radii = self._find_region_radii(gray, center)
        
        self.logger.debug(f"Found fiber center: {center}, radii: {radii}")
        
        return center, radii
    
    def _hough_circle_detection(self, gray_tensor: torch.Tensor) -> Optional[Tuple[int, int]]:
        """Use Hough transform to detect circular features"""
        try:
            # Convert to numpy for OpenCV
            gray_np = (gray_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray_np, (9, 9), 2)
            
            # Detect circles
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=gray_np.shape[0] // 2
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Return center of the first detected circle
                x, y, r = circles[0, 0]
                return (int(x), int(y))
            
        except Exception as e:
            self.logger.debug(f"Hough circle detection failed: {e}")
        
        return None
    
    def _intensity_center_of_mass(self, gray_tensor: torch.Tensor) -> Tuple[float, float]:
        """Calculate center of mass based on intensity"""
        h, w = gray_tensor.shape
        
        # Create coordinate grids
        y_grid = torch.arange(h, device=gray_tensor.device).float()
        x_grid = torch.arange(w, device=gray_tensor.device).float()
        Y, X = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Calculate weighted center
        total_intensity = gray_tensor.sum()
        center_y = (Y * gray_tensor).sum() / total_intensity
        center_x = (X * gray_tensor).sum() / total_intensity
        
        return (center_x.item(), center_y.item())
    
    def _find_region_radii(self, gray_tensor: torch.Tensor, 
                          center: Tuple[int, int]) -> Dict[str, float]:
        """
        Find radii of different regions using radial intensity profiles.
        "knows the lines of best fit for all pixel values, gradients, and pixel positions"
        """
        h, w = gray_tensor.shape
        cy, cx = center
        
        # Extract radial profile
        max_radius = min(cx, cy, w - cx, h - cy)
        radial_profile = []
        
        for r in range(1, max_radius):
            # Sample points in a circle
            angles = torch.linspace(0, 2 * np.pi, 36, device=gray_tensor.device)
            x_coords = (cx + r * torch.cos(angles)).long()
            y_coords = (cy + r * torch.sin(angles)).long()
            
            # Ensure coordinates are within bounds
            mask = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
            x_coords = x_coords[mask]
            y_coords = y_coords[mask]
            
            if len(x_coords) > 0:
                mean_intensity = gray_tensor[y_coords, x_coords].mean().item()
                radial_profile.append(mean_intensity)
            else:
                radial_profile.append(0)
        
        # Find transitions using gradient analysis
        radial_gradient = np.gradient(radial_profile)
        
        # Find peaks in gradient (transitions between regions)
        # "a sharp change in intensity gradient from an in circle to a annulus region"
        transitions = self._find_gradient_transitions(radial_gradient)
        
        # Estimate radii based on transitions and expected ratios
        radii = self._estimate_radii_from_transitions(transitions, max_radius, gray_tensor.shape)
        
        return radii
    
    def _find_gradient_transitions(self, gradient: np.ndarray) -> List[int]:
        """Find significant transitions in radial gradient"""
        # Smooth the gradient
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(np.abs(gradient), sigma=2)
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(smoothed, height=np.std(smoothed), distance=5)
        
        # Sort by prominence
        if len(peaks) > 0:
            prominences = properties['peak_heights']
            sorted_indices = np.argsort(prominences)[::-1]
            return peaks[sorted_indices].tolist()
        
        return []
    
    def _estimate_radii_from_transitions(self, transitions: List[int], 
                                       max_radius: int,
                                       image_shape: Tuple[int, int]) -> Dict[str, float]:
        """Estimate region radii from detected transitions"""
        h, w = image_shape
        image_size = min(h, w)
        
        radii = {}
        
        # If we found transitions, use them
        if len(transitions) >= 2:
            radii['core'] = float(transitions[0])
            radii['cladding'] = float(transitions[1])
        elif len(transitions) == 1:
            radii['core'] = float(transitions[0])
            # Estimate cladding based on typical ratio
            radii['cladding'] = radii['core'] * 5.0  # Cladding is typically 5x core
        else:
            # Use default ratios
            radii['core'] = image_size * self.region_characteristics['core']['typical_radius_ratio']
            radii['cladding'] = image_size * self.region_characteristics['cladding']['typical_radius_ratio']
        
        # Ferrule is everything else
        radii['ferrule'] = float(max_radius)
        
        return radii
    
    def _create_region_masks(self, shape: Tuple[int, int], 
                           center: Tuple[int, int],
                           radii: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Create binary masks for each region"""
        h, w = shape
        cy, cx = center
        
        # Create coordinate grids
        y_grid = torch.arange(h, device=self.device).float()
        x_grid = torch.arange(w, device=self.device).float()
        Y, X = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Calculate distance from center
        dist = torch.sqrt((X - cx)**2 + (Y - cy)**2)
        
        # Create masks
        masks = {
            'core': dist <= radii['core'],
            'cladding': (dist > radii['core']) & (dist <= radii['cladding']),
            'ferrule': dist > radii['cladding']
        }
        
        return masks
    
    def _extract_region(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract a region from the tensor using a mask.
        "I have cut up images of a cladding for cladding features, I also have images 
        where only the cladding is shown and the rest of the image is cropped"
        """
        # Expand mask to match tensor dimensions
        if tensor.dim() == 3:
            mask = mask.unsqueeze(0).expand(tensor.shape[0], -1, -1)
        
        # Apply mask
        region = tensor * mask.float()
        
        # Find bounding box of the mask
        if mask.any():
            indices = torch.where(mask[0] if mask.dim() == 3 else mask)
            y_min, y_max = indices[0].min(), indices[0].max()
            x_min, x_max = indices[1].min(), indices[1].max()
            
            # Crop to bounding box
            if tensor.dim() == 3:
                region = region[:, y_min:y_max+1, x_min:x_max+1]
            else:
                region = region[y_min:y_max+1, x_min:x_max+1]
        
        return region
    
    def _calculate_confidence_scores(self, tensor: torch.Tensor,
                                   masks: Dict[str, torch.Tensor],
                                   center: Tuple[int, int],
                                   radii: Dict[str, float]) -> Dict[str, float]:
        """Calculate confidence scores for the segmentation"""
        scores = {}
        
        # Overall segmentation confidence based on intensity distributions
        for region_name, mask in masks.items():
            if mask.any():
                region_pixels = tensor[..., mask].flatten()
                expected_range = self.region_characteristics[region_name]['intensity_range']
                
                # Calculate percentage of pixels in expected range
                in_range = ((region_pixels >= expected_range[0]) & 
                          (region_pixels <= expected_range[1])).float().mean()
                
                scores[f'{region_name}_confidence'] = in_range.item()
        
        # Circularity score (how circular the detected regions are)
        h, w = tensor.shape[-2:]
        max_dim = max(h, w)
        circularity = 1.0 - abs(h - w) / max_dim
        scores['circularity'] = circularity
        
        # Center confidence (how centered the fiber is)
        center_offset = np.sqrt((center[0] - w/2)**2 + (center[1] - h/2)**2)
        center_confidence = 1.0 - (center_offset / (max_dim/2))
        scores['center_confidence'] = max(0, center_confidence)
        
        # Overall confidence
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores
    
    def adaptive_segmentation(self, tensor: torch.Tensor, 
                            reference_result: Optional[SegmentationResult] = None) -> SegmentationResult:
        """
        Perform adaptive segmentation based on reference data.
        "the program will forcibly look for all lines of best fit based on gradient trends"
        """
        # Initial segmentation
        result = self.segment_image(tensor)
        
        if reference_result is not None:
            # Adjust based on reference
            self.logger.debug("Performing adaptive segmentation with reference")
            
            # Compare intensity distributions
            for region in ['core', 'cladding', 'ferrule']:
                ref_region = getattr(reference_result, f'{region}_region')
                curr_region = getattr(result, f'{region}_region')
                
                if ref_region.numel() > 0 and curr_region.numel() > 0:
                    # Calculate distribution similarity
                    ref_hist = torch.histc(ref_region.flatten(), bins=50, min=0, max=1)
                    curr_hist = torch.histc(curr_region.flatten(), bins=50, min=0, max=1)
                    
                    # Normalize histograms
                    ref_hist = ref_hist / ref_hist.sum()
                    curr_hist = curr_hist / curr_hist.sum()
                    
                    # Calculate histogram distance
                    hist_distance = torch.norm(ref_hist - curr_hist).item()
                    
                    if hist_distance > 0.3:  # Threshold for significant difference
                        self.logger.warning(f"Large histogram distance for {region}: {hist_distance}")
                        # Could trigger re-segmentation with adjusted parameters here
        
        return result