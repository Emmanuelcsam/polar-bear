import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import time

from ..utils.logger import get_logger
from ..config.config import get_config


class ImageProcessor:
    """
    Handles image preprocessing and tensorization for fiber optic images.
    "the image will then be tensorized so it can better be computed by pytorch"
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.device = self.config.get_device()
        
        # Standard image size for neural network
        self.standard_size = (256, 256)  # Can be adjusted based on needs
        
        self.logger.info(f"Initialized ImageProcessor with device: {self.device}")
    
    def process_image(self, image_path: Path) -> torch.Tensor:
        """
        Load and tensorize a single image.
        "an image will be selected from a dataset folder"
        """
        start_time = time.time()
        
        # Log the processing start
        self.logger.log_image_processing(str(image_path), "loading", "started")
        
        # Load image using OpenCV
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess the image
        processed_image = self._preprocess_image(image)
        
        # Convert to tensor
        tensor = self._to_tensor(processed_image)
        
        # Move to appropriate device
        tensor = tensor.to(self.device)
        
        elapsed = time.time() - start_time
        self.logger.log_image_processing(str(image_path), "tensorization", 
                                       f"completed in {elapsed:.3f}s")
        self.logger.log_tensor_operation("image_tensorized", tensor.shape, 
                                       f"device={self.device}")
        
        return tensor
    
    def process_batch(self, image_paths: List[Path]) -> torch.Tensor:
        """
        Process multiple images for batch processing.
        "or multiple images for batch processing or realtime processing"
        """
        self.logger.info(f"Processing batch of {len(image_paths)} images")
        
        tensors = []
        for path in image_paths:
            try:
                tensor = self.process_image(path)
                tensors.append(tensor)
            except Exception as e:
                self.logger.error(f"Failed to process {path}: {e}")
        
        # Stack into batch
        if tensors:
            batch = torch.stack(tensors)
            self.logger.log_tensor_operation("batch_created", batch.shape, 
                                           f"num_images={len(tensors)}")
            return batch
        else:
            raise ValueError("No images successfully processed in batch")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to the image"""
        original_shape = image.shape
        
        # Resize to standard size if needed
        if image.shape[:2] != self.standard_size:
            image = cv2.resize(image, self.standard_size, interpolation=cv2.INTER_LINEAR)
            self.logger.debug(f"Resized image from {original_shape} to {image.shape}")
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply any additional preprocessing
        # Could add denoising, contrast enhancement, etc. here based on needs
        
        return image
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to PyTorch tensor"""
        # Convert HWC to CHW format
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        
        # Ensure float32 type
        tensor = tensor.float()
        
        return tensor
    
    def calculate_image_statistics(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Calculate various statistics for the image tensor.
        "each segment will be analyzed by multiple comparisons, statistics"
        """
        stats = {}
        
        # Basic statistics
        stats['mean'] = tensor.mean().item()
        stats['std'] = tensor.std().item()
        stats['min'] = tensor.min().item()
        stats['max'] = tensor.max().item()
        
        # Channel-wise statistics
        if tensor.dim() >= 3:
            for i in range(tensor.shape[0]):
                stats[f'channel_{i}_mean'] = tensor[i].mean().item()
                stats[f'channel_{i}_std'] = tensor[i].std().item()
        
        # Gradient statistics
        gradient_intensity = self._calculate_gradient_intensity(tensor)
        stats['gradient_intensity'] = gradient_intensity
        
        # Pixel position statistics
        avg_x, avg_y = self._calculate_center_of_mass(tensor)
        stats['center_x'] = avg_x
        stats['center_y'] = avg_y
        
        # Entropy (measure of information content)
        stats['entropy'] = self._calculate_entropy(tensor)
        
        self.logger.debug(f"Calculated statistics: {stats}")
        
        return stats
    
    def _calculate_gradient_intensity(self, tensor: torch.Tensor) -> float:
        """
        Calculate gradient intensity for the tensor.
        "the weights of the neural network will be dependent on the average intensity gradient"
        """
        # Convert to grayscale if needed
        if tensor.dim() == 3 and tensor.shape[0] == 3:
            gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
        else:
            gray = tensor.squeeze()
        
        # Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=tensor.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=tensor.device)
        
        # Apply convolution
        gray_unsqueezed = gray.unsqueeze(0).unsqueeze(0)
        
        grad_x = torch.nn.functional.conv2d(gray_unsqueezed, 
                                           sobel_x.unsqueeze(0).unsqueeze(0), 
                                           padding=1)
        grad_y = torch.nn.functional.conv2d(gray_unsqueezed, 
                                           sobel_y.unsqueeze(0).unsqueeze(0), 
                                           padding=1)
        
        # Calculate magnitude
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = gradient_magnitude.mean().item()
        
        return avg_gradient
    
    def _calculate_center_of_mass(self, tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate center of mass (average pixel position weighted by intensity).
        "another weight will be dependent on the average pixel position"
        """
        # Use mean across channels
        if tensor.dim() == 3:
            intensity = tensor.mean(dim=0)
        else:
            intensity = tensor
        
        h, w = intensity.shape
        
        # Create coordinate grids
        y_coords = torch.arange(h, dtype=torch.float32, device=tensor.device)
        x_coords = torch.arange(w, dtype=torch.float32, device=tensor.device)
        
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Calculate weighted average
        total_intensity = intensity.sum()
        if total_intensity > 0:
            center_y = (Y * intensity).sum() / total_intensity
            center_x = (X * intensity).sum() / total_intensity
        else:
            center_y = h / 2.0
            center_x = w / 2.0
        
        return center_x.item(), center_y.item()
    
    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """Calculate Shannon entropy of the image"""
        # Flatten and convert to probabilities
        flat = tensor.flatten()
        
        # Create histogram
        hist = torch.histc(flat, bins=256, min=0, max=1)
        hist = hist / hist.sum()  # Normalize to probabilities
        
        # Calculate entropy
        # Add small epsilon to avoid log(0)
        entropy = -(hist * torch.log(hist + 1e-10)).sum().item()
        
        return entropy
    
    def extract_color_features(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract color-based features from the image tensor.
        "any possible way an image can be analyzed and compared"
        """
        features = {}
        
        if tensor.dim() == 3 and tensor.shape[0] == 3:
            # Color histograms for each channel
            for i, color in enumerate(['red', 'green', 'blue']):
                hist = torch.histc(tensor[i].flatten(), bins=64, min=0, max=1)
                features[f'{color}_histogram'] = hist
            
            # HSV conversion for additional features
            # Convert to numpy for cv2 operation
            rgb_np = tensor.permute(1, 2, 0).cpu().numpy()
            hsv = cv2.cvtColor((rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv_tensor = torch.from_numpy(hsv).permute(2, 0, 1).float() / 255.0
            hsv_tensor = hsv_tensor.to(tensor.device)
            
            # Hue histogram (important for color classification)
            features['hue_histogram'] = torch.histc(hsv_tensor[0].flatten(), 
                                                   bins=36, min=0, max=1)
            
            # Saturation statistics
            features['saturation_mean'] = hsv_tensor[1].mean()
            features['saturation_std'] = hsv_tensor[1].std()
            
            # Value (brightness) statistics  
            features['value_mean'] = hsv_tensor[2].mean()
            features['value_std'] = hsv_tensor[2].std()
        
        return features
    
    def apply_preprocessing_filters(self, tensor: torch.Tensor, 
                                  denoise: bool = True,
                                  enhance_contrast: bool = True) -> torch.Tensor:
        """Apply optional preprocessing filters to improve image quality"""
        processed = tensor.clone()
        
        if denoise:
            # Apply Gaussian blur for denoising
            kernel_size = 3
            sigma = 0.5
            processed = self._gaussian_blur(processed, kernel_size, sigma)
            self.logger.debug("Applied Gaussian denoising")
        
        if enhance_contrast:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            processed = self._enhance_contrast(processed)
            self.logger.debug("Applied contrast enhancement")
        
        return processed
    
    def _gaussian_blur(self, tensor: torch.Tensor, kernel_size: int, 
                      sigma: float) -> torch.Tensor:
        """Apply Gaussian blur to tensor"""
        # Create Gaussian kernel
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., 
                         device=tensor.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # Apply convolution
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        blurred = []
        for i in range(tensor.shape[0]):
            channel = tensor[i].unsqueeze(0).unsqueeze(0)
            blurred_channel = torch.nn.functional.conv2d(channel, kernel, padding=kernel_size//2)
            blurred.append(blurred_channel.squeeze())
        
        return torch.stack(blurred)
    
    def _enhance_contrast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Enhance contrast using adaptive histogram equalization"""
        # Convert to numpy for CLAHE
        img_np = tensor.permute(1, 2, 0).cpu().numpy()
        img_uint8 = (img_np * 255).astype(np.uint8)
        
        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        enhanced_channels = []
        for i in range(img_uint8.shape[2]):
            enhanced = clahe.apply(img_uint8[:, :, i])
            enhanced_channels.append(enhanced)
        
        # Stack and convert back to tensor
        enhanced_img = np.stack(enhanced_channels, axis=2)
        enhanced_tensor = torch.from_numpy(enhanced_img).float() / 255.0
        enhanced_tensor = enhanced_tensor.permute(2, 0, 1).to(tensor.device)
        
        return enhanced_tensor