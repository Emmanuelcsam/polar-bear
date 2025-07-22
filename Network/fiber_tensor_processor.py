#!/usr/bin/env python3
"""
Tensor Processor module for Fiber Optics Neural Network
"the image will then be tensorized so it can better be computed by pytorch"
Handles conversion between images and tensors, and basic tensor operations
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional
from datetime import datetime

from fiber_config import get_config
from fiber_logger import get_logger

class TensorProcessor:
    """Handles all tensor operations for fiber optic images"""
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing TensorProcessor")
        print(f"[{datetime.now()}] Previous script: logger.py")
        
        self.config = get_config()
        self.logger = get_logger("TensorProcessor")
        
        self.logger.log_class_init("TensorProcessor")
        
        # Standard normalization values for fiber optic images
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        
        # Device for tensor operations
        self.device = self.config.get_device()
        
        self.logger.info(f"TensorProcessor initialized on device: {self.device}")
        print(f"[{datetime.now()}] TensorProcessor initialized successfully")
    
    def image_to_tensor(self, image: Union[np.ndarray, str, Path]) -> torch.Tensor:
        """
        Convert image to tensor
        "the image will then be tensorized so it can better be computed by pytorch"
        """
        self.logger.log_function_entry("image_to_tensor", image_type=type(image).__name__)
        
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            self.logger.debug(f"Loading image from: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                raise ValueError(f"Could not load image from {image_path}")
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")
        
        self.logger.debug(f"Original image shape: {image.shape}")
        
        # Resize to standard size if needed
        if image.shape[:2] != self.config.INPUT_SIZE:
            image = cv2.resize(image, self.config.INPUT_SIZE)
            self.logger.debug(f"Resized image to: {self.config.INPUT_SIZE}")
        
        # Convert to tensor
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        # Normalize
        tensor = self.normalize_tensor(tensor)
        
        self.logger.log_tensor_info("output_tensor", tensor)
        self.logger.log_function_exit("image_to_tensor")
        
        return tensor
    
    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to image"""
        self.logger.log_function_entry("tensor_to_image")
        
        # Move to CPU if needed
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Denormalize
        tensor = self.denormalize_tensor(tensor)
        
        # Convert to numpy
        image = tensor.permute(1, 2, 0).numpy()  # CHW to HWC
        image = (image * 255).astype(np.uint8)
        
        self.logger.log_function_exit("tensor_to_image")
        return image
    
    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor using standard values"""
        for c in range(3):
            tensor[c] = (tensor[c] - self.mean[c]) / self.std[c]
        return tensor
    
    def denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor"""
        tensor = tensor.clone()
        for c in range(3):
            tensor[c] = tensor[c] * self.std[c] + self.mean[c]
        return torch.clamp(tensor, 0, 1)
    
    def calculate_gradient_intensity(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate gradient intensity for the tensor
        "the weights of the neural network will be dependent on the average intensity gradient of the images"
        """
        self.logger.log_function_entry("calculate_gradient_intensity")
        
        # Ensure tensor is on correct device
        tensor = tensor.to(self.device)
        
        # Add batch dimension if needed
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # Calculate gradients using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Apply to each channel
        grad_x_list = []
        grad_y_list = []
        
        for c in range(3):
            channel = tensor[:, c:c+1, :, :]
            grad_x = torch.nn.functional.conv2d(channel, sobel_x, padding=1)
            grad_y = torch.nn.functional.conv2d(channel, sobel_y, padding=1)
            grad_x_list.append(grad_x)
            grad_y_list.append(grad_y)
        
        # Combine gradients
        grad_x = torch.cat(grad_x_list, dim=1)
        grad_y = torch.cat(grad_y_list, dim=1)
        
        # Calculate magnitude
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate average gradient
        avg_gradient = gradient_magnitude.mean()
        
        # Calculate gradient map (average across channels)
        gradient_map = gradient_magnitude.mean(dim=1, keepdim=True)
        
        results = {
            'gradient_x': grad_x,
            'gradient_y': grad_y,
            'gradient_magnitude': gradient_magnitude,
            'gradient_map': gradient_map,
            'average_gradient': avg_gradient
        }
        
        self.logger.info(f"Average gradient intensity: {avg_gradient.item():.4f}")
        self.logger.log_function_exit("calculate_gradient_intensity")
        
        return results
    
    def calculate_pixel_positions(self, tensor_shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        """
        Calculate pixel position maps
        "another weight will be dependent on the average pixel position"
        """
        self.logger.log_function_entry("calculate_pixel_positions", shape=tensor_shape)
        
        if len(tensor_shape) == 4:
            b, c, h, w = tensor_shape
        else:
            c, h, w = tensor_shape
            b = 1
        
        # Create position grids
        y_pos = torch.linspace(-1, 1, h, device=self.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_pos = torch.linspace(-1, 1, w, device=self.device).view(1, 1, 1, w).expand(b, 1, h, w)
        
        # Calculate radial position (distance from center)
        radial_pos = torch.sqrt(x_pos**2 + y_pos**2)
        
        # Calculate angular position
        angular_pos = torch.atan2(y_pos, x_pos)
        
        # Average positions
        avg_x_pos = x_pos.mean()
        avg_y_pos = y_pos.mean()
        avg_radial_pos = radial_pos.mean()
        
        results = {
            'x_positions': x_pos,
            'y_positions': y_pos,
            'radial_positions': radial_pos,
            'angular_positions': angular_pos,
            'average_x': avg_x_pos,
            'average_y': avg_y_pos,
            'average_radial': avg_radial_pos
        }
        
        self.logger.debug(f"Average radial position: {avg_radial_pos.item():.4f}")
        self.logger.log_function_exit("calculate_pixel_positions")
        
        return results
    
    def load_tensor_file(self, tensor_path: Path) -> Dict[str, torch.Tensor]:
        """
        Load a .pt tensor file
        "I have a folder with a large database of .pt files"
        """
        self.logger.log_function_entry("load_tensor_file", path=str(tensor_path))
        
        if not tensor_path.exists():
            self.logger.error(f"Tensor file not found: {tensor_path}")
            raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
        
        try:
            # Load tensor data
            data = torch.load(tensor_path, map_location=self.device, weights_only=False)
            
            # Handle different formats
            if isinstance(data, torch.Tensor):
                tensor_data = {'tensor': data}
            elif isinstance(data, dict):
                tensor_data = data
            else:
                self.logger.error(f"Unknown tensor format in {tensor_path}")
                raise ValueError(f"Unknown tensor format")
            
            # Log tensor information
            if 'tensor' in tensor_data:
                self.logger.log_tensor_info("loaded_tensor", tensor_data['tensor'])
            
            self.logger.log_function_exit("load_tensor_file", "Success")
            return tensor_data
            
        except Exception as e:
            self.logger.log_error(f"Failed to load tensor file: {tensor_path}", e)
            raise
    
    def save_tensor_file(self, tensor: torch.Tensor, save_path: Path, metadata: Optional[Dict] = None):
        """Save tensor to .pt file with metadata"""
        self.logger.log_function_entry("save_tensor_file", path=str(save_path))
        
        # Prepare data to save
        save_data = {
            'tensor': tensor.cpu(),
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            save_data['metadata'] = metadata
        
        # Create directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        torch.save(save_data, save_path)
        
        self.logger.info(f"Saved tensor to: {save_path}")
        self.logger.log_function_exit("save_tensor_file")
    
    def create_multi_scale_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Create multi-scale versions of tensor
        "Features from different scales are correlated to ensure consistency"
        """
        self.logger.log_function_entry("create_multi_scale_tensors")
        
        multi_scale = []
        
        for scale in self.config.SCALES:
            if scale == 1.0:
                scaled = tensor
            else:
                # Calculate new size
                if tensor.dim() == 4:
                    b, c, h, w = tensor.shape
                else:
                    c, h, w = tensor.shape
                    b = 1
                
                new_h = int(h * scale)
                new_w = int(w * scale)
                
                # Resize
                scaled = torch.nn.functional.interpolate(
                    tensor.unsqueeze(0) if tensor.dim() == 3 else tensor,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )
                
                if tensor.dim() == 3:
                    scaled = scaled.squeeze(0)
            
            multi_scale.append(scaled)
            self.logger.debug(f"Created scale {scale}: shape={scaled.shape}")
        
        self.logger.log_function_exit("create_multi_scale_tensors", f"{len(multi_scale)} scales")
        return multi_scale
    
    def extract_region_mask(self, tensor: torch.Tensor, region_type: str) -> torch.Tensor:
        """
        Extract region mask for core, cladding, or ferrule
        "the network converges into three specific regions the core, cladding and ferrule"
        """
        self.logger.log_function_entry("extract_region_mask", region=region_type)
        
        # This is a placeholder - actual implementation would use trained model
        # For now, create simple circular masks based on region type
        
        if tensor.dim() == 4:
            b, c, h, w = tensor.shape
        else:
            c, h, w = tensor.shape
            b = 1
        
        # Create coordinate grids
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        center_y, center_x = h // 2, w // 2
        
        # Calculate distance from center
        dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2).float()
        
        # Define region boundaries (these would be learned in actual implementation)
        if region_type == 'core':
            mask = (dist < h * 0.1).float()
        elif region_type == 'cladding':
            mask = ((dist >= h * 0.1) & (dist < h * 0.3)).float()
        elif region_type == 'ferrule':
            mask = (dist >= h * 0.3).float()
        else:
            raise ValueError(f"Unknown region type: {region_type}")
        
        # Move to correct device
        mask = mask.to(self.device)
        
        self.logger.debug(f"Created {region_type} mask with coverage: {mask.mean().item():.2%}")
        self.logger.log_function_exit("extract_region_mask")
        
        return mask
    
    def get_tensor_statistics(self, tensor: torch.Tensor) -> Dict[str, Union[float, List[int]]]:
        """
        Calculate statistics for a tensor
        """
        return {
            'mean': float(tensor.mean().item()),
            'std': float(tensor.std().item()),
            'min': float(tensor.min().item()),
            'max': float(tensor.max().item()),
            'shape': list(tensor.shape)
        }

# Test the tensor processor
if __name__ == "__main__":
    processor = TensorProcessor()
    logger = get_logger("TensorProcessorTest")
    
    logger.log_process_start("Tensor Processor Test")
    
    # Create test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Convert to tensor
    tensor = processor.image_to_tensor(test_image)
    logger.info(f"Created tensor with shape: {tensor.shape}")
    
    # Calculate gradients
    gradients = processor.calculate_gradient_intensity(tensor)
    logger.info(f"Gradient results: avg={gradients['average_gradient'].item():.4f}")
    
    # Calculate positions
    positions = processor.calculate_pixel_positions(tensor.shape)
    logger.info(f"Position results: avg_radial={positions['average_radial'].item():.4f}")
    
    # Create multi-scale
    multi_scale = processor.create_multi_scale_tensors(tensor)
    logger.info(f"Created {len(multi_scale)} scales")
    
    # Extract region masks
    for region in ['core', 'cladding', 'ferrule']:
        mask = processor.extract_region_mask(tensor, region)
        logger.info(f"{region} mask coverage: {mask.mean().item():.2%}")
    
    logger.log_process_end("Tensor Processor Test")
    logger.log_script_transition("tensor_processor.py", "feature_extractor.py")
    
    print(f"[{datetime.now()}] Tensor processor test completed")
    print(f"[{datetime.now()}] Next script: feature_extractor.py")
