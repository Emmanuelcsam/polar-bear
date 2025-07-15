#!/usr/bin/env python3
"""
Script Wrappers for PyTorch Production Module
Provides wrapper functions for all scripts to work with connectors
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

# Import the PixelGenerator model
class PixelGenerator(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(size, size, 3, 256))

    def get_probs(self):
        return torch.softmax(self.logits, dim=3)


class ScriptWrappers:
    """Wrapper functions for all scripts with connector integration"""
    
    def __init__(self):
        self.logger = logging.getLogger('ScriptWrappers')
        self.shared_state = {}
        
    def initialize_generator(self, img_size: int = 128, model_filename: str = 'generator_model.pth', 
                           force: bool = False) -> Dict[str, Any]:
        """Initialize PyTorch generator model (preprocess.py functionality)"""
        try:
            if os.path.exists(model_filename) and not force:
                return {
                    'status': 'exists',
                    'message': f"Model '{model_filename}' already exists",
                    'path': model_filename
                }
            
            self.logger.info("Initializing a new PyTorch generator model...")
            model = PixelGenerator(img_size)
            torch.save(model.state_dict(), model_filename)
            
            # Store in shared state
            self.shared_state['model'] = model
            self.shared_state['model_path'] = model_filename
            
            return {
                'status': 'success',
                'message': f"Generator model initialized and saved to '{model_filename}'",
                'path': model_filename,
                'img_size': img_size
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def load_and_prepare_data_auto(self, directory: str, img_size: int = 128, 
                                  output_file: str = 'target_data.pt') -> Dict[str, Any]:
        """Load and prepare data automatically without user interaction"""
        try:
            if not os.path.isdir(directory):
                return {
                    'status': 'error',
                    'error': 'Invalid directory path'
                }
            
            images = []
            loaded_files = []
            
            for filename in os.listdir(directory):
                try:
                    img_path = os.path.join(directory, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (img_size, img_size))
                        images.append(img)
                        loaded_files.append(filename)
                except Exception as e:
                    self.logger.warning(f"Could not read {filename}: {e}")
            
            if not images:
                return {
                    'status': 'error',
                    'error': 'No images found or loaded'
                }
            
            # Calculate average image
            avg_image_np = np.mean(images, axis=0).astype(np.uint8)
            target_tensor = torch.from_numpy(avg_image_np).long()
            
            # Save target tensor
            torch.save(target_tensor, output_file)
            
            # Store in shared state
            self.shared_state['target_data'] = target_tensor
            self.shared_state['target_path'] = output_file
            
            return {
                'status': 'success',
                'message': f"Target tensor saved to '{output_file}'",
                'path': output_file,
                'images_loaded': len(images),
                'files': loaded_files[:5]  # First 5 files
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def train_model_batch(self, iterations: int = 100, img_size: int = 128,
                         learning_rate: float = 0.1, 
                         model_filename: str = 'generator_model.pth',
                         target_filename: str = 'target_data.pt',
                         save_interval: int = 50) -> Dict[str, Any]:
        """Train model for specified iterations without visualization"""
        try:
            # Load or create model
            model = PixelGenerator(img_size)
            if os.path.exists(model_filename):
                model.load_state_dict(torch.load(model_filename))
                
            # Setup optimizer and loss
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            loss_fn = nn.CrossEntropyLoss()
            
            # Load target data
            if not os.path.exists(target_filename):
                return {
                    'status': 'error',
                    'error': f'Target data file {target_filename} not found'
                }
                
            target = torch.load(target_filename)
            
            # Training loop
            losses = []
            for i in range(iterations):
                optimizer.zero_grad()
                
                # Calculate loss
                logits_flat = model.logits.view(-1, 256)
                target_flat = target.view(-1)
                loss = loss_fn(logits_flat, target_flat)
                
                # Backprop
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                # Save periodically
                if (i + 1) % save_interval == 0:
                    torch.save(model.state_dict(), model_filename)
                    self.logger.info(f"Iteration {i+1}: Loss = {loss.item():.4f}")
            
            # Final save
            torch.save(model.state_dict(), model_filename)
            
            # Store in shared state
            self.shared_state['model'] = model
            self.shared_state['final_loss'] = losses[-1]
            
            return {
                'status': 'success',
                'message': f'Training completed: {iterations} iterations',
                'final_loss': losses[-1],
                'avg_loss': np.mean(losses),
                'model_saved': model_filename
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def generate_final_image(self, img_size: int = 128,
                           model_filename: str = 'generator_model.pth',
                           output_filename: str = 'final_generated_image.png',
                           return_array: bool = False) -> Dict[str, Any]:
        """Generate final image from trained model"""
        try:
            # Load model
            if not os.path.exists(model_filename):
                return {
                    'status': 'error',
                    'error': f'Model file {model_filename} not found'
                }
                
            model = PixelGenerator(img_size)
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            
            # Generate image
            with torch.no_grad():
                probs = model.get_probs()
                dist = torch.distributions.Categorical(probs)
                generated_tensor = dist.sample()
                img_np = generated_tensor.byte().cpu().numpy()
            
            # Save image
            cv2.imwrite(output_filename, img_np)
            
            result = {
                'status': 'success',
                'message': f'Image saved to {output_filename}',
                'path': output_filename,
                'shape': img_np.shape
            }
            
            if return_array:
                result['image_array'] = img_np.tolist()
                
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def get_model_info(self, model_filename: str = 'generator_model.pth') -> Dict[str, Any]:
        """Get information about a saved model"""
        try:
            if not os.path.exists(model_filename):
                return {
                    'status': 'error',
                    'error': 'Model file not found'
                }
            
            state_dict = torch.load(model_filename)
            
            # Extract info from state dict
            logits_shape = state_dict['logits'].shape
            img_size = logits_shape[0]
            
            return {
                'status': 'success',
                'model_file': model_filename,
                'image_size': img_size,
                'logits_shape': list(logits_shape),
                'file_size': os.path.getsize(model_filename),
                'parameters': sum(p.numel() for p in state_dict.values())
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


# Global instance
wrappers = ScriptWrappers()


# Standalone execution functions that maintain original script behavior
def run_preprocess_standalone():
    """Run preprocess.py in standalone mode"""
    from preprocess import initialize_generator
    initialize_generator()
    

def run_load_standalone():
    """Run load.py in standalone mode"""
    from load import load_and_prepare_data
    load_and_prepare_data()
    

def run_train_standalone():
    """Run train.py in standalone mode"""
    # Import would fail due to missing generative_script_pytorch_2
    # This is handled in the updated train.py below
    pass
    

def run_final_standalone():
    """Run final.py in standalone mode"""
    # Import would fail due to missing generative_script_pytorch_2
    # This is handled in the updated final.py below
    pass


import traceback