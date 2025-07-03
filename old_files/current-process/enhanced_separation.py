"""
Enhanced Separation Module
- ML-powered segmentation
- Multi-method consensus from old-processes
- In-memory processing
- No argparse
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import concurrent.futures
import subprocess
import tempfile
import json
import ast
import inspect
from dataclasses import dataclass
from collections import defaultdict

from config_manager import get_config
from enhanced_logging import get_logger, log_execution, log_performance, ProgressLogger

logger = get_logger(__name__)


@dataclass
class SegmentationResult:
    """Result from a segmentation method"""
    method_name: str
    core_mask: Optional[np.ndarray] = None
    cladding_mask: Optional[np.ndarray] = None
    ferrule_mask: Optional[np.ndarray] = None
    confidence: float = 0.0
    parameters: Dict[str, Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class MLSegmentationModel:
    """ML-powered segmentation using U-Net or similar"""
    
    def __init__(self):
        self.config = get_config()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained segmentation model"""
        if not self.config.processing.ml_enabled:
            return
        
        try:
            if self.config.processing.pytorch_enabled:
                self._load_pytorch_unet()
            elif self.config.processing.tensorflow_enabled:
                self._load_tensorflow_unet()
        except Exception as e:
            logger.warning(f"Failed to load ML segmentation model: {e}")
    
    def _load_pytorch_unet(self):
        """Load PyTorch U-Net model"""
        try:
            import torch
            import torch.nn as nn
            
            class UNet(nn.Module):
                def __init__(self, n_channels=3, n_classes=4):
                    super().__init__()
                    # Encoder
                    self.enc1 = self._block(n_channels, 64)
                    self.enc2 = self._block(64, 128)
                    self.enc3 = self._block(128, 256)
                    self.enc4 = self._block(256, 512)
                    
                    # Bottleneck
                    self.bottleneck = self._block(512, 1024)
                    
                    # Decoder
                    self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
                    self.dec4 = self._block(1024, 512)
                    self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                    self.dec3 = self._block(512, 256)
                    self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                    self.dec2 = self._block(256, 128)
                    self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                    self.dec1 = self._block(128, 64)
                    
                    # Output
                    self.out = nn.Conv2d(64, n_classes, 1)
                    self.pool = nn.MaxPool2d(2)
                
                def _block(self, in_ch, out_ch):
                    return nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    )
                
                def forward(self, x):
                    # Encoder
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool(e1))
                    e3 = self.enc3(self.pool(e2))
                    e4 = self.enc4(self.pool(e3))
                    
                    # Bottleneck
                    b = self.bottleneck(self.pool(e4))
                    
                    # Decoder
                    d4 = self.dec4(torch.cat([self.upconv4(b), e4], 1))
                    d3 = self.dec3(torch.cat([self.upconv3(d4), e3], 1))
                    d2 = self.dec2(torch.cat([self.upconv2(d3), e2], 1))
                    d1 = self.dec1(torch.cat([self.upconv1(d2), e1], 1))
                    
                    return self.out(d1)
            
            self.model = UNet(n_channels=3, n_classes=4)  # bg, core, cladding, ferrule
            
            # Load weights if available
            model_path = self.config.model_dir / "unet_segmentation.pth"
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                logger.info("Loaded PyTorch U-Net segmentation model")
            else:
                logger.info("Using untrained PyTorch U-Net model")
                
        except ImportError:
            logger.warning("PyTorch not available for segmentation")
    
    def _load_tensorflow_unet(self):
        """Load TensorFlow U-Net model"""
        try:
            import tensorflow as tf
            
            def unet_model(input_shape=(None, None, 3), n_classes=4):
                inputs = tf.keras.Input(shape=input_shape)
                
                # Encoder
                c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
                c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
                p1 = tf.keras.layers.MaxPooling2D()(c1)
                
                c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
                c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
                p2 = tf.keras.layers.MaxPooling2D()(c2)
                
                c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
                c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
                p3 = tf.keras.layers.MaxPooling2D()(c3)
                
                c4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
                c4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
                p4 = tf.keras.layers.MaxPooling2D()(c4)
                
                # Bottleneck
                c5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
                c5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)
                
                # Decoder
                u6 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
                u6 = tf.keras.layers.concatenate([u6, c4])
                c6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
                c6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(c6)
                
                u7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
                u7 = tf.keras.layers.concatenate([u7, c3])
                c7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
                c7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c7)
                
                u8 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
                u8 = tf.keras.layers.concatenate([u8, c2])
                c8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
                c8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c8)
                
                u9 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
                u9 = tf.keras.layers.concatenate([u9, c1])
                c9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
                c9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c9)
                
                outputs = tf.keras.layers.Conv2D(n_classes, 1, activation='softmax')(c9)
                
                return tf.keras.Model(inputs=[inputs], outputs=[outputs])
            
            model_path = self.config.model_dir / "unet_segmentation.h5"
            if model_path.exists():
                self.model = tf.keras.models.load_model(model_path)
                logger.info("Loaded TensorFlow U-Net segmentation model")
            else:
                self.model = unet_model()
                logger.info("Using untrained TensorFlow U-Net model")
                
        except ImportError:
            logger.warning("TensorFlow not available for segmentation")
    
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """Predict segmentation masks using ML model"""
        if self.model is None:
            return SegmentationResult(
                method_name="ml_segmentation",
                error="ML model not available"
            )
        
        try:
            # Preprocess image
            h, w = image.shape[:2]
            
            # Resize to multiple of 32 for U-Net
            new_h = ((h + 31) // 32) * 32
            new_w = ((w + 31) // 32) * 32
            resized = cv2.resize(image, (new_w, new_h))
            
            if self.config.processing.pytorch_enabled:
                import torch
                x = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                
                with torch.no_grad():
                    output = self.model(x)
                    probs = torch.softmax(output, dim=1).squeeze(0).numpy()
            else:
                x = resized[np.newaxis, ...] / 255.0
                probs = self.model.predict(x, verbose=0)[0]
                probs = probs.transpose(2, 0, 1)  # HWC to CHW
            
            # Extract masks (0: bg, 1: core, 2: cladding, 3: ferrule)
            core_mask = (probs[1] > 0.5).astype(np.uint8) * 255
            cladding_mask = (probs[2] > 0.5).astype(np.uint8) * 255
            ferrule_mask = (probs[3] > 0.5).astype(np.uint8) * 255
            
            # Resize back to original
            core_mask = cv2.resize(core_mask, (w, h))
            cladding_mask = cv2.resize(cladding_mask, (w, h))
            ferrule_mask = cv2.resize(ferrule_mask, (w, h))
            
            # Calculate confidence
            confidence = float(np.mean([
                probs[1].max(),
                probs[2].max(),
                probs[3].max()
            ]))
            
            return SegmentationResult(
                method_name="ml_segmentation",
                core_mask=core_mask,
                cladding_mask=cladding_mask,
                ferrule_mask=ferrule_mask,
                confidence=confidence,
                parameters={"model_type": "unet"}
            )
            
        except Exception as e:
            logger.error(f"ML segmentation failed: {e}")
            return SegmentationResult(
                method_name="ml_segmentation",
                error=str(e)
            )


class MethodExecutor:
    """Executes segmentation methods in isolated processes"""
    
    def __init__(self):
        self.config = get_config()
        self.zone_methods_dir = Path("zone_methods")
    
    def discover_methods(self) -> List[str]:
        """Discover available segmentation methods"""
        methods = []
        
        for file_path in self.zone_methods_dir.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
            
            try:
                with open(file_path, 'r') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == "segment_zones":
                        methods.append(file_path.stem)
                        logger.debug(f"Found method: {file_path.stem}")
                        break
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
        
        return methods
    
    def execute_method(self, method_name: str, image: np.ndarray, 
                      variation_results: Dict[str, np.ndarray]) -> SegmentationResult:
        """Execute a segmentation method in isolated process"""
        start_time = cv2.getTickCount()
        
        try:
            # Create temporary files for communication
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as img_file:
                np.save(img_file.name, image)
                img_path = img_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as var_file:
                # Save variation names
                json.dump(list(variation_results.keys()), var_file)
                var_path = var_file.name
            
            # Prepare command
            script = f"""
import sys
sys.path.insert(0, '{self.zone_methods_dir}')
import numpy as np
import json
from {method_name} import segment_zones

# Load data
image = np.load('{img_path}')
with open('{var_path}', 'r') as f:
    variation_names = json.load(f)

# Create mock variation results
variation_results = {{name: image for name in variation_names}}

# Execute segmentation
try:
    masks = segment_zones(image, variation_results)
    
    # Save results
    result = {{
        'success': True,
        'core_mask': masks.get('core'),
        'cladding_mask': masks.get('cladding'),
        'ferrule_mask': masks.get('ferrule')
    }}
except Exception as e:
    result = {{
        'success': False,
        'error': str(e)
    }}

# Save result
import pickle
with open('{img_path}.result', 'wb') as f:
    pickle.dump(result, f)
"""
            
            # Execute in subprocess
            process = subprocess.Popen(
                [sys.executable, '-c', script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait with timeout
            timeout = self.config.separation.timeout_seconds
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Load result
            import pickle
            with open(f"{img_path}.result", 'rb') as f:
                result = pickle.load(f)
            
            # Clean up temp files
            Path(img_path).unlink()
            Path(var_path).unlink()
            Path(f"{img_path}.result").unlink()
            
            # Process result
            if result['success']:
                execution_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                
                return SegmentationResult(
                    method_name=method_name,
                    core_mask=result['core_mask'],
                    cladding_mask=result['cladding_mask'],
                    ferrule_mask=result['ferrule_mask'],
                    confidence=0.8,  # Default confidence
                    execution_time=execution_time
                )
            else:
                return SegmentationResult(
                    method_name=method_name,
                    error=result['error']
                )
            
        except subprocess.TimeoutExpired:
            logger.error(f"Method {method_name} timed out")
            return SegmentationResult(
                method_name=method_name,
                error="Timeout"
            )
        except Exception as e:
            logger.error(f"Method {method_name} failed: {e}")
            return SegmentationResult(
                method_name=method_name,
                error=str(e)
            )


class ConsensusBuilder:
    """Builds consensus from multiple segmentation results"""
    
    def __init__(self):
        self.config = get_config()
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> Dict[str, float]:
        """Load method performance scores"""
        kb_path = self.config.knowledge_base_path
        
        if kb_path.exists():
            try:
                with open(kb_path, 'r') as f:
                    data = json.load(f)
                    return data.get('segmentation_scores', {})
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")
        
        return {}
    
    def _save_knowledge_base(self):
        """Save updated method scores"""
        kb_path = self.config.knowledge_base_path
        
        try:
            # Load existing data
            if kb_path.exists():
                with open(kb_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {}
            
            # Update segmentation scores
            data['segmentation_scores'] = self.knowledge_base
            
            # Save
            with open(kb_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
    
    def build_consensus(self, results: List[SegmentationResult], 
                       image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Build consensus masks from multiple results"""
        h, w = image_shape
        
        # Filter successful results
        valid_results = [r for r in results if r.error is None]
        
        if not valid_results:
            logger.warning("No valid segmentation results")
            return self._create_default_masks(h, w)
        
        # Get method weights
        weights = {}
        for result in valid_results:
            weight = self.knowledge_base.get(result.method_name, 0.5)
            weight *= result.confidence
            weights[result.method_name] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Weighted voting for each zone
        consensus_masks = {}
        
        for zone in ['core', 'cladding', 'ferrule']:
            # Accumulate weighted votes
            vote_map = np.zeros((h, w), dtype=np.float32)
            
            for result in valid_results:
                mask = getattr(result, f"{zone}_mask")
                if mask is not None:
                    weight = weights.get(result.method_name, 0)
                    vote_map += (mask > 0).astype(np.float32) * weight
            
            # Apply threshold
            threshold = self.config.separation.consensus_threshold
            consensus_mask = (vote_map >= threshold).astype(np.uint8) * 255
            
            # Post-process mask
            consensus_mask = self._post_process_mask(consensus_mask, zone)
            consensus_masks[zone] = consensus_mask
        
        # Update knowledge base with IoU scores
        self._update_scores(valid_results, consensus_masks)
        
        return consensus_masks
    
    def _post_process_mask(self, mask: np.ndarray, zone: str) -> np.ndarray:
        """Post-process mask to improve quality"""
        # Remove small components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Keep largest component for core
        if zone == 'core':
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Keep largest
                largest = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(mask)
                cv2.drawContours(mask, [largest], -1, 255, -1)
        
        return mask
    
    def _update_scores(self, results: List[SegmentationResult], 
                      consensus_masks: Dict[str, np.ndarray]):
        """Update method scores based on agreement with consensus"""
        learning_rate = self.config.separation.learning_rate
        
        for result in results:
            ious = []
            
            # Calculate IoU for each zone
            for zone in ['core', 'cladding', 'ferrule']:
                result_mask = getattr(result, f"{zone}_mask")
                consensus_mask = consensus_masks.get(zone)
                
                if result_mask is not None and consensus_mask is not None:
                    intersection = np.logical_and(result_mask > 0, consensus_mask > 0).sum()
                    union = np.logical_or(result_mask > 0, consensus_mask > 0).sum()
                    
                    if union > 0:
                        iou = intersection / union
                        ious.append(iou)
            
            # Update score
            if ious:
                avg_iou = np.mean(ious)
                old_score = self.knowledge_base.get(result.method_name, 0.5)
                new_score = old_score * (1 - learning_rate) + avg_iou * learning_rate
                self.knowledge_base[result.method_name] = new_score
                
                logger.debug(f"Updated {result.method_name} score: {old_score:.3f} -> {new_score:.3f}")
        
        # Save updated scores
        self._save_knowledge_base()
    
    def _create_default_masks(self, h: int, w: int) -> Dict[str, np.ndarray]:
        """Create default masks when no methods succeed"""
        # Simple circular approximation
        center = (w // 2, h // 2)
        radius = min(center) * 0.8
        
        masks = {}
        
        # Core (inner circle)
        core_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(core_mask, center, int(radius * 0.2), 255, -1)
        masks['core'] = core_mask
        
        # Cladding (ring)
        cladding_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(cladding_mask, center, int(radius * 0.6), 255, -1)
        cv2.circle(cladding_mask, center, int(radius * 0.2), 0, -1)
        masks['cladding'] = cladding_mask
        
        # Ferrule (outer ring)
        ferrule_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ferrule_mask, center, int(radius), 255, -1)
        cv2.circle(ferrule_mask, center, int(radius * 0.6), 0, -1)
        masks['ferrule'] = ferrule_mask
        
        return masks


class EnhancedSeparator:
    """Enhanced separator with ML and consensus"""
    
    def __init__(self):
        self.config = get_config()
        self.ml_model = MLSegmentationModel()
        self.executor = MethodExecutor()
        self.consensus_builder = ConsensusBuilder()
        
        # Thread pool for parallel execution
        if self.config.separation.parallel_execution:
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.processing.max_workers
            )
        else:
            self.thread_pool = None
    
    @log_performance
    def separate_zones(self, image: np.ndarray, 
                      variation_results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Separate image into core, cladding, and ferrule zones"""
        logger.info("Starting zone separation")
        
        results = []
        
        # ML segmentation
        if self.config.detection.use_ml_detection:
            ml_result = self.ml_model.predict(image)
            if ml_result.error is None:
                results.append(ml_result)
                logger.info("ML segmentation completed", confidence=ml_result.confidence)
        
        # Traditional methods
        methods = self.executor.discover_methods()
        enabled_methods = [m for m in methods if m in self.config.separation.methods_enabled]
        
        logger.info(f"Running {len(enabled_methods)} traditional methods")
        
        if self.thread_pool:
            # Parallel execution
            futures = []
            for method in enabled_methods:
                future = self.thread_pool.submit(
                    self.executor.execute_method,
                    method, image, variation_results
                )
                futures.append((method, future))
            
            # Collect results
            progress = ProgressLogger("separation_methods", len(futures))
            for method, future in futures:
                try:
                    result = future.result()
                    results.append(result)
                    progress.update(1, f"Completed {method}")
                except Exception as e:
                    logger.error(f"Method {method} failed: {e}")
        else:
            # Sequential execution
            for method in enabled_methods:
                result = self.executor.execute_method(method, image, variation_results)
                results.append(result)
        
        # Build consensus
        consensus_masks = self.consensus_builder.build_consensus(results, image.shape[:2])
        
        # Log statistics
        successful = len([r for r in results if r.error is None])
        logger.info(f"Separation complete", 
                   total_methods=len(results),
                   successful=successful,
                   consensus_threshold=self.config.separation.consensus_threshold)
        
        return consensus_masks
    
    def visualize_separation(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
        """Create visualization of separated zones"""
        overlay = image.copy()
        
        # Define colors for each zone
        colors = {
            'core': (0, 255, 0),      # Green
            'cladding': (255, 255, 0), # Yellow
            'ferrule': (255, 0, 0)     # Red
        }
        
        # Apply colored overlays
        for zone, color in colors.items():
            mask = masks.get(zone)
            if mask is not None:
                # Create colored mask
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0] = color
                
                # Blend with original
                alpha = 0.3
                overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
        
        # Add contours
        for zone, color in colors.items():
            mask = masks.get(zone)
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
        
        return overlay
    
    def __del__(self):
        """Cleanup"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


def main():
    """Test the enhanced separator"""
    config = get_config()
    config.interactive_mode = False
    
    separator = EnhancedSeparator()
    
    # Test with sample image
    test_image = config.input_dir / "test.jpg"
    if test_image.exists():
        image = cv2.imread(str(test_image))
        
        # Mock variation results
        variations = {"original": image}
        
        # Separate zones
        masks = separator.separate_zones(image, variations)
        
        # Visualize
        vis = separator.visualize_separation(image, masks)
        
        # Save result
        output_path = config.output_dir / "separation_test.jpg"
        cv2.imwrite(str(output_path), vis)
        logger.info(f"Saved visualization to {output_path}")
    else:
        logger.warning("No test image found")


if __name__ == "__main__":
    main()