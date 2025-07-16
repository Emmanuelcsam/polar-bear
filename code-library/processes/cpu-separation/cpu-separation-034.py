
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict

from .base_module import BaseModule

# ---------------------------------------------------------------------------- #
#                           Image Segmentation Module                          #
# ---------------------------------------------------------------------------- #

class _ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class _UpBlock(nn.Module):
    def __init__(self, in_c, bridge_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = nn.Sequential(_ConvBlock(out_c + bridge_c, out_c),
                                  _ConvBlock(out_c, out_c))

    def forward(self, x, bridge):
        x = self.up(x)
        diffY = bridge.size()[2] - x.size()[2]
        diffX = bridge.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([bridge, x], 1)
        return self.conv(x)

class UNet34(nn.Module):
    def __init__(self, n_classes: int = 4, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet34
        base = resnet34(pretrained=pretrained)
        self.input = nn.Sequential(base.conv1, base.bn1,
                                   base.relu, base.maxpool)
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4
        self.center = _ConvBlock(512, 512)
        self.up4 = _UpBlock(512, 256, 256)
        self.up3 = _UpBlock(256, 128, 128)
        self.up2 = _UpBlock(128, 64,  64)
        self.up1 = _UpBlock(64,  64,  32)
        self.logits = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x0 = self.input(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        center = self.center(x4)
        d4 = self.up4(center, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        return self.logits(d1)

class SegmentationModule(BaseModule):
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = self.config.get("segmentation_settings.n_classes", 4)
        self.model = UNet34(n_classes=self.n_classes)
        
        weights_path = self.config.get("paths.model_weights") + "/" + self.config.get("segmentation_settings.default_weights")
        if Path(weights_path).exists():
            self.logger.info(f"Loading segmentation model weights from {weights_path}")
            ckpt = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
        else:
            self.logger.warning(f"Segmentation model weights not found at {weights_path}. Model is untrained.")
            
        self.model.eval().to(self.device)
        self.trans = T.Compose([T.ToTensor(), T.Normalize(self._MEAN, self._STD)])
        
        self._register_tunable_parameter("morph_kernel_size", 3)

    def execute(self, img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        self.logger.info("Executing image segmentation.")
        h, w = img_bgr.shape[:2]
        inp = self.trans(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        with torch.no_grad():
            logits = self.model(inp.unsqueeze(0).to(self.device))
        mask = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        out = {
            'core':      (mask == 1).astype(np.uint8),
            'cladding':  (mask == 2).astype(np.uint8),
            'ferrule':   (mask == 3).astype(np.uint8),
            'defect':    (mask == 4).astype(np.uint8)
        }
        
        kernel_size = self.get_tunable_parameters()["morph_kernel_size"]
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        for k in out:
            out[k] = cv2.morphologyEx(out[k], cv2.MORPH_OPEN, kernel)
        
        self.logger.info("Segmentation complete.")
        return out

# ---------------------------------------------------------------------------- #
#                            Anomaly Detection Module                          #
# ---------------------------------------------------------------------------- #

class _Down(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c,  out_c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

class _Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(_Down(3, 32), _Down(32, 64), _Down(64, 128), _Down(128, 256))
        self.dec = nn.Sequential(_Up(256, 128), _Up(128, 64), _Up(64, 32), _Up(32, 3))
    def forward(self, x):
        return self.dec(self.enc(x))

class AnomalyDetectionModule(BaseModule):
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CAE().to(self.device).eval()
        
        weights_path = self.config.get("paths.model_weights") + "/" + self.config.get("anomaly_settings.default_weights")
        if Path(weights_path).exists():
            self.logger.info(f"Loading anomaly model weights from {weights_path}")
            ckpt = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
        else:
            self.logger.warning(f"Anomaly model weights not found at {weights_path}. Model is untrained.")

        recon_loss = self.config.get("anomaly_settings.reconstruction_loss", "l2")
        self.criterion = F.mse_loss if recon_loss == 'l2' else F.l1_loss
        
        self.trans = T.Compose([T.ToTensor(), T.Normalize(self._MEAN, self._STD)])
        self.inv_trans = T.Compose([
            T.Normalize(mean=[-m/s for m, s in zip(self._MEAN, self._STD)], std=[1/s for s in self._STD])
        ])
        
        self._register_tunable_parameter("threshold_std_factor", 3.0)
        self._register_tunable_parameter("min_defect_area", 20)
        self._register_tunable_parameter("gaussian_blur_kernel", 11)

    @torch.no_grad()
    def execute(self, img_bgr: np.ndarray, mask_fiber: np.ndarray = None) -> Tuple[np.ndarray, list[Dict]]:
        self.logger.info("Executing anomaly detection.")
        h, w = img_bgr.shape[:2]
        inp = self.trans(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        recon = self.model(inp.unsqueeze(0).to(self.device)).squeeze(0)
        recon = self.inv_trans(recon).clamp(0, 1)

        diff = torch.mean((recon - inp.cpu()).abs(), dim=0).numpy()
        
        blur_kernel = int(self.get_tunable_parameters()['gaussian_blur_kernel'])
        if blur_kernel % 2 == 0: blur_kernel += 1 # Must be odd
        diff = cv2.GaussianBlur(diff, (blur_kernel, blur_kernel), 0)
        
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        if mask_fiber is not None:
            diff = diff * mask_fiber.astype(np.float32)

        std_factor = self.get_tunable_parameters()['threshold_std_factor']
        thr = float(np.mean(diff) + std_factor * np.std(diff))
        bin_map = (diff > thr).astype(np.uint8)

        contours, _ = cv2.findContours(bin_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        min_area = self.get_tunable_parameters()['min_defect_area']
        for i, c in enumerate(contours, 1):
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            defects.append({
                'defect_id': f"AUTO_{i:03d}",
                'bbox': [int(x), int(y), int(bw), int(bh)],
                'area_px': int(area),
                'confidence': float(diff[y:y+bh, x:x+bw].max())
            })
            
        self.logger.info(f"Anomaly detection complete. Found {len(defects)} defects.")
        return diff, defects

# ---------------------------------------------------------------------------- #
#                        Quality Classification Module                         #
# ---------------------------------------------------------------------------- #

class SimpleCNN(nn.Module):
    """Simple CNN for quality classification."""
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)

class QualityClassificationModule(BaseModule):
    
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.class_map = {0: "PASS", 1: "FAIL"}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN(n_classes=len(self.class_map)).to(self.device)
        
        # Note: This model doesn't have pre-trained weights in the original scripts,
        # so we assume it needs to be trained first.
        self.logger.warning("Quality classification model is initialized with random weights.")

        self.model.eval()
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((28, 28)),
            T.ToTensor(),
        ])
        
        self._register_tunable_parameter("confidence_threshold", 0.5)

    @torch.no_grad()
    def execute(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        self.logger.info("Executing quality classification.")
        if img_bgr.ndim == 3:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        x = self.preprocess(img_bgr).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = self.class_map[idx]
        
        self.logger.info(f"Classification complete. Result: {label} (Confidence: {confidence:.2f})")
        return label, confidence

# ---------------------------------------------------------------------------- #
#                      TensorFlow Classification Module                        #
# ---------------------------------------------------------------------------- #

# We need to ensure tensorflow is imported if available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class TFClassificationModule(BaseModule):

    def __init__(self, config_manager):
        super().__init__(config_manager)
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow is not installed. This module cannot be used.")
            self.model = None
            return

        self.class_names = [f"class_{i}" for i in range(10)]
        self.model = self._build_model()
        
        # Note: This model also starts untrained.
        self.logger.warning("TensorFlow classification model is initialized with random weights.")

    def _build_model(self, input_shape: Tuple[int, int] = (28, 28), num_classes: int = 10) -> keras.Model:
        model = keras.Sequential(name="simple_mlp")
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(num_classes, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def execute(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        if not self.model:
            self.logger.error("TensorFlow model is not available.")
            return "ERROR", 0.0

        self.logger.info("Executing TensorFlow classification.")
        if img_bgr.ndim == 3:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Preprocess for Fashion-MNIST style
        img_resized = cv2.resize(img_bgr, (28, 28), interpolation=cv2.INTER_AREA) / 255.0
        img_tensor = np.expand_dims(img_resized, axis=0) # Add batch dimension

        pred = self.model(img_tensor, training=False).numpy()[0]
        idx = int(np.argmax(pred))
        label = self.class_names[idx]
        confidence = float(pred[idx])

        self.logger.info(f"TF classification complete. Result: {label} (Confidence: {confidence:.2f})")
        return label, confidence
