# SPDX-License-Identifier: MIT
"""
ai_models.segmenter
===================
One‑shot multi‑class U‑Net for fibre‑optic end‑face inspection.

Outputs 4 classes:
0 = background, 1 = fibre core, 2 = cladding, 3 = ferrule+ferrule‑edge,
4 = defect/anomaly (optional – can be disabled at inference time).
"""

from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# ────────────────────────────────────────────────────────────
# A minimal U‑Net with ResNet‑34 encoder (pre‑trained on ImageNet)
# ----------------------------------------------------------------

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
        # centre‑crop if needed (possible when odd dims)
        diffY = bridge.size()[2] - x.size()[2]
        diffX = bridge.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([bridge, x], 1)
        return self.conv(x)

class UNet34(nn.Module):
    """U‑Net encoder = torchvision.resnet34() layers 0‑7"""
    def __init__(self, n_classes: int = 4, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet34
        base = resnet34(pretrained=pretrained)
        self.input = nn.Sequential(base.conv1, base.bn1,
                                   base.relu, base.maxpool)      # 1/4
        self.enc1 = base.layer1                                     # 1/4
        self.enc2 = base.layer2                                     # 1/8
        self.enc3 = base.layer3                                     # 1/16
        self.enc4 = base.layer4                                     # 1/32

        self.center = _ConvBlock(512, 512)

        self.up4 = _UpBlock(512, 256, 256)
        self.up3 = _UpBlock(256, 128, 128)
        self.up2 = _UpBlock(128, 64,  64)
        self.up1 = _UpBlock(64,  64,  32)

        self.logits = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x0 = self.input(x)   # (64)    1/4
        x1 = self.enc1(x0)   # (64)    1/4
        x2 = self.enc2(x1)   # (128)   1/8
        x3 = self.enc3(x2)   # (256)   1/16
        x4 = self.enc4(x3)   # (512)   1/32
        center = self.center(x4)

        d4 = self.up4(center, x3)   # (256)
        d3 = self.up3(d4, x2)       # (128)
        d2 = self.up2(d3, x1)       # (64)
        d1 = self.up1(d2, x0)       # (32)

        return self.logits(d1)


# ────────────────────────────────────────────────────────────
# Inference wrapper
# ----------------------------------------------------------------

class AI_Segmenter:
    """
    Drop‑in replacement for UnifiedSegmentationSystem.
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(self,
                 weight_path: str | Path,
                 device: str | torch.device = None,
                 n_classes: int = 4):
        self.device = torch.device(device or
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = UNet34(n_classes=n_classes)
        ckpt = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval().to(self.device)

        # torchvision‑like preprocessing
        self.trans = T.Compose([
            T.ToTensor(),
            T.Normalize(self._MEAN, self._STD)
        ])

    # ──────────────────────────────
    def segment(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Args
        ----
        img : BGR uint8 image from cv2 (H×W×3)

        Returns
        -------
        dict with binary masks: 'core', 'cladding', 'ferrule', 'defect'
        """
        h, w = img.shape[:2]
        inp = self.trans(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        with torch.no_grad():
            logits = self.model(inp.unsqueeze(0).to(self.device))
        mask = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        out = {
            'core':      (mask == 1).astype(np.uint8),
            'cladding':  (mask == 2).astype(np.uint8),
            'ferrule':   (mask == 3).astype(np.uint8),
            'defect':    (mask == 4).astype(np.uint8)
        }
        # simple morphological clean‑up
        kernel = np.ones((3,3), np.uint8)
        for k in out: out[k] = cv2.morphologyEx(out[k], cv2.MORPH_OPEN, kernel)
        return out