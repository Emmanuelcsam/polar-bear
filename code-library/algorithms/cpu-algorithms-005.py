# SPDX-License-Identifier: MIT
"""
ai_models.anomaly_detector
==========================
Pixel‑wise unsupervised anomaly detection via convolutional autoencoder.
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
        self.enc = nn.Sequential(
            _Down(3, 32),   # 1/2
            _Down(32, 64),  # 1/4
            _Down(64, 128), # 1/8
            _Down(128, 256))# 1/16

        self.dec = nn.Sequential(
            _Up(256, 128),
            _Up(128, 64),
            _Up(64, 32),
            _Up(32, 3))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

# ────────────────────────────────────────────────────────────
class AI_AnomalyDetector:
    """
    Replacement for OmniFiberAnalyzer.detect_anomalies_comprehensive().
    Usage:
        detector = AI_AnomalyDetector('cae_last.pth')
        score_map, defects = detector.detect(image_bgr, core_mask)
    """
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(self,
                 weight_path: str | Path,
                 device: str | torch.device = None,
                 recon_loss: str = 'l2'):
        self.device = torch.device(device or
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = CAE().to(self.device).eval()
        ckpt = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.criterion = F.mse_loss if recon_loss == 'l2' else F.l1_loss
        self.trans = T.Compose([
            T.ToTensor(),
            T.Normalize(self._MEAN, self._STD)
        ])
        self.inv_trans = T.Compose([
            T.Normalize(mean=[-m/s for m, s in zip(self._MEAN, self._STD)],
                        std=[1/s for s in self._STD])
        ])

    # ──────────────────────────────
    @torch.no_grad()
    def detect(self,
               img_bgr: np.ndarray,
               mask_fiber: np.ndarray | None = None
               ) -> Tuple[np.ndarray, list[Dict]]:
        """
        Returns
        -------
        score_map : float32 [0..1] same H×W (higher = more anomalous)
        defects   : list[dict] with bbox/area/score (ready for JSON report)
        """
        h, w = img_bgr.shape[:2]
        inp = self.trans(Image.fromarray(cv2.cvtColor(img_bgr,
                                                      cv2.COLOR_BGR2RGB)))
        recon = self.model(inp.unsqueeze(0).to(self.device)).squeeze(0)
        recon = self.inv_trans(recon).clamp(0, 1)

        # reconstruction error per‑pixel
        diff = torch.mean((recon - inp.cpu()).abs(), dim=0).numpy()
        diff = cv2.GaussianBlur(diff, (11, 11), 0)
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        # restrict search to fibre region if provided
        if mask_fiber is not None:
            diff = diff * mask_fiber.astype(np.float32)

        # threshold: mean + 3σ
        thr = float(np.mean(diff) + 3 * np.std(diff))
        bin_map = (diff > thr).astype(np.uint8)

        # find contours -> defect blobs
        contours, _ = cv2.findContours(bin_map, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        defects = []
        for i, c in enumerate(contours, 1):
            area = cv2.contourArea(c)
            if area < 20:        # ignore noise
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            defects.append({
                'defect_id': f"AUTO_{i:03d}",
                'bbox': [int(x), int(y), int(bw), int(bh)],
                'area_px': int(area),
                'confidence': float(diff[y:y+bh, x:x+bw].max())
            })

        return diff, defects