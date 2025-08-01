#!/usr/bin/env python3
"""
torch_quality_classifier.py
---------------------------
A lightweight PyTorch image‑quality classifier for the unified fibre‑optic
pipeline.

•  The model architecture is the same "100‑seconds‑of‑PyTorch" CNN described in
   the transcript (Flatten → [Linear + ReLU] × 2 → Linear → Softmax).
•  It supports:
      – training from scratch or fine‑tuning on your own data
      – exporting / importing weights (`*.pt`) for inference‑only deployments
      – automatic CPU/GPU selection (CUDA if available)
•  Integration points:
      – `predict(image: np.ndarray) -> (label: str, confidence: float)`
      – optional `train(...)` helper for quick prototyping
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2

# -----------------------------  MODEL DEFINITION  -----------------------------


class SimpleCNN(nn.Module):
    """CNN identical to the one sketched in the transcript."""
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


# -----------------------------  DATA PIPELINE  --------------------------------

class FibreDataset(Dataset):
    """
    Very small helper dataset that expects:
        root/
          good/      (all "PASS" or "OK" images)
          defective/ (all anomalous images)
    If your use‑case has more classes just add more sub‑folders.
    """
    def __init__(self, root: str | Path, img_size: int = 28) -> None:
        self.samples: List[Tuple[Path, int]] = []
        self.classes: Dict[str, int] = {}

        root = Path(root)
        for idx, class_dir in enumerate(sorted([p for p in root.iterdir() if p.is_dir()])):
            self.classes[class_dir.name] = idx
            for img in class_dir.glob("*.png"):
                self.samples.append((img, idx))
            for img in class_dir.glob("*.jpg"):
                self.samples.append((img, idx))

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),                 # → [0, 1]
        ])

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read {img_path}")
        img_tensor = self.transform(img)
        return img_tensor, label


# -----------------------------  TRAIN / EVAL  ---------------------------------

class TorchQualityClassifier:
    """
    Thin wrapper that can be instantiated once (e.g. in `detection.py`) and
    re‑used for every image.
    """
    class_map = {0: "PASS", 1: "FAIL"}      # edit if you add more classes

    def __init__(self, weights: str | Path | None = None,
                 device: torch.device | None = None) -> None:
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = SimpleCNN(n_classes=len(self.class_map)).to(self.device)
        if weights:
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.eval()

        # Same 28×28 preprocessing that is described in the transcript
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((28, 28)),
            T.ToTensor(),
        ])

    # ---------- Inference -----------------------------------------------------

    @torch.no_grad()
    def predict(self, img_np: np.ndarray) -> Tuple[str, float]:
        """
        Args:
            img_np – BGR or grayscale OpenCV image.

        Returns:
            (label, confidence)  e.g.  ("FAIL", 0.92)
        """
        if img_np.ndim == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        x = self.preprocess(img_np).unsqueeze(0).to(self.device)   # [1, 1, 28, 28]
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return self.class_map[idx], float(probs[idx])

    # ---------- Quick training (optional) -------------------------------------

    def train(self,
              data_root: str | Path,
              epochs: int = 10,
              batch_size: int = 64,
              lr: float = 1e-3,
              val_split: float = 0.2,
              save_to: str | Path | None = None) -> None:
        """
        Quick helper to (re)train the model on your own labelled data.
        Each class = sub‑folder name under data_root.

        Typical call:
            tc = TorchQualityClassifier()
            tc.train("training_data/", epochs=15, save_to="torch_weights.pt")
        """
        ds = FibreDataset(data_root)
        val_len = int(len(ds) * val_split)
        train_len = len(ds) - val_len
        train_set, val_set = torch.utils.data.random_split(ds, [train_len, val_len])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=batch_size)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                opt.step()

            # ---- validation ----
            self.model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb).argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    total += yb.numel()
            print(f"[{epoch:02}/{epochs}] validation acc = {correct / total:.3%}")

        if save_to:
            torch.save(self.model.state_dict(), save_to)
            print(f"✓ Saved weights to {save_to}")