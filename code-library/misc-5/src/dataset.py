# dataset.py
# Copyright (c) 2025, Fiber-AI
#
# A chunk-aware image Dataset for the William-&Mary Bora cluster.
#  • Streams JPEG/PNG/TIFF files from project-directory/dataset/**
#  • Optionally reads pre-built LMDBs for >10× faster network I/O
#  • Plays nicely with SLURM DDP by sharding the index on $SLURM_PROCID
#  • Uses Albumentations → PyTorch tensor transforms

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import lmdb  # optional, only required if you enable LMDB caching
except ImportError:  # keep the import lazy so the code still runs without lmdb
    lmdb = None


def build_default_transforms(train: bool = True, img_size: int = 256) -> A.Compose:
    """Return an Albumentations transform pipeline."""
    if train:
        aug = [
            A.SmallestMaxSize(img_size + 32),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
        ]
    else:
        aug = [
            A.SmallestMaxSize(img_size + 32),
            A.CenterCrop(img_size, img_size),
        ]
    aug += [
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(aug)


class EndfaceChunkDataset(Dataset):
    """
    Parameters
    ----------
    root : str | Path
        Path to project-directory/dataset
    transforms : albumentations.Compose | None
        Data-augmentation pipeline (Albumentations).
    lmdb_dir : str | Path | None
        Optional path to an LMDB folder. If supplied the database will be
        opened **read-only** and used instead of the filesystem.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root: Union[str, Path],
        transforms: Optional[A.Compose] = None,
        lmdb_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.transforms = transforms or build_default_transforms(train=True)
        self.lmdb_env: Optional[lmdb.Environment] = None
        self._keys: List[str]

        if lmdb_dir is not None:
            if lmdb is None:
                raise RuntimeError("lmdb package not available, install with `pip install lmdb`")
            self.lmdb_env = lmdb.open(
                str(lmdb_dir),
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=32,
            )
            with self.lmdb_env.begin(write=False) as txn:
                # Keys are stored as NULL-terminated bytes
                self._keys = [k.decode("ascii") for k, _ in txn.cursor()]
        else:
            self._keys = sorted(
                [str(p) for p in self.root.rglob("*") if p.suffix.lower() in self.IMG_EXTS]
            )

        # -------- Distributed sharding (works for SLURM + torchrun) ----------
        # Each process sees only its own slice of the global index.
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
        if world_size > 1:
            self._keys = self._keys[rank::world_size]

    # --------------------------------------------------------------------- #
    #                           Helper functions                            #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _decode_img(buf: bytes) -> np.ndarray:
        """JPEG/PNG buffer → HWC uint8 OpenCV image."""
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # OpenCV handles RGB-ish decode[24]
        if img is None:
            raise ValueError("imdecode failed (file may be corrupted)")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_from_fs(self, path: str) -> np.ndarray:
        """Read & decode an image from the filesystem."""
        buf = Path(path).read_bytes()
        return self._decode_img(buf)

    def _load_from_lmdb(self, idx: int) -> np.ndarray:
        """Read & decode an image stored in an LMDB entry."""
        assert self.lmdb_env is not None
        key = self._keys[idx].encode("ascii")
        with self.lmdb_env.begin(write=False) as txn:
            buf = txn.get(key)
        if buf is None:
            raise KeyError(f"Missing LMDB key: {key!r}")
        return self._decode_img(buf)

    # --------------------------------------------------------------------- #
    #                           Dataset interface                           #
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        if self.lmdb_env is not None:
            img = self._load_from_lmdb(idx)
            path = self._keys[idx]
        else:
            path = self._keys[idx]
            img = self._load_from_fs(path)

        # Albumentations expects HWC uint8
        sample = self.transforms(image=img)
        tensor = sample["image"]  # CHW float32 in [0,1] after Normalize+ToTensorV2

        return tensor, path  # return path for debugging / provenance

    # --------------------------------------------------------------------- #
    #                        LMDB-safe multiprocessing                       #
    # --------------------------------------------------------------------- #
    def __getstate__(self):
        # LMDB envs cannot be pickled; we drop them for DataLoader workers[33]
        state = self.__dict__.copy()
        state["lmdb_env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if state["lmdb_env"] is None and hasattr(self, "_keys") and lmdb is not None:
            # Re-open inside the worker process if needed
            lmdb_dir = os.getenv("FIBER_LMDB_PATH")
            if lmdb_dir:
                self.lmdb_env = lmdb.open(
                    lmdb_dir, readonly=True, lock=False, readahead=False, max_readers=32
                )


# ------------------- convenience factory ---------------------------------- #
def make_dataloader(
    root: str | Path,
    batch_size: int = 8,
    train: bool = True,
    num_workers: int = 8,
    lmdb_dir: str | None = None,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    tfms = build_default_transforms(train)
    ds = EndfaceChunkDataset(root, tfms, lmdb_dir)
    sampler = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=shuffle)
        shuffle = False  # Sampler already handles shuffling
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return dl 