#!/usr/bin/env python3
# deep_image_classifier.py
"""
Deep Image Classifier / Renamer / Auto‑Learner
----------------------------------------------

Features
========
* 🔍 **Deep recursive crawl** through any directory tree you specify.
* 🖼️ **Multimodal feature extraction** (CLIP‑ViT + ResNet‑50 + colour histogram) for robust similarity search.
* 🤖 **Self‑training knowledge‑bank** (`knowledgebank.pkl`) – grows after every run.
* ✨ **Automatic new‑class discovery**: unknown images are clustered, named, and added as fresh classes.
* 🖥️ **Interactive manual mode** with real‑time feedback and on‑the‑fly class creation.
* 🗂️ **Safe renaming** (numbered suffixes for duplicates) that never overwrites originals.
* 📝 **Action log** (`session‑YYYYmmdd‑HHMMSS.log`) written *live* to both console and file.
* ⚙️ **Zero‑hassle dependencies** – missing libraries are auto‑installed (latest versions).
* 🔧 **Optional JSON config** (`config.json`) lets you pre‑set reference & dataset paths and similarity thresholds.

Classification naming convention
================================
`"<core‑diameter>-<fiber_type>-<clean|dirty>-<defect_type>-<extra_characteristics>"`  
…or *any* subset / permutation – the script keeps whatever tokens are present.

--------------------------------------------------------------------
"""

# ───────────────────────── 1. STANDARD LIBS ─────────────────────────
import io
import json
import os
import pickle
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ───────────────────────── 2. AUTO‑INSTALL DEPENDENCIES ────────────
REQUIRED = {
    "torch": "torch",
    "torchvision": "torchvision",
    "open_clip_torch": "open_clip_torch",          # fast CLIP implementation
    "numpy": "numpy",
    "opencv-python": "cv2",
    "scikit-learn": "sklearn",
    "Pillow": "PIL",
    "tqdm": "tqdm",
    "termcolor": "termcolor",                      # pretty CLI output
}

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str, logfile: io.TextIOWrapper = None, color: str = None):
    from termcolor import colored  # safe: ensured below
    txt = f"[{timestamp()}] {msg}"
    print(colored(txt, color) if color else txt, flush=True)
    if logfile:
        logfile.write(txt + "\n")
        logfile.flush()

def ensure(package: str, import_name: str):
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        print(f"[{timestamp()}] Installing {package} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

for pkg, mod in REQUIRED.items():
    ensure(pkg, mod)

# ───────────────────────── 3. THIRD‑PARTY IMPORTS ──────────────────
import cv2
import numpy as np
import torch
import open_clip_torch as open_clip
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ───────────────────────── 4. GLOBAL CONSTANTS ─────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}
KB_FILENAME = "knowledgebank.pkl"
CFG_FILENAME = "config.json"
DEFAULT_SIM_THRESHOLD = 0.72       # can be changed in config or prompt
NEW_CLASS_PREFIX = "autocluster"

# ───────────────────────── 5. FEATURE EXTRACTOR ────────────────────
class FeatureExtractor:
    """Hybrid CLIP (ViT‑B/32) + ResNet‑50 + colour histogram."""
    def __init__(self, device):
        self.device = device
        # CLIP
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.clip_model = model.to(device).eval()
        self.clip_preprocess = preprocess
        # ResNet‑50
        from torchvision import models, transforms
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1]).to(device).eval()
        self.resnet_tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def __call__(self, img_path: str) -> np.ndarray:
        """Return  ×‑normalised concatenated feature vector."""
        img = Image.open(img_path).convert("RGB")

        # CLIP embedding
        clip_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        clip_feat = self.clip_model.encode_image(clip_tensor)
        clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

        # ResNet‑50 embedding
        res_tensor = self.resnet_tf(img).unsqueeze(0).to(self.device)
        res_feat = self.resnet(res_tensor).flatten(1)
        res_feat = res_feat / res_feat.norm(dim=-1, keepdim=True)

        # Colour histogram (HSV, 8×8×8)
        hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256]).flatten()
        hist = hist / np.linalg.norm(hist)

        # Concatenate
        vec = torch.cat([clip_feat.squeeze(), res_feat.squeeze(),
                         torch.from_numpy(hist).to(self.device)])
        return vec.cpu().numpy()

# ───────────────────────── 6. KNOWLEDGE BANK ───────────────────────
class KnowledgeBank:
    """Stores {class_name: [feature_vectors]} and incremental stats."""
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, List[np.ndarray]] = defaultdict(list)
        if path.exists():
            self.load()

    def add(self, class_name: str, feature: np.ndarray):
        self.data[class_name].append(feature)

    def mean_vectors(self) -> Dict[str, np.ndarray]:
        return {c: np.mean(v, axis=0) for c, v in self.data.items() if v}

    # ───────── persistence ─────────
    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(dict(self.data), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.path, "rb") as f:
            self.data = pickle.load(f)  # type: ignore

# ───────────────────────── 7. UTILS ────────────────────────────────
def is_image_file(fp: str) -> bool:
    return Path(fp).suffix.lower() in IMAGE_EXTS

def clean_class_name(stem: str) -> str:
    # strip trailing ‑001 etc.
    return re.sub(r"[-_]\d+$", "", stem)

def unique_filename(directory: Path, basename: str, ext: str) -> Path:
    candidate = directory / f"{basename}{ext}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{basename}-{counter}{ext}"
        counter += 1
    return candidate

def load_config() -> dict:
    if Path(CFG_FILENAME).exists():
        with open(CFG_FILENAME, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                log("config.json exists but is invalid – ignoring.", color="yellow")
    return {}

# ───────────────────────── 8. MAIN OPERATION ───────────────────────
class DeepClassifier:
    def __init__(self, log_stream):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Using device: {self.device}", log_stream, "green")
        self.extractor = FeatureExtractor(self.device)
        self.kb = KnowledgeBank(Path(KB_FILENAME))
        log(f"KnowledgeBank loaded: {len(self.kb.data)} classes.", log_stream, "green")
        self.log_stream = log_stream

    # ───────── reference ingestion ─────────
    def build_reference(self, reference_root: Path):
        log("Scanning reference tree …", self.log_stream, "cyan")
        for root, _, files in os.walk(reference_root):
            for f in files:
                if is_image_file(f):
                    fp = Path(root, f)
                    class_name = clean_class_name(fp.stem)
                    feat = self.extractor(str(fp))
                    self.kb.add(class_name, feat)
        log(f"Reference ingestion complete – {len(self.kb.data)} total classes.", self.log_stream, "green")
        self.kb.save()

    # ───────── similarity search ─────────
    def best_match(self, feat: np.ndarray, threshold: float) -> Tuple[str, float]:
        means = self.kb.mean_vectors()
        if not means:                        # no data yet
            return "", 0.0
        names = list(means.keys())
        mat = np.stack([means[n] for n in names])
        sims = cosine_similarity(feat.reshape(1, -1), mat).flatten()
        idx = sims.argmax()
        return (names[idx], sims[idx]) if sims[idx] >= threshold else ("", sims[idx])

    # ───────── dataset pass (auto) ─────────
    def process_dataset(self, dataset_root: Path, threshold: float):
        # Step 1: collect image paths
        img_files: List[Path] = []
        for root, _, files in os.walk(dataset_root):
            for f in files:
                if is_image_file(f):
                    img_files.append(Path(root, f))
        log(f"Found {len(img_files)} images to analyse.", self.log_stream, "cyan")

        # Step 2: iterate
        unknown_feats = []
        unknown_paths = []

        for fp in tqdm(img_files, desc="Classifying"):
            feat = self.extractor(str(fp))
            cls, sim = self.best_match(feat, threshold)
            if cls:
                # rename
                new_path = unique_filename(fp.parent, cls, fp.suffix)
                fp.rename(new_path)
                log(f"RENAMED {fp.name} → {new_path.name}  (sim={sim:.3f})",
                    self.log_stream, "green")
                self.kb.add(cls, feat)
            else:
                # keep for clustering
                unknown_feats.append(feat)
                unknown_paths.append(fp)

        # Step 3: discover new classes
        if unknown_feats:
            n_clusters = max(1, int(len(unknown_feats) / 25))   # heuristic: 25 imgs per cluster
            km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
            labels = km.fit_predict(np.vstack(unknown_feats))
            for lbl in set(labels):
                members = [p for p, lab in zip(unknown_paths, labels) if lab == lbl]
                new_class = f"{NEW_CLASS_PREFIX}-{datetime.now().strftime('%H%M%S')}-{lbl}"
                for p in members:
                    new_path = unique_filename(p.parent, new_class, p.suffix)
                    p.rename(new_path)
                    feat = self.extractor(str(new_path))
                    self.kb.add(new_class, feat)
                log(f"⚡ Discovered new class '{new_class}' with {len(members)} images.",
                    self.log_stream, "cyan")
        self.kb.save()

    # ───────── manual interactive mode ─────────
    def manual_mode(self, dataset_root: Path, threshold: float):
        imgs = [Path(root, f)
                for root, _, files in os.walk(dataset_root)
                for f in files if is_image_file(f)]
        imgs.sort()
        log(f"Manual mode – {len(imgs)} images queued.", self.log_stream, "cyan")

        for fp in imgs:
            # show image
            img = cv2.imread(str(fp))
            cv2.imshow("Image (press q to quit)", img)
            feat = self.extractor(str(fp))
            guess, sim = self.best_match(feat, threshold)
            print(f"\nFile: {fp}\nGuess: {guess or '<none>'}  (sim={sim:.3f})")
            print("Commands: [y] accept  [n] new class  [c] choose existing  [q] quit")
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('y') and guess:
                new_path = unique_filename(fp.parent, guess, fp.suffix)
                fp.rename(new_path)
                self.kb.add(guess, feat)
                log(f"Accepted → {new_path}", self.log_stream, "green")
            elif key == ord('n'):
                cname = input("Enter new class name: ").strip()
                if cname:
                    new_path = unique_filename(fp.parent, cname, fp.suffix)
                    fp.rename(new_path)
                    self.kb.add(cname, feat)
                    log(f"New class '{cname}' → {new_path}", self.log_stream, "green")
            elif key == ord('c'):
                existing = list(self.kb.data.keys())
                for i, n in enumerate(existing, 1):
                    print(f"{i:>2}. {n}")
                idx = int(input("Select #: ")) - 1
                if 0 <= idx < len(existing):
                    cname = existing[idx]
                    new_path = unique_filename(fp.parent, cname, fp.suffix)
                    fp.rename(new_path)
                    self.kb.add(cname, feat)
                    log(f"Classified as '{cname}' → {new_path}", self.log_stream, "green")
            cv2.destroyAllWindows()
        self.kb.save()

# ───────────────────────── 9. USER FLOW ────────────────────────────
def main():
    # rolling log file
    logpath = Path(f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    with open(logpath, "w", encoding="utf-8") as log_stream:
        log(f"Session log → {logpath}", log_stream, "blue")

        cfg = load_config()
        # prompt helpers
        def ask_path(prompt: str, default: str = "") -> Path:
            while True:
                p = input(f"{prompt} " + (f"[{default}] " if default else "")).strip() or default
                p = os.path.expanduser(p)
                if Path(p).is_dir():
                    return Path(p)
                print("Invalid directory – please try again.")

        def ask_float(prompt: str, default: float) -> float:
            while True:
                s = input(f"{prompt} [{default}]: ").strip() or str(default)
                try:
                    v = float(s)
                    if 0 < v <= 1: return v
                except ValueError:
                    pass
                print("Enter a number between 0 and 1.")

        # mode selection
        mode = ""
        while mode not in {"1", "2"}:
            print("\nSelect mode:\n  [1] Process (automatic)\n  [2] Manual (interactive)")
            mode = input("Choice: ").strip()

        # use config?
        use_cfg = cfg and input("Load paths from config.json? (y/n) ").lower().startswith("y")

        ref_path = Path(cfg.get("reference_path", "")) if use_cfg else ask_path("Reference folder path:")
        data_path = Path(cfg.get("dataset_path", ""))   if use_cfg else ask_path("Dataset folder path:")
        sim_th =   cfg.get("similarity_threshold", DEFAULT_SIM_THRESHOLD) if use_cfg else \
                   ask_float("Similarity threshold (0–1)", DEFAULT_SIM_THRESHOLD)

        # classifier instance
        dc = DeepClassifier(log_stream)

        # ingest reference set every time (cheap & ensures sync)
        dc.build_reference(ref_path)

        start = time.time()
        if mode == "1":
            dc.process_dataset(data_path, sim_th)
        else:
            dc.manual_mode(data_path, sim_th)
        elapsed = time.time() - start
        log(f"Finished in {elapsed/60:.1f} min.", log_stream, "blue")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

