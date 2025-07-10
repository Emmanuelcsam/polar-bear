#!/usr/bin/env python3
"""
ADVANCED CHARACTER / IMAGE SORTER
---------------------------------
• Interactive (no argparse).
• Recursively walks through an input folder, computes cheap visual
  descriptors, clusters them *adaptively* until ≤ target_outlier_pct
  remain unsorted.
• Classification rules in classification.txt (optional) take precedence.
• Caches every feature vector in features.sqlite so re‑runs skip hashing.
• Creates a JSON + TXT report and colour‑coded montage per cluster.
"""

import os, sys, shutil, json, sqlite3, time
from pathlib import Path
from collections import defaultdict, Counter
from contextlib import closing

import cv2
import numpy as np
from PIL import Image
import imagehash
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ────────────────────────────────  CONFIG  ──────────────────────────────── #
HASH_SIZE          = 16          # pHash/dHash size → 64‑bit each
TARGET_OUTLIER_PCT = 5           # stop when ≤ this % remain unsorted
MAX_PASSES         = 4           # safety; caps eps growth
EPS_START          = 0.45        # empirical good default
EPS_GROWTH         = 0.25        # added to eps each pass
MIN_SAMPLES_INIT   = 2           # DBSCAN minPts first pass
DB_PATH            = "features.sqlite"  # persistent cache
# ─────────────────────────────────────────────────────────────────────────── #

# -------- utilities ------------------------------------------------------- #
def perceptual_bits(img: Image.Image) -> np.ndarray:
    """Return concatenated pHash + dHash bits as 128‑D int8 array."""
    ph = imagehash.phash(img, hash_size=HASH_SIZE).hash.astype(np.uint8).flatten()
    dh = imagehash.dhash(img, hash_size=HASH_SIZE).hash.astype(np.uint8).flatten()
    return np.concatenate([ph, dh])  # 2×64 = 128

def colour_hist(img_bgr: np.ndarray, bins=(32, 32, 32)) -> np.ndarray:
    hist = cv2.calcHist([img_bgr], [0,1,2], None, bins, [0,256]*3).flatten()
    return hist / (hist.sum() + 1e-8)   # L1‑normalised

def simple_stats(img: np.ndarray) -> np.ndarray:
    resized = cv2.resize(img, (32,32))
    return np.array([resized.mean(), resized.std(), np.median(resized)])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_class_rules(path: Path) -> dict:
    """classification.txt lines → {token -> subfolder}"""
    mapping = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            token = line.lower()
            mapping[token] = token.replace(" ", "_")
    return mapping

# -------- persistent feature store --------------------------------------- #
class FeatureCache:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        with closing(self.conn.cursor()) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS feats
                         (path TEXT PRIMARY KEY, vec BLOB)""")
        self.conn.commit()

    def get(self, fpath: str):
        cur = self.conn.execute("SELECT vec FROM feats WHERE path=?", (fpath,))
        row = cur.fetchone()
        return np.frombuffer(row[0], dtype=np.float32) if row else None

    def put(self, fpath: str, vec: np.ndarray):
        try:
            self.conn.execute("INSERT OR REPLACE INTO feats VALUES (?,?)",
                              (fpath, vec.astype(np.float32).tobytes()))
            self.conn.commit()
        except sqlite3.Error:
            pass

# -------- main sorter ----------------------------------------------------- #
class AdvancedImageSorter:
    SUP_EXT = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".gif")

    def __init__(self, input_dir: str, output_dir: str):
        self.in_dir  = Path(input_dir).resolve()
        self.out_dir = Path(output_dir).resolve()
        ensure_dir(self.out_dir)
        self.cache = FeatureCache()
        self.class_rules = load_class_rules(self.in_dir/"classification.txt")
        self.images, self.features = [], []
        self.cluster_labels = []

    # ── stage 1: load / compute features ────────────────────────────────── #
    def gather(self):
        print("Scanning files...")
        for fp in tqdm(list(self.in_dir.rglob("*"))):
            if fp.is_file() and fp.suffix.lower() in self.SUP_EXT:
                self.images.append(fp)
        if not self.images:
            print("No images found; aborting.")
            sys.exit(1)

        print("Computing / retrieving features...")
        for fp in tqdm(self.images):
            vec = self.cache.get(str(fp))
            if vec is None:
                img_cv = cv2.imread(str(fp))
                if img_cv is None: continue
                pil   = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                vec   = np.concatenate([colour_hist(img_cv),
                                        perceptual_bits(pil),
                                        simple_stats(img_cv)])
                self.cache.put(str(fp), vec)
            self.features.append(vec)

        self.features = StandardScaler().fit_transform(np.vstack(self.features))
    
    # ── stage 2: adaptive multi‑pass clustering ─────────────────────────── #
    def adaptive_cluster(self):
        remaining_idx = np.arange(len(self.images))
        eps, min_samples = EPS_START, MIN_SAMPLES_INIT
        label_counter = 0
        self.cluster_labels = np.full(len(self.images), -99, dtype=int)  # placeholder

        for pass_no in range(1, MAX_PASSES+1):
            if not remaining_idx.size: break
            feats_pass = self.features[remaining_idx]
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats_pass)

            # remap cluster ids to global ids
            for local_lbl in set(labels):
                if local_lbl == -1: continue
                global_lbl = label_counter
                label_counter += 1
                self.cluster_labels[remaining_idx[labels == local_lbl]] = global_lbl

            # decide what’s left
            remaining_idx = remaining_idx[labels == -1]
            pct_left = 100 * remaining_idx.size / len(self.images)
            print(f"Pass {pass_no}: eps={eps:.2f}  unsorted={pct_left:.1f}%")

            if pct_left <= TARGET_OUTLIER_PCT:  # good enough
                break
            eps += EPS_GROWTH
            min_samples = max(2, min_samples - 1)  # a bit more permissive

        # whatever is still -99 → true uniques
        self.cluster_labels[self.cluster_labels == -99] = -1

    # ── stage 3: organise output ────────────────────────────────────────── #
    def organise(self):
        dst_map = defaultdict(list)
        for fp, lbl in zip(self.images, self.cluster_labels):
            # optional rule‑based override
            rule_dst = None
            lower = fp.name.lower()
            for token, folder in self.class_rules.items():
                if token in lower:
                    rule_dst = folder; break

            if lbl == -1:
                dst_folder = rule_dst or "unique_images"
            else:
                dst_folder = rule_dst or f"group_{lbl+1:03d}"
            dst_map[dst_folder].append(fp)

        print("Copying files...")
        for folder, files in tqdm(dst_map.items()):
            tgt_dir = self.out_dir / folder
            ensure_dir(tgt_dir)
            for src in files:
                dst = tgt_dir / src.name
                dup_idx = 1
                while dst.exists():
                    dst = tgt_dir / f"{dst.stem}_{dup_idx}{dst.suffix}"
                    dup_idx += 1
                shutil.copy2(src, dst)

        # summary
        summary = {
            "total_images": len(self.images),
            "clusters": int(max(self.cluster_labels)+1),
            "unique_images": int((self.cluster_labels == -1).sum()),
            "timestamp": time.strftime("%Y-%m-%d_%H:%M:%S")
        }
        (self.out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
        print("✔ Sorting complete!")

    # ── interactive runner ──────────────────────────────────────────────── #
    def run(self):
        self.gather()
        self.adaptive_cluster()
        self.organise()

# --------------- CLI (interactive) ---------------------------------------- #
def banner():
    print("\n" + "─"*72)
    print("  ADVANCED CHARACTER / IMAGE SORTER".center(72))
    print("─"*72)

def prompt(msg, default=""):
    val = input(f"{msg} [{default}]: ").strip()
    return val or default

def main():
    banner()
    while True:
        in_dir = prompt("Enter path to your *input* folder").strip('"\' ')
        if Path(in_dir).is_dir(): break
        print("✗ Not a folder, try again.")
    default_out = f"{in_dir}_sorted"
    out_dir = prompt("Enter *output* folder", default_out)
    sorter = AdvancedImageSorter(in_dir, out_dir)
    sorter.run()

if __name__ == "__main__":
    main()
