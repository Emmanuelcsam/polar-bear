#!/usr/bin/env python3
"""
IMAGE SORTER
---------------------------------
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
HASH_SIZE          = 16          # Size for perceptual hashes (16x16 grid = 256 bits per hash)
TARGET_OUTLIER_PCT = 5           # Stop clustering when only this percentage remains unclustered
MAX_PASSES         = 4           # Maximum clustering iterations to prevent infinite loops
EPS_START          = 0.45        # Initial distance threshold for DBSCAN clustering
EPS_GROWTH         = 0.25        # Amount to increase eps each pass to capture more distant images
MIN_SAMPLES_INIT   = 2           # Minimum cluster size for first clustering pass
DB_PATH            = "features.sqlite"  # SQLite database file to cache computed features
# ─────────────────────────────────────────────────────────────────────────── #

# -------- utilities ------------------------------------------------------- #
def perceptual_bits(img: Image.Image) -> np.ndarray:
    """Return concatenated pHash + dHash bits as 128‑D int8 array."""
    # Compute perceptual hash and difference hash for image similarity detection
    ph = imagehash.phash(img, hash_size=HASH_SIZE).hash.astype(np.uint8).flatten()  # Create perceptual hash and convert to flat uint8 array
    dh = imagehash.dhash(img, hash_size=HASH_SIZE).hash.astype(np.uint8).flatten()  # Create difference hash and convert to flat uint8 array
    return np.concatenate([ph, dh])  # Combine both hash arrays into single 128-element feature vector

def colour_hist(img_bgr: np.ndarray, bins=(32, 32, 32)) -> np.ndarray:
    # Compute normalized RGB color histogram as feature vector
    hist = cv2.calcHist([img_bgr], [0,1,2], None, bins, [0,256]*3).flatten()  # Calculate 3D color histogram with specified bins per channel
    return hist / (hist.sum() + 1e-8)   # Normalize histogram values to sum to 1 (L1 normalization)

def simple_stats(img: np.ndarray) -> np.ndarray:
    # Extract basic statistical features (mean, std, median) from resized image
    resized = cv2.resize(img, (32,32))  # Resize image to 32x32 for consistent statistics
    return np.array([resized.mean(), resized.std(), np.median(resized)])  # Return mean, standard deviation, and median as feature vector

def ensure_dir(p: Path):
    # Create directory and all parent directories if they don't exist
    p.mkdir(parents=True, exist_ok=True)  # Create directory path recursively, ignore if exists

def load_class_rules(path: Path) -> dict:
    """classification.txt lines → {token -> subfolder}"""
    mapping = {}  # Initialize empty dictionary for classification rules
    if path.is_file():  # Check if classification file exists
        for line in path.read_text().splitlines():  # Read file and split into lines
            line = line.strip()  # Remove leading/trailing whitespace
            if not line or line.startswith("#"): continue  # Skip empty lines and comments
            token = line.lower()  # Convert to lowercase for case-insensitive matching
            mapping[token] = token.replace(" ", "_")  # Map token to folder name (spaces to underscores)
    return mapping  # Return dictionary of classification rules

# -------- persistent feature store --------------------------------------- #
class FeatureCache:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)  # Connect to SQLite database for feature caching
        with closing(self.conn.cursor()) as c:  # Create cursor with automatic cleanup
            c.execute("""CREATE TABLE IF NOT EXISTS feats
                         (path TEXT PRIMARY KEY, vec BLOB)""")  # Create table for storing file paths and feature vectors
        self.conn.commit()  # Save table creation to database

    def get(self, fpath: str):
        cur = self.conn.execute("SELECT vec FROM feats WHERE path=?", (fpath,))  # Query for existing feature vector
        row = cur.fetchone()  # Get first (and only) result
        return np.frombuffer(row[0], dtype=np.float32) if row else None  # Convert binary data to numpy array, or None if not found

    def put(self, fpath: str, vec: np.ndarray):
        try:
            self.conn.execute("INSERT OR REPLACE INTO feats VALUES (?,?)",
                              (fpath, vec.astype(np.float32).tobytes()))  # Store file path and feature vector as binary data
            self.conn.commit()  # Save changes to database
        except sqlite3.Error:
            pass  # Silently ignore database errors

# -------- main sorter ----------------------------------------------------- #
class AdvancedImageSorter:
    SUP_EXT = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".gif")  # Supported image file extensions

    def __init__(self, input_dir: str, output_dir: str):
        self.in_dir  = Path(input_dir).resolve()  # Convert input directory to absolute Path object
        self.out_dir = Path(output_dir).resolve()  # Convert output directory to absolute Path object
        ensure_dir(self.out_dir)  # Create output directory if it doesn't exist
        self.cache = FeatureCache()  # Initialize feature cache for storing computed features
        self.class_rules = load_class_rules(self.in_dir/"classification.txt")  # Load classification rules from file
        self.images, self.features = [], []  # Initialize empty lists for image paths and feature vectors
        self.cluster_labels = []  # Initialize empty list for cluster assignments

    # ── stage 1: load / compute features ────────────────────────────────── #
    def gather(self):
        print("Scanning files...")  # User feedback for file scanning phase
        for fp in tqdm(list(self.in_dir.rglob("*"))):  # Recursively find all files in input directory with progress bar
            if fp.is_file() and fp.suffix.lower() in self.SUP_EXT:  # Check if file is supported image type
                self.images.append(fp)  # Add image path to list
        if not self.images:  # Check if any images were found
            print("No images found; aborting.")  # Inform user no images found
            sys.exit(1)  # Exit program with error code

        print("Computing / retrieving features...")  # User feedback for feature computation phase
        for fp in tqdm(self.images):  # Process each image with progress bar
            vec = self.cache.get(str(fp))  # Try to get cached feature vector
            if vec is None:  # If not cached, compute features
                img_cv = cv2.imread(str(fp))  # Load image using OpenCV
                if img_cv is None: continue  # Skip if image couldn't be loaded
                pil   = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))  # Convert OpenCV BGR to PIL RGB format
                vec   = np.concatenate([colour_hist(img_cv),
                                        perceptual_bits(pil),
                                        simple_stats(img_cv)])  # Combine all feature types into single vector
                self.cache.put(str(fp), vec)  # Cache computed feature vector
            self.features.append(vec)  # Add feature vector to list

        self.features = StandardScaler().fit_transform(np.vstack(self.features))  # Standardize all features (zero mean, unit variance)
    
    # ── stage 2: adaptive multi‑pass clustering ─────────────────────────── #
    def adaptive_cluster(self):
        remaining_idx = np.arange(len(self.images))  # Start with all images as unclustered
        eps, min_samples = EPS_START, MIN_SAMPLES_INIT  # Initialize clustering parameters
        label_counter = 0  # Counter for assigning unique cluster IDs
        self.cluster_labels = np.full(len(self.images), -99, dtype=int)  # Initialize all labels as placeholder value

        for pass_no in range(1, MAX_PASSES+1):  # Iterate through clustering passes
            if not remaining_idx.size: break  # Stop if no images remain unclustered
            feats_pass = self.features[remaining_idx]  # Get features for remaining unclustered images
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats_pass)  # Perform DBSCAN clustering

            # remap cluster ids to global ids
            for local_lbl in set(labels):  # Iterate through unique cluster labels from this pass
                if local_lbl == -1: continue  # Skip noise points (label -1)
                global_lbl = label_counter  # Assign new global cluster ID
                label_counter += 1  # Increment counter for next cluster
                self.cluster_labels[remaining_idx[labels == local_lbl]] = global_lbl  # Assign global label to cluster members

            # decide what's left
            remaining_idx = remaining_idx[labels == -1]  # Keep only noise points for next pass
            pct_left = 100 * remaining_idx.size / len(self.images)  # Calculate percentage of unclustered images
            print(f"Pass {pass_no}: eps={eps:.2f}  unsorted={pct_left:.1f}%")  # Print progress information

            if pct_left <= TARGET_OUTLIER_PCT:  # Check if enough images are clustered
                break  # Stop clustering if target reached
            eps += EPS_GROWTH  # Increase distance threshold for next pass
            min_samples = max(2, min_samples - 1)  # Decrease minimum samples (more permissive clustering)

        # whatever is still -99 → true uniques
        self.cluster_labels[self.cluster_labels == -99] = -1  # Mark remaining unclustered images as unique

    # ── stage 3: organise output ────────────────────────────────────────── #
    def organise(self):
        dst_map = defaultdict(list)  # Dictionary to group images by destination folder
        for fp, lbl in zip(self.images, self.cluster_labels):  # Process each image and its cluster label
            # optional rule‑based override
            rule_dst = None  # Initialize rule-based destination as None
            lower = fp.name.lower()  # Get lowercase filename for rule matching
            for token, folder in self.class_rules.items():  # Check each classification rule
                if token in lower:  # If rule token found in filename
                    rule_dst = folder; break  # Set rule destination and stop checking

            if lbl == -1:  # If image is unclustered (unique)
                dst_folder = rule_dst or "unique_images"  # Use rule destination or default unique folder
            else:
                dst_folder = rule_dst or f"group_{lbl+1:03d}"  # Use rule destination or numbered cluster folder
            dst_map[dst_folder].append(fp)  # Add image to appropriate destination folder

        print("Copying files...")  # User feedback for file copying phase
        for folder, files in tqdm(dst_map.items()):  # Process each destination folder with progress bar
            tgt_dir = self.out_dir / folder  # Create target directory path
            ensure_dir(tgt_dir)  # Create target directory
            for src in files:  # Process each file in the folder
                dst = tgt_dir / src.name  # Create destination file path
                dup_idx = 1  # Initialize duplicate counter
                while dst.exists():  # Handle filename conflicts
                    dst = tgt_dir / f"{dst.stem}_{dup_idx}{dst.suffix}"  # Add number to filename
                    dup_idx += 1  # Increment duplicate counter
                shutil.copy2(src, dst)  # Copy file with metadata preservation

        # summary
        summary = {
            "total_images": len(self.images),  # Total number of processed images
            "clusters": int(max(self.cluster_labels)+1),  # Number of clusters created
            "unique_images": int((self.cluster_labels == -1).sum()),  # Number of unclustered images
            "timestamp": time.strftime("%Y-%m-%d_%H:%M:%S")  # Processing timestamp
        }
        (self.out_dir/"summary.json").write_text(json.dumps(summary, indent=2))  # Write summary to JSON file
        print("✔ Sorting complete!")  # Inform user that processing is complete

    # ── interactive runner ──────────────────────────────────────────────── #
    def run(self):
        self.gather()  # Execute stage 1: gather images and compute features
        self.adaptive_cluster()  # Execute stage 2: perform adaptive clustering
        self.organise()  # Execute stage 3: organize files and create output

# --------------- CLI (interactive) ---------------------------------------- #
def banner():
    print("\n" + "─"*72)  # Print decorative line separator
    print("  ADVANCED CHARACTER / IMAGE SORTER".center(72))  # Print centered title
    print("─"*72)  # Print decorative line separator

def prompt(msg, default=""):
    val = input(f"{msg} [{default}]: ").strip()  # Get user input with default value shown
    return val or default  # Return user input or default if empty

def main():
    banner()  # Display program banner
    while True:  # Loop until valid input directory provided
        in_dir = prompt("Enter path to your *input* folder").strip('"\' ')  # Get input directory, removing quotes
        if Path(in_dir).is_dir(): break  # Break loop if valid directory
        print("✗ Not a folder, try again.")  # Inform user of invalid input
    default_out = f"{in_dir}_sorted"  # Create default output directory name
    out_dir = prompt("Enter *output* folder", default_out)  # Get output directory with default
    sorter = AdvancedImageSorter(in_dir, out_dir)  # Create sorter instance
    sorter.run()  # Execute the sorting process

if __name__ == "__main__":
    main()  # Run main function if script executed directly
