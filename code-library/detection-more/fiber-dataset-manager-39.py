"""
PyTorch Dataset Builder and CNN Trainer for Fiber Optic Inspection
Converts existing pipeline outputs into ML-ready datasets and trains models
"""

from pathlib import Path
import shutil, json, csv, itertools, sqlite3, hashlib, logging
import numpy as np
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dataset_builder")

def _init_sqlite(db_path: Path):
    """Create or connect DB – returns connection."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # images table
    c.execute("""
        CREATE TABLE IF NOT EXISTS images(
            image_id TEXT PRIMARY KEY,
            filename TEXT,
            centre_x REAL, centre_y REAL,
            core_r REAL, clad_r REAL, ferrule_r REAL,
            has_defect INTEGER,  -- 1 yes, 0 no
            split TEXT           -- train / val / test
        );""")
    # region stats
    c.execute("""
        CREATE TABLE IF NOT EXISTS region_intensity(
            image_id TEXT,
            region TEXT,              -- core / cladding / ferrule
            mean REAL, std REAL, min REAL, max REAL, p25 REAL, p50 REAL, p75 REAL,
            FOREIGN KEY(image_id) REFERENCES images(image_id)
        );""")
    # defects
    c.execute("""
        CREATE TABLE IF NOT EXISTS defects(
            image_id TEXT,
            defect_id TEXT,
            dtype TEXT, severity TEXT,
            x INTEGER, y INTEGER, area_px INTEGER,
            confidence REAL,
            FOREIGN KEY(image_id) REFERENCES images(image_id)
        );""")
    # correlations (sparse upper‑triangular)
    c.execute("""
        CREATE TABLE IF NOT EXISTS correlations(
            image_id_a TEXT,
            image_id_b TEXT,
            pearson REAL,
            spearman REAL
        );""")
    conn.commit()
    return conn

def extract_region_intensity_stats(gray_img: np.ndarray,
                                   masks: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    stats = {}
    for region, mask in masks.items():
        pixels = gray_img[mask > 0]
        if pixels.size == 0:
            continue
        stats[region] = {
            "mean": float(pixels.mean()),
            "std":  float(pixels.std()),
            "min":  float(pixels.min()),
            "max":  float(pixels.max()),
            "p25":  float(np.percentile(pixels, 25)),
            "p50":  float(np.percentile(pixels, 50)),
            "p75":  float(np.percentile(pixels, 75)),
        }
    return stats

def extract_pixel_trend_metrics(gray_img: np.ndarray,
                                masks: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    # gradient magnitude & Laplacian inside each region
    grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    lap = cv2.Laplacian(gray_img, cv2.CV_32F)

    trends = {}
    for region, mask in masks.items():
        idx = mask > 0
        if not idx.any():
            continue
        trends[region] = {
            "grad_mean":  float(grad_mag[idx].mean()),
            "grad_std":   float(grad_mag[idx].std()),
            "lap_mean":   float(lap[idx].mean()),
            "lap_std":    float(lap[idx].std()),
            "pixel_count": int(idx.sum()),
        }
    return trends

def compute_pairwise_correlations(image_paths: List) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # grayscale load then flatten
    imgs = []
    ids = []
    for p in image_paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        imgs.append(g.flatten().astype(np.float32))
        ids.append(p.stem)
    n = len(imgs)
    pearson = np.eye(n)
    spearman = np.eye(n)
    for (i, x), (j, y) in itertools.combinations(enumerate(imgs), 2):
        r, _ = pearsonr(x, y)
        s, _ = spearmanr(x, y)
        pearson[i, j] = pearson[j, i] = r
        spearman[i, j] = spearman[j, i] = s
    return pearson, spearman, ids

def build_dataset(raw_folder: str,
                  output_root: str = "dataset",
                  config_path: str = "config/config.json",
                  train_fraction: float = 0.7,
                  val_fraction: float = 0.15):
    """
    End‑to‑end routine:
    1. Run existing pipeline on every image, saving masks/defect jsons.
    2. Extract numeric features & store in SQLite + CSVs.
    3. Compute global correlations (optionally heavy – can toggle).
    4. Copy (or symlink) all imagery into dataset/images.
    """
    
    # Note: This assumes you have the original pipeline modules available
    # For demo purposes, we'll simulate the pipeline outputs
    
    raw_folder = Path(raw_folder)
    output_root = Path(output_root)
    out_img = output_root / "images"
    out_masks = output_root / "masks"
    out_defects = output_root / "defects"
    out_features = output_root / "features"
    out_db = output_root / "index.sqlite"

    for d in (out_img, out_masks, out_defects, out_features):
        d.mkdir(parents=True, exist_ok=True)
    conn = _init_sqlite(out_db)
    cursor = conn.cursor()

    # pass 1 – process every image and fill core tables
    processed_ids = []
    for img_path in tqdm(sorted(raw_folder.glob("*.*")), desc="pipeline"):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue

        # Simulate pipeline outputs (in real implementation, call actual pipeline)
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        h, w = image.shape[:2]
        
        # Simulate zone masks
        center = (w//2, h//2)
        core_r = min(h, w) // 10
        clad_r = min(h, w) // 4
        
        masks = {
            'core': np.zeros((h, w), dtype=np.uint8),
            'cladding': np.zeros((h, w), dtype=np.uint8),
            'ferrule': np.zeros((h, w), dtype=np.uint8)
        }
        cv2.circle(masks['core'], center, core_r, 255, -1)
        cv2.circle(masks['cladding'], center, clad_r, 255, -1)
        cv2.circle(masks['cladding'], center, core_r, 0, -1)
        masks['ferrule'] = 255 - masks['core'] - masks['cladding']

        image_id = hashlib.md5(img_path.name.encode()).hexdigest()[:16]
        
        # Simulate defects (random for demo)
        has_defect = np.random.random() > 0.5
        num_defects = np.random.randint(0, 5) if has_defect else 0
        
        cursor.execute("""INSERT OR REPLACE INTO images VALUES
            (?,?,?,?,?,?,?,?,?);""",
            (image_id, img_path.name, center[0], center[1],
             core_r, clad_r, clad_r*1.5, int(has_defect), "TBD"))
        processed_ids.append((image_id, img_path))

        # Save artifacts
        shutil.copy2(img_path, out_img / img_path.name)
        for name, mask in masks.items():
            np.save(out_masks / f"{image_id}_{name}.npy", mask)
        
        # Simulate defects
        if has_defect:
            defects = []
            for i in range(num_defects):
                defects.append({
                    "unique_id": f"DEF_{i:03d}",
                    "type": np.random.choice(["SCRATCH", "PIT", "CONTAMINATION"]),
                    "severity": np.random.choice(["LOW", "MEDIUM", "HIGH"]),
                    "location": {"x": np.random.randint(0, w), "y": np.random.randint(0, h)},
                    "properties": {
                        "area_px": np.random.randint(10, 100),
                        "confidence": np.random.random()
                    }
                })
            
            with open(out_defects / f"{image_id}_defects.json", 'w') as f:
                json.dump({"defects": defects}, f)
            
            for d in defects:
                cursor.execute("""INSERT INTO defects VALUES
                    (?,?,?,?,?,?,?,?);""",
                    (image_id, d["unique_id"], d["type"], d["severity"],
                     d["location"]["x"], d["location"]["y"], d["properties"]["area_px"],
                     d["properties"]["confidence"]))
    
    conn.commit()

    # pass 2 – compute intensity stats & pixel trends
    for image_id, img_path in tqdm(processed_ids, desc="features"):
        # load masks
        masks = {}
        for reg in ["core", "cladding", "ferrule"]:
            mask_file = out_masks / f"{image_id}_{reg}.npy"
            if mask_file.exists():
                masks[reg] = np.load(mask_file)
        if not masks:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # intensity per region
        intens = extract_region_intensity_stats(img, masks)
        for reg, stats in intens.items():
            cursor.execute("""INSERT INTO region_intensity VALUES
                (?,?,?,?,?,?,?,?,?);""",
                (image_id, reg, stats["mean"], stats["std"], stats["min"],
                 stats["max"], stats["p25"], stats["p50"], stats["p75"]))

        # pixel trend metrics
        trends = extract_pixel_trend_metrics(img, masks)
        trend_file = out_features / f"{image_id}_trends.json"
        with open(trend_file, 'w') as f:
            json.dump(trends, f)

    conn.commit()

    # pass 3 – split assignment
    np.random.shuffle(processed_ids)
    n = len(processed_ids)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    for idx, (image_id, _) in enumerate(processed_ids):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"
        cursor.execute("UPDATE images SET split=? WHERE image_id=?;", (split, image_id))
    conn.commit()
    conn.close()

    log.info(f"Dataset build COMPLETE – {n} images processed → {output_root}")

# PyTorch Dataset and Model
class FiberDataset(Dataset):
    """
    Simple image‑level dataset (classification). For segmentation swap to mask‑returning DataSet.
    """
    def __init__(self, split: str, db_path: str, images_dir: str, aug: bool = True):
        self.conn = sqlite3.connect(db_path)
        cur = self.conn.cursor()
        cur.execute("SELECT image_id, filename, has_defect FROM images WHERE split=?;", (split,))
        self.rows = cur.fetchall()
        self.images_dir = Path(images_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.aug = aug

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image_id, fname, label = self.rows[idx]
        img = cv2.imread(str(self.images_dir / fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(fname)
        if self.aug and np.random.random() < 0.5:
            img = cv2.flip(img, 1)
        img = cv2.resize(img, (256, 256))
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def train_cnn(dataset_root="dataset",
              db_file="dataset/index.sqlite",
              epochs=20,
              batch_sz=32,
              lr=1e-3,
              model_out="models/baseline_cnn.pt"):

    train_ds = FiberDataset("train", db_file, f"{dataset_root}/images")
    val_ds   = FiberDataset("val", db_file, f"{dataset_root}/images", aug=False)
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_sz)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SimpleCNN().to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    Path(model_out).parent.mkdir(exist_ok=True)
    best_val = 0.0
    
    for ep in range(1, epochs + 1):
        # Training
        net.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        # Validation
        net.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                pred = outputs.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        log.info(f"Epoch {ep:02d}/{epochs} – train_loss: {train_loss/len(train_loader):.4f}, "
                f"val_loss: {val_loss/len(val_loader):.4f}, val_acc: {acc:.3f}")
        
        if acc > best_val:
            torch.save(net.state_dict(), model_out)
            best_val = acc
    
    log.info(f"Training complete – best val acc {best_val:.3f} -> {model_out}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build dataset")
    parser.add_argument("--train", action="store_true", help="Train CNN")
    parser.add_argument("--raw", default="raw_images", help="Raw images folder")
    parser.add_argument("--dataset", default="dataset", help="Dataset output folder")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    
    args = parser.parse_args()
    
    if args.build:
        build_dataset(args.raw, args.dataset)
    
    if args.train:
        train_cnn(
            dataset_root=args.dataset,
            db_file=f"{args.dataset}/index.sqlite",
            epochs=args.epochs
        )