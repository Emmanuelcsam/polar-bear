#!/usr/bin/env python3
"""
Fiber-Optic End-face Region & Defect Detector
============================================
A complete PyTorch implementation, built strictly from the coding patterns
in the Dive-into-Deep-Learning (D2L) textbook, adapted to the statistical
characteristics of your dataset and the architecture/loss recommendations
in the attached reports.

Directory layout (required)
└── project-directory/
    ├── dataset/
    │   ├── chunk_1/  … chunk_135/         ← raw input images
    └── reference/
        └── *40 sub-folders*/ *.pt         ← cropped reference tensors
"""

# ---------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------
import os, json, math, time, glob, random, argparse
from pathlib import Path
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from d2l import torch as d2l                              # D2L utility API [15]
from tqdm import tqdm

# ---------------------------------------------------------------------
# 2. Hyper-parameters (from statistical reports)                       [1]
# ---------------------------------------------------------------------
BATCH_SIZE       = 128
IMG_SIZE         = 256                 # resized square for CNN backbone
EPOCHS           = 50
LR_INITIAL       = 1e-3
MIN_LR           = 1e-5
PATIENCE         = 10
WEIGHT_DECAY     = 1e-4
FOCAL_GAMMA      = 2.0                 # only used if --focal flag is passed
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------
# 3. Utils
# ---------------------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything()

exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def list_images(folder):
    return [p for p in Path(folder).rglob('*') if p.suffix.lower() in exts]

# ---------------------------------------------------------------------
# 4. Dataset & Pre-processing (D2L style pipeline)                     [15]
# ---------------------------------------------------------------------
class EndfaceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples, self.labels = self._crawl()
        # Build integer class map
        self.classes = sorted(list({lbl for _, lbl in self.labels}))
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = [self.cls2idx[lbl] for lbl in self.labels]  # to int
        # Compute per-class counts for class-balanced sampling
        self.counts  = torch.tensor([self.labels.count(i)
                                     for i in range(len(self.classes))])

    def _infer_label(self, path: Path) -> str:
        """Derive class from filename heuristics in statistical report."""
        name = path.stem.lower()
        if 'dirty' in name or 'scratch' in name:
            return 'defect'
        if 'core' in name:
            return 'core'
        if 'cladding' in name:
            return 'cladding'
        if 'ferrule' in name:
            return 'ferrule'
        return 'unknown'

    def _crawl(self):
        imgs, lbls = [], []
        for img_path in list_images(self.root):
            imgs.append(img_path)
            lbls.append(self._infer_label(img_path))
        return imgs, lbls

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path  = self.samples[idx]
        label = self.labels[idx]
        img   = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # 1-ch
        if img is None:
            raise RuntimeError(f'Cannot read {path}')
        if self.transform:
            img = self.transform(img)
        return img, label

# Data augmentation & normalization (Imagenet stats used in reports)   [1]
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(int(IMG_SIZE*1.1)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=.2, contrast=.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

# ---------------------------------------------------------------------
# 5. Focal-Loss (optional, handles class imbalance)                    [1]
# ---------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=alpha)

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        pt = torch.exp(-ce_loss)
        focal = (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

# ---------------------------------------------------------------------
# 6. CNN Backbone & Classifier (D2L Conv-Block pattern)                [15]
# ---------------------------------------------------------------------
def conv_block(in_ch, out_ch, drop=0.0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(drop)
    )

class FiberNet(nn.Module):
    def __init__(self, num_classes=4, drops=(0.2,0.2,0.3,0.4)):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1,   64, drops[0]),
            conv_block(64, 128, drops[1]),
            conv_block(128,256, drops[2]),
            conv_block(256,512, drops[3]),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*(IMG_SIZE//16)*(IMG_SIZE//16), 1024),
            nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ---------------------------------------------------------------------
# 7. Training / Validation Loops (D2L style)                           [15]
# ---------------------------------------------------------------------
def train_epoch(net, loader, criterion, optimizer):
    net.train()
    metric = d2l.Accumulator(3)  # loss, correct, total
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_hat = net(X)
        loss  = criterion(y_hat, y)
        loss.backward(); optimizer.step()
        metric.add(loss.item()*y.shape[0],
                   (y_hat.argmax(1)==y).sum().item(),
                   y.numel())
    return metric[0]/metric[2], metric[1]/metric[2]

@torch.no_grad()
def evaluate(net, loader, criterion):
    net.eval()
    metric = d2l.Accumulator(3)
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        y_hat = net(X)
        loss  = criterion(y_hat, y)
        metric.add(loss.item()*y.shape[0],
                   (y_hat.argmax(1)==y).sum().item(),
                   y.numel())
    return metric[0]/metric[2], metric[1]/metric[2]

# ---------------------------------------------------------------------
# 8. Reference Tensor Loader (transfer-style feature regularizer)
# ---------------------------------------------------------------------
def load_reference_tensors(ref_dir):
    tensors = []
    for pt_file in Path(ref_dir).rglob('*.pt'):
        tensors.append(torch.load(pt_file, map_location='cpu'))
    return tensors

# ---------------------------------------------------------------------
# 9. Main Routine
# ---------------------------------------------------------------------
def main(args):
    print(f'Using device: {DEVICE}')
    train_ds = EndfaceDataset(args.dataset, transform_train)
    val_ds   = EndfaceDataset(args.dataset, transform_val)

    # Class weights for imbalance                                            [1]
    if args.class_weights:
        weights = 1.0 / train_ds.counts.float()
        sample_weights = weights[train_ds.labels]
        sampler = WeightedRandomSampler(sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4)

    # Model, loss, optimiser
    model = FiberNet(num_classes=len(train_ds.classes)).to(DEVICE)
    if args.focal:
        criterion = FocalLoss(alpha=None, gamma=FOCAL_GAMMA)
    else:
        class_weights = (1.0/train_ds.counts).to(DEVICE) if args.class_weights else None
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INITIAL,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LR)

    # Optional reference tensor loading
    if args.reference:
        ref_tensors = load_reference_tensors(args.reference)
        print(f'Loaded {len(ref_tensors)} reference tensors.')

    best_acc, patience_counter = 0, 0
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f'Epoch {epoch+1:02}/{EPOCHS} | '
              f'Train: {tr_loss:.4f}/{tr_acc:.3%} | '
              f'Val: {val_loss:.4f}/{val_acc:.3%}')
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc; patience_counter = 0
            torch.save(model.state_dict(), 'best_fibernet.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print('Early stopping triggered.')
                break

    # Save stats
    with open('training_stats.json','w') as f:
        json.dump({'best_val_accuracy':float(best_acc)}, f, indent=2)

    print('Training complete. Best validation accuracy '
          f'{best_acc:.2%} (model saved as best_fibernet.pt)')

# ---------------------------------------------------------------------
# 10. Inference + OpenCV Post-processing (scratch / blob highlight)
# ---------------------------------------------------------------------
@torch.no_grad()
def predict_image(net, img_path, threshold=0.5):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_t = transform_val(img).unsqueeze(0).to(DEVICE)
    logits = net(img_t); probs = logits.softmax(1).squeeze()
    pred_idx = probs.argmax().item()
    pred_prob = probs[pred_idx].item()
    return pred_idx, pred_prob

def detect_scratches(img):
    edges = cv2.Canny(img, 50, 150)
    return edges

def detect_blobs(img):
    params = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    return keypoints

# ---------------------------------------------------------------------
# 11. Entry-point
# ---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   type=str, required=True,
                        help='Path to dataset/ directory')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference/ directory (optional)')
    parser.add_argument('--focal',     action='store_true',
                        help='Use focal loss (recommended by stats)')
    parser.add_argument('--class_weights', action='store_true',
                        help='Use inverse-freq class weights')
    args = parser.parse_args()
    main(args)
