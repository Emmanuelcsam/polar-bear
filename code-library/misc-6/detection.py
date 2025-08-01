#!/usr/bin/env python
# detect_defects.py
# Requirements: torch, torchvision, opencv-python, numpy, scikit-image

import os, json, glob, random, math
from pathlib import Path
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import sigmoid_focal_loss

##############################################################################
# 2.1  Dataset helpers
##############################################################################

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def is_image(fname): return fname.lower().endswith(IMG_EXT)

class EndFaceDataset(Dataset):
    """
    Walks through project-directory/dataset/**/.* to yield:
        image_tensor , region_label  (stage-1)
        └── region_label ∈ {core, cladding, ferrule, background}
    """
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.paths = [p for p in self.root.glob("**/*") if is_image(p.name)]
        self.xform = transform or transforms.Compose([
            transforms.ToTensor(),                       # HWC→CHW, [0,1]
            transforms.Normalize((0.5,0.5,0.5),
                                 (0.5,0.5,0.5))         # to (−1,1)
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img_p = self.paths[idx]
        img   = cv.imread(str(img_p), cv.IMREAD_COLOR)
        img   = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        timg  = self.xform(img)
        # Region label is inferred from chunk name or .json (placeholder: 0)
        label = 0
        return timg, label

##############################################################################
# 2.2  Neural blocks – clean, modular
##############################################################################

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, p=1, s=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class RegionClassifier(nn.Module):
    """
    Tiny backbone + 1×1 conv head → {core,cladding,ferrule,bg}
    """
    def __init__(self, c=3, num_regions=4):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(c, 32, 3), ConvBlock(32, 64, 3, s=2),
            ConvBlock(64,128,3, s=2), nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Conv2d(128, num_regions, 1)

    def forward(self, x):
        feats = self.backbone(x)          # N,C,1,1
        logits = self.head(feats).squeeze(-1).squeeze(-1)  # N,R
        return logits                     # no Softmax here

##############################################################################
# 2.3  Training Region Classifier
##############################################################################

def train_region_classifier():
    ds = EndFaceDataset("dataset")
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

    net  = RegionClassifier().cuda()
    opt  = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(15):
        net.train()
        tot_loss=0
        for img, y in dl:
            img,y = img.cuda(), y.cuda()
            logits = net(img)
            loss = nn.CrossEntropyLoss()(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()*img.size(0)
        print(f"Epoch {epoch:02d} | CE loss = {tot_loss/len(ds):.4f}")

    Path("model_ckpt").mkdir(exist_ok=True)
    torch.save(net.state_dict(), "model_ckpt/region_clf.pt")

##############################################################################
# 2.4  Defect detector – operate on regions
##############################################################################

class DefectDetector(nn.Module):
    """
    Simple UNet-like segmentation head → scratch/dirt/contamination map
    """
    def __init__(self, in_ch=3, classes=3):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64, s=2)
        self.enc3 = ConvBlock(64,128, s=2)
        self.mid  = ConvBlock(128,256)
        self.up2  = nn.ConvTranspose2d(256,128,2,2)
        self.dec2 = ConvBlock(256,128)
        self.up1  = nn.ConvTranspose2d(128,64,2,2)
        self.dec1 = ConvBlock(128,64)
        self.out  = nn.Conv2d(64, classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)         #   N,32,H,W
        e2 = self.enc2(e1)        #   N,64,H/2,W/2
        e3 = self.enc3(e2)        #   N,128,H/4,W/4
        m  = self.mid(e3)         #   N,256,H/4,W/4
        d2 = torch.cat((self.up2(m), e3), 1)
        d2 = self.dec2(d2)
        d1 = torch.cat((self.up1(d2), e2), 1)
        d1 = self.dec1(d1)
        return self.out(d1)       # logits N,C,H/2,W/2

##############################################################################
# 2.5  OpenCV pre-processing utilities
##############################################################################

def find_regions_opencv(rgb):
    """
    Locate core, cladding, ferrule using morphology & HoughCircles. Returns masks.
    """
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    gray = cv.medianBlur(gray, 5)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                              param1=120, param2=30,
                              minRadius=30, maxRadius=500)
    # parse circles→ masks
    masks = {}
    if circles is not None:
        circles = np.uint16(np.around(circles[0,:]))
        # first circle ~ core, second ~ cladding (empirical)
        if len(circles) >= 2:
            cx,cy,r1 = circles[0]
            cx,cy,r2 = circles[1]
            h,w = gray.shape
            Y,X = np.ogrid[:h,:w]
            core_mask      = (X-cx)**2 + (Y-cy)**2 <= r1**2
            cladding_mask  = (X-cx)**2 + (Y-cy)**2 <= r2**2
            ferrule_mask   = ~cladding_mask
            masks = dict(core=core_mask, cladding=cladding_mask,
                         ferrule=ferrule_mask)
    return masks

##############################################################################
# 2.6  Main – demo inference on one image
##############################################################################

@torch.no_grad()
def infer_on_image(img_path):
    # Load networks
    reg_net = RegionClassifier(); reg_net.load_state_dict(
        torch.load("model_ckpt/region_clf.pt", map_location="cpu"))
    reg_net.eval()

    det_net = DefectDetector(); det_net.eval()   # assume pretrained

    # Read & tensor-ize
    raw = cv.imread(img_path, cv.IMREAD_COLOR)
    rgb = cv.cvtColor(raw, cv.COLOR_BGR2RGB)
    t   = transforms.ToTensor()(rgb).unsqueeze(0)

    # Region prediction
    reg_logits = reg_net(t)
    reg_prob   = torch.softmax(reg_logits, -1).squeeze()

    # OpenCV geometry segmentation
    masks = find_regions_opencv(rgb)

    # Defect prediction (per-region crop)
    preds = {}
    for region,mask in masks.items():
        roi = rgb.copy(); roi[~mask]=0
        troi = transforms.ToTensor()(roi).unsqueeze(0)
        out  = det_net(troi)              # logits
        prob = torch.sigmoid(out)         # N,C,H/2,W/2
        preds[region] = prob.squeeze(0).cpu().numpy()

    # Aggregate statistics
    stats = {}
    for reg,p in preds.items():
        stats[reg] = { 'scratch': float(p[0].mean()),
                       'dirt'   : float(p[1].mean()),
                       'contam' : float(p[2].mean()) }

    print(json.dumps(stats, indent=2))

##############################################################################
if __name__ == "__main__":
    # 1) train_region_classifier()   # first-time only
    # 2) then run inference
    infer_on_image("dataset/chunk_1/sample.jpg")
