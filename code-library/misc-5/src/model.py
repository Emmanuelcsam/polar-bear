# model.py
# Unified CNN for Fiber Optic Endface Region Segmentation and Defect Classification
# Architecture: Deep custom ConvNet (ResNet-inspired), Hybrid dual-head, and statistical prior fusion

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Core Encoder: ResNet34-like (no pretraining) ---
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def make_layer(block, in_planes, planes, num_blocks, stride, norm_layer):
    downsample = None
    if stride != 1 or in_planes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, stride, bias=False),
            norm_layer(planes)
        )
    layers = []
    layers.append(block(in_planes, planes, stride, downsample, norm_layer))
    for _ in range(1, num_blocks):
        layers.append(block(planes, planes, 1, None, norm_layer))
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # Input: 3xH*Wx
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # Layer configuration: [64,128,256,512] channels, as in ResNet34
        self.layer1 = make_layer(BasicBlock, 64,   64,  3, 1, norm_layer)
        self.layer2 = make_layer(BasicBlock, 64,  128,  4, 2, norm_layer)
        self.layer3 = make_layer(BasicBlock,128,  256,  6, 2, norm_layer)
        self.layer4 = make_layer(BasicBlock,256,  512,  3, 2, norm_layer)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

# --- UNet-style Decoder for Region Segmentation ---
class Decoder(nn.Module):
    def __init__(self, num_classes=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # Upsample blocks: concatenate encoder skip features
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1), norm_layer(256), nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1), norm_layer(128), nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), norm_layer(64), nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, feats):
        f1, f2, f3, f4 = feats
        x = self.up3(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.dec3(x)
        x = self.up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, f1], dim=1)
        x = self.dec1(x)
        mask_logits = self.final_conv(x)  # [B,3,H,W]
        return mask_logits

# --- Global Region Classification (Defect Head) ---
class DefectHead(nn.Module):
    """
    Fuses encoder spatial features and global-pool for multi-class defect classification.
    """
    def __init__(self, num_classes=40):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        vec = self.gap(x).flatten(1) # [B,512]
        logits = self.fc(vec)        # [B,num_classes]
        return logits

# --- Feature Extraction/Bottleneck for Statistical Output (PCA, Mahalanobis, etc.) ---
class StatHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=88):  # 88-dimensional statistical feature vector
        super().__init__()
        # Mimics principal components / statistical mapping
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        vec = self.gap(x).flatten(1)  # [B,512]
        stat_feats = self.proj(vec)   # [B,88]
        return stat_feats

# --- Complete EndfaceNet Model ---
class EndfaceNet(nn.Module):
    """
    Returns:
      - mask_logits: [B,3,H,W], softmax for region segmentation
      - defect_logits: [B,num_classes], multi-class defect
      - stat_feats: [B,88], domain statistical features for Mahalanobis/PCA loss tie-in
    """
    def __init__(self, num_classes=40):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=3)  # core, cladding, ferrule
        self.defect_head = DefectHead(num_classes)
        self.stat_head = StatHead(512, 88)

    def forward(self, x):
        feats = self.encoder(x)
        mask_logits = self.decoder(feats)
        defect_logits = self.defect_head(feats[-1])
        stat_feats = self.stat_head(feats[-1])
        return mask_logits, defect_logits, stat_feats

# --- Composite Loss for training (Dice, Focal, Mahalanobis/PCA prior) ---
class CompositeLoss(nn.Module):
    def __init__(self, prior_stats=None, class_weights=None, dice_weight=1.0, focal_weight=1.0, stat_weight=1.0, gamma=2.0, alpha=0.25):
        """
        prior_stats: dict with 'mu', 'inv_cov' for Mahalanobis; optional
        class_weights: tensor or None for focal loss (class imbalance)
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.stat_weight = stat_weight
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("mu", None)
        self.register_buffer("inv_cov", None)
        if prior_stats is not None:
            self.mu = torch.tensor(prior_stats['mu'], dtype=torch.float32)
            self.inv_cov = torch.tensor(prior_stats['inv_cov'], dtype=torch.float32)

    def dice_loss(self, inputs, targets, smooth=1):
        # inputs: [B,3,H,W] logits, targets: same shape, floats in [0,1]
        inputs = torch.sigmoid(inputs)
        targets = targets
        intersect = (inputs * targets).sum(dim=(2,3))
        suma = (inputs + targets).sum(dim=(2,3))
        dice = (2 * intersect + smooth) / (suma + smooth)
        loss = 1 - dice
        return loss.mean()

    def focal_loss(self, logits, targets):
        # logits: [B,num_classes], targets: [B,num_classes] (one-hot)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.class_weights is not None:
            focal = focal * self.class_weights[None, :]
        return focal.mean()

    def mahalanobis_loss(self, feats):
        # feats: [B,88]
        if self.mu is None or self.inv_cov is None:
            return feats.new_zeros(1)
        diffs = feats - self.mu
        dist = torch.sum(diffs @ self.inv_cov * diffs, dim=1)
        return dist.mean()

    def forward(self, pred_masks, tgt_masks, pred_logits, tgt_labels, stat_feats, ref_stats=None):
        l_dice = self.dice_loss(pred_masks, tgt_masks)
        l_focal = self.focal_loss(pred_logits, tgt_labels)
        l_stat = self.mahalanobis_loss(stat_feats)
        return self.dice_weight * l_dice + self.focal_weight * l_focal + self.stat_weight * l_stat

# ---------- END OF FILE -------------- 