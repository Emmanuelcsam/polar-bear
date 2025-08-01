# -*- coding: utf-8 -*-
"""
fiber_inspector.py
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import glob
import argparse
import time
from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# SECTION 1: UTILITY FUNCTIONS (Inspired by d2l.ai utilities)
# ==============================================================================

def box_corner_to_center(boxes):
    """Convert corner-format bounding boxes to center-format.
    Reference: d2l.ai, Chapter 14.2 (Bounding Boxes) [10]
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)

def box_center_to_corner(boxes):
    """Convert center-format bounding boxes to corner-format.
    Reference: d2l.ai, Chapter 14.2 (Bounding Boxes) [10]
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)

def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of boxes.
    Reference: d2l.ai, Chapter 14.5 (Anchor Boxes) [11]
    """
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel.
    Reference: d2l.ai, Chapter 14.5 (Anchor Boxes) [11]
    """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device, dtype=torch.float32)
    ratio_tensor = torch.tensor(ratios, device=device, dtype=torch.float32)
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor),
                   size_tensor * torch.sqrt(ratio_tensor[1:]))) \
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor),
                   size_tensor / torch.sqrt(ratio_tensor[1:])))
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes.
    Reference: d2l.ai, Chapter 14.5 (Anchor Boxes) [11]
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    if num_gt_boxes == 0:
        return torch.full((num_anchors,), -1, dtype=torch.long, device=device)

    jaccard = box_iou(anchors, ground_truth)
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    
    # Force assign each gt box to an anchor
    for i in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard[:, i])
        anchors_bbox_map[max_idx] = i

    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets.
    Reference: d2l.ai, Chapter 14.5 (Anchor Boxes) [11]
    """
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes.
    Reference: d2l.ai, Chapter 14.5 (Anchor Boxes) [11]
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        # Filter out padding boxes
        valid_labels = label[label[:, 0]!= -1]

        anchors_bbox_map = assign_anchor_to_bbox(
            valid_labels[:, 1:], anchors, device)
        
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        
        indices_true = torch.nonzero(anchors_bbox_map >= 0).reshape(-1)
        if len(indices_true) > 0:
            bb_idx = anchors_bbox_map[indices_true]
            class_labels[indices_true] = valid_labels[bb_idx, 0].long() + 1
            assigned_bb[indices_true] = valid_labels[bb_idx, 1:]
        
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        
        batch_offset.append(offset)
        batch_mask.append(bbox_mask.view(-1))
        batch_class_labels.append(class_labels)
        
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes.
    Reference: d2l.ai, Chapter 14.6 (Multiscale Object Detection) [11]
    """
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.1):
    """Predict bounding boxes using non-maximum suppression.
    Reference: d2l.ai, Chapter 14.6 (Multiscale Object Detection) [11]
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[2], cls_probs.shape[1]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i]
        
        # Apply softmax to get probabilities
        cls_prob = torch.softmax(cls_prob, dim=1)
        
        # Inverse transform offsets to get predicted boxes
        anc = box_corner_to_center(anchors)
        pred_bbox_xy = (offset_pred[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
        pred_bbox_wh = torch.exp(offset_pred[:, 2:] / 5) * anc[:, 2:]
        pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
        predicted_bb = box_center_to_corner(pred_bbox)
        
        output = []
        # Skip background class (0)
        for c in range(1, num_classes):
            class_scores = cls_prob[:, c]
            conf_mask = class_scores > pos_threshold
            if not conf_mask.any():
                continue
            
            scores_c = class_scores[conf_mask]
            boxes_c = predicted_bb[conf_mask]
            
            keep = nms(boxes_c, scores_c, nms_threshold)
            
            # class_id, score, x_min, y_min, x_max, y_max
            detections = torch.cat([
                torch.full_like(scores_c[keep], c - 1, dtype=torch.float32).unsqueeze(1),
                scores_c[keep].unsqueeze(1),
                boxes_c[keep]], dim=1)
            output.append(detections)
            
        if output:
            output = torch.cat(output, dim=0)
            out.append(output)
        else:
            out.append(torch.zeros((0, 6), device=device))
            
    return out

# ==============================================================================
# SECTION 2: DATA INGESTION AND TRANSFORMATION PIPELINE
# ==============================================================================

class FiberOpticDataset(Dataset):
    """
    Custom PyTorch Dataset for fiber optic end-face inspection.
    Implements on-the-fly ground truth generation using reference tensors.
    Inspired by d2l.ai's VOCSegDataset.[9]
    """
    def __init__(self, dataset_path, reference_path, img_size=(256, 256), is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        
        # 1. Parse dataset directory - look for actual images (jpg, png) instead of just masks
        if os.path.exists(dataset_path):
            # Look for actual image files in the dataset path
            image_extensions = ['*.png', '*.jpg', '*.jpeg']
            self.image_files = []
            for ext in image_extensions:
                self.image_files.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
            
            # Filter out mask files
            self.image_files = [f for f in self.image_files if 'mask_' not in os.path.basename(f)]
            print(f"Found {len(self.image_files)} images in {dataset_path}")
        else:
            # Use test images from version4 as fallback
            test_dir = os.path.join(os.path.dirname(dataset_path), 'version4')
            self.image_files = glob.glob(os.path.join(test_dir, '*.jpg'))
            print(f"Dataset path not found, using test images from {test_dir}: {len(self.image_files)} images")

        # 2. Parse reference directory
        self.reference_tensors = self._load_reference_tensors(reference_path)
        print(f"Loaded {len(self.reference_tensors)} reference tensors.")

        # 3. Define classes
        self.seg_classes = {'background': 0, 'ferrule': 1, 'cladding': 2, 'core': 3}
        self.defect_classes = {'scratch': 0, 'pit': 1} # Example defect classes
        self.inv_seg_classes = {v: k for k, v in self.seg_classes.items()}
        
        # 4. Define transformations (inspired by d2l.ai's normalization)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_reference_tensors(self, reference_path):
        refs = {}
        pt_files = glob.glob(os.path.join(reference_path, '**', '*.pt'), recursive=True)
        for pt_file in pt_files:
            try:
                # Key is the filename without extension, e.g., 'core'
                key = os.path.splitext(os.path.basename(pt_file))[0]
                refs[key] = torch.load(pt_file, weights_only=False)  # Allow loading older tensors
            except Exception as e:
                print(f"Warning: Could not load reference tensor {pt_file}: {e}")
        return refs

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        # On-the-fly ground truth generation
        seg_mask, defect_boxes = self._generate_ground_truth(image)
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Pad defect boxes to a fixed size for batching
        max_boxes = 20
        padded_boxes = torch.full((max_boxes, 5), -1.0)
        num_boxes = min(max_boxes, defect_boxes.shape[0])
        if num_boxes > 0:
            padded_boxes[:num_boxes] = defect_boxes[:num_boxes]

        return image_tensor, seg_mask.long(), padded_boxes

    def _generate_ground_truth(self, image):
        """
        Generates segmentation mask and defect boxes using template matching
        with reference tensors. This is a crucial step that automates labeling.
        """
        h, w, _ = image.shape
        seg_mask = torch.zeros((h, w), dtype=torch.uint8)
        defect_boxes = []

        # Convert image to grayscale for template matching
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Generate Segmentation Mask
        for name, class_id in self.seg_classes.items():
            if name in self.reference_tensors and name != 'background':
                template = self.reference_tensors[name]
                # Convert tensor to numpy if needed
                if isinstance(template, torch.Tensor):
                    template = template.cpu().numpy().astype(np.uint8)
                else:
                    template = template.astype(np.uint8)
                    
                if template.ndim > 2: # Ensure template is grayscale
                    template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
                
                res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                if max_val > 0.7: # Confidence threshold
                    th, tw = template.shape
                    top_left = max_loc
                    bottom_right = (top_left[0] + tw, top_left[1] + th)
                    # "Paint" the class ID onto the mask
                    seg_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = class_id
        
        # Generate Defect Bounding Boxes
        for name, class_id in self.defect_classes.items():
            if name in self.reference_tensors:
                template = self.reference_tensors[name]
                # Convert tensor to numpy if needed  
                if isinstance(template, torch.Tensor):
                    template = template.cpu().numpy().astype(np.uint8)
                else:
                    template = template.astype(np.uint8)
                    
                if template.ndim > 2:
                    template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
                
                res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.8) # Threshold for multiple detections
                th, tw = template.shape
                
                for pt in zip(*loc[::-1]):
                    x1, y1 = pt[0], pt[1]
                    x2, y2 = x1 + tw, y1 + th
                    # Normalize coordinates to [0,1]
                    box = [class_id, x1/w, y1/h, x2/w, y2/h]
                    defect_boxes.append(box)

        return seg_mask, torch.tensor(defect_boxes, dtype=torch.float32)

# ==============================================================================
# SECTION 3: UNIFIED MODEL ARCHITECTURE
# ==============================================================================

class UnifiedFiberInspector(nn.Module):
    """
    The main multi-task model for fiber inspection.
    Combines a shared backbone, an FCN head for segmentation, and an
    SSD head for mask-guided defect detection.
    """
    def __init__(self, num_seg_classes, num_defect_classes):
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.num_defect_classes = num_defect_classes + 1 # Add background class

        # --- Shared Backbone (ResNet-18) ---
        # Inspired by d2l.ai FCN example [4]
        pretrained_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(pretrained_net.children())[:-2])
        
        # --- FCN Segmentation Head ---
        # Inspired by d2l.ai FCN example [4]
        self.fcn_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, self.num_seg_classes, kernel_size=2, stride=2)
        )
        
        # --- SSD Defect Detection Head ---
        # Inspired by d2l.ai TinySSD example [6]
        self.defect_sizes = [[0.1, 0.15, 0.2]] # Example sizes for defects
        self.defect_ratios = [[1, 2, 0.5]] # Ratios for scratches/pits
        num_anchors = len(self.defect_sizes) + len(self.defect_ratios) - 1

        # Use the feature map from the backbone (512 channels)
        self.cls_predictor = nn.Conv2d(512, num_anchors * self.num_defect_classes, kernel_size=3, padding=1)
        self.bbox_predictor = nn.Conv2d(512, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        # 1. Backbone feature extraction
        features = self.backbone(x)
        
        # 2. FCN head for segmentation
        seg_logits = self.fcn_head(features)
        
        # 3. Mask-guided detection
        with torch.no_grad():
            seg_pred = torch.argmax(seg_logits, dim=1)
            # Create mask for 'core' (3) and 'cladding' (2)
            attention_mask = ((seg_pred == 2) | (seg_pred == 3)).float().unsqueeze(1)
            # Resize mask to match feature map size
            attention_mask_resized = nn.functional.interpolate(attention_mask, size=features.shape[-2:])
        
        # Apply mask to features
        masked_features = features * attention_mask_resized
        
        # 4. SSD head for defect detection
        anchors = multibox_prior(features, self.defect_sizes, self.defect_ratios)
        cls_preds = self.cls_predictor(masked_features)
        bbox_preds = self.bbox_predictor(masked_features)
        
        # Reshape predictions for loss calculation
        # Inspired by d2l.ai concat_preds and reshape [6]
        def flatten_pred(pred):
            return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)
        
        cls_preds = flatten_pred(cls_preds).reshape(x.shape[0], -1, self.num_defect_classes)
        bbox_preds = flatten_pred(bbox_preds).reshape(x.shape[0], -1, 4)
        
        return seg_logits, cls_preds, bbox_preds, anchors

# ==============================================================================
# SECTION 4: TRAINING COMPONENTS
# ==============================================================================

class Accumulator:
    """For accumulating sums over `n` variables.
    Reference: d2l.ai, used in many training loops [6]
    """
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def calc_total_loss(seg_logits, seg_labels, cls_preds, bbox_preds, anchors, bbox_labels, alpha=1.0, beta=1.0):
    """
    Calculates the grand composite loss for the unified model.
    - alpha: weight for segmentation loss
    - beta: weight for detection loss
    """
    # Segmentation Loss (Cross-Entropy)
    seg_loss_fn = nn.CrossEntropyLoss(ignore_index=0) # Ignore background
    seg_loss = seg_loss_fn(seg_logits, seg_labels)
    
    # Detection Loss (Classification + Regression)
    # Inspired by d2l.ai TinySSD loss calculation [5, 6]
    cls_loss_fn = nn.CrossEntropyLoss(reduction='none')
    bbox_loss_fn = nn.SmoothL1Loss(reduction='none') # As suggested in [5]
    
    bbox_offset, bbox_mask, cls_labels = multibox_target(anchors, bbox_labels)
    
    # Class loss
    cls_loss = cls_loss_fn(cls_preds.reshape(-1, cls_preds.shape[-1]), cls_labels.reshape(-1))
    cls_loss = cls_loss.reshape(cls_labels.shape)
    # Only consider positive and negative anchors, not ignored ones
    cls_loss = (cls_loss * (cls_labels >= 0).float()).sum() / (cls_labels >= 0).float().sum().clamp(min=1)

    # Bbox regression loss
    bbox_loss = bbox_loss_fn(bbox_preds, bbox_offset) * bbox_mask.float()
    bbox_loss = bbox_loss.sum() / bbox_mask.sum().clamp(min=1)
    
    det_loss = cls_loss + bbox_loss
    
    # Total weighted loss
    total_loss = alpha * seg_loss + beta * det_loss
    
    return total_loss, seg_loss, det_loss

def train_epoch(net, train_iter, loss_fn, optimizer, device):
    """The training loop for one epoch."""
    net.train()
    metric = Accumulator(4) # total_loss, seg_loss, det_loss, num_examples
    for batch in train_iter:
        optimizer.zero_grad()
        X, seg_y, bbox_y = [v.to(device) for v in batch]
        
        seg_logits, cls_preds, bbox_preds, anchors = net(X)
        
        l, seg_l, det_l = loss_fn(seg_logits, seg_y, cls_preds, bbox_preds, anchors, bbox_y)
        
        l.backward()
        optimizer.step()
        
        metric.add(l.item(), seg_l.item(), det_l.item(), X.shape[0])
    
    return metric[0]/metric[3], metric[1]/metric[3], metric[2]/metric[3]

# ==============================================================================
# SECTION 5: EXECUTION AND INFERENCE
# ==============================================================================

def run_inference(model_path, image_path, device, seg_classes, defect_classes):
    """
    Loads a trained model and runs inference on a single image,
    saving a visualized output.
    """
    print(f"Running inference on {image_path} with model {model_path}...")
    
    # Load model
    num_seg_classes = len(seg_classes)
    num_defect_classes = len(defect_classes)
    net = UnifiedFiberInspector(num_seg_classes, num_defect_classes)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_image = cv2.resize(image, (256, 256))
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(orig_image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        seg_logits, cls_preds, bbox_preds, anchors = net(img_tensor)

    # Post-process segmentation
    seg_pred = torch.argmax(seg_logits.squeeze(0), dim=0).cpu().numpy()
    
    # Post-process detections
    detections = multibox_detection(cls_preds, bbox_preds, anchors)
    
    # Visualize results
    # Inspired by d2l.ai show_bboxes [11]
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(orig_image)
    
    # Overlay segmentation mask
    seg_colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8) # B,G,R,C
    seg_overlay = seg_colors[seg_pred]
    ax.imshow(seg_overlay, alpha=0.3)
    
    # Draw detection boxes
    for det in detections[0]:  # detections is a list, get first item
        class_id, score, x1, y1, x2, y2 = det.cpu().numpy()
        if score < 0.5: continue
        
        class_name = list(defect_classes.keys())[int(class_id)]
        
        box_w = (x2 - x1) * 256
        box_h = (y2 - y1) * 256
        rect = patches.Rectangle((x1 * 256, y1 * 256), box_w, box_h,
                                 linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1 * 256, y1 * 256 - 5, f'{class_name}: {score:.2f}',
                bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')
    
    plt.axis('off')
    output_path = os.path.join(os.path.dirname(image_path), "inspection_result.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Inference result saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fiber Optic End-Face Inspector Training Script")
    parser.add_argument('--dataset_path', type=str, default='../dataset', help="Path to the dataset directory")
    parser.add_argument('--reference_path', type=str, default='../reference', help="Path to the reference tensors directory")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Training batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help="Run mode: train or inference")
    parser.add_argument('--model_path', type=str, default='./fiber_inspector.pth', help="Path to save/load the model")
    parser.add_argument('--image_path', type=str, help="Path to an image for inference mode")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert relative paths to absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.dataset_path):
        args.dataset_path = os.path.join(script_dir, args.dataset_path)
    if not os.path.isabs(args.reference_path):
        args.reference_path = os.path.join(script_dir, args.reference_path)
    
    print(f"Dataset path: {args.dataset_path}")
    print(f"Reference path: {args.reference_path}")

    # Instantiate dataset and dataloader
    fiber_dataset = FiberOpticDataset(args.dataset_path, args.reference_path)
    
    # Validate dataset
    if len(fiber_dataset) == 0:
        print("Error: No images found in the dataset directory!")
        print("Please check that the dataset path contains valid image files.")
        exit(1)
    
    print(f"Dataset loaded successfully with {len(fiber_dataset)} images")
    
    if args.mode == 'train':
        # Adjust batch size if dataset is smaller
        effective_batch_size = min(args.batch_size, len(fiber_dataset))
        if effective_batch_size != args.batch_size:
            print(f"Adjusted batch size from {args.batch_size} to {effective_batch_size} based on dataset size")
            
        train_loader = DataLoader(fiber_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=0)  # Use 0 workers for Windows compatibility
        
        # Instantiate model
        net = UnifiedFiberInspector(
            num_seg_classes=len(fiber_dataset.seg_classes),
            num_defect_classes=len(fiber_dataset.defect_classes)
        ).to(device)

        # Optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        print("Starting training...")
        start_time = time.time()
        
        stats = []
        for epoch in range(args.epochs):
            train_loss, seg_loss, det_loss = train_epoch(net, train_loader, calc_total_loss, optimizer, device)
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Total Loss: {train_loss:.4f} | "
                  f"Seg Loss: {seg_loss:.4f} | "
                  f"Det Loss: {det_loss:.4f}")
            stats.append([epoch+1, train_loss, seg_loss, det_loss])
        
        end_time = time.time()
        total_training_time = end_time - start_time
        
        # Save model
        torch.save(net.state_dict(), args.model_path)
        print(f"Training complete. Model saved to {args.model_path}")

        # --- Statistical Report ---
        print("\n--- STATISTICAL REPORT ---")
        print(f"Total Training Time: {total_training_time:.2f} seconds")
        print(f"Images Processed: {len(fiber_dataset) * args.epochs}")
        print(f"Throughput: {(len(fiber_dataset) * args.epochs) / total_training_time:.2f} images/sec")
        
        df = pd.DataFrame(stats, columns=['Epoch', 'Total_Loss', 'Seg_Loss', 'Det_Loss'])
        print("\nLoss Progression:")
        print(df.to_string(index=False))
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Total_Loss'], label='Total Loss')
        plt.plot(df['Epoch'], df['Seg_Loss'], label='Segmentation Loss')
        plt.plot(df['Epoch'], df['Det_Loss'], label='Detection Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss_plot.png')
        print("\nTraining loss plot saved to training_loss_plot.png")
        
    elif args.mode == 'inference':
        if not args.image_path:
            raise ValueError("Must provide --image_path for inference mode.")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found at {args.model_path}. Please train a model first.")
            
        run_inference(args.model_path, args.image_path, device, fiber_dataset.seg_classes, fiber_dataset.defect_classes)