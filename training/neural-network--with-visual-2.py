# -*- coding: utf-8 -*-
"""
Fiber Optic Endface Defect Detection Neural Network
Based on examples from https://d2l.ai/
Emmanuel Sampson
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # Add TensorBoard import
import numpy as np
import cv2
import os
import glob
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def box_corner_to_center(boxes):
    """Convert corner-format bounding boxes to center-format.
    Chapter 14.2 (Bounding Boxes) [10]
    """
    # Extract corner coordinates (x1, y1, x2, y2)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]  # Get corner coordinates from tensor
    # Calculate center point and dimensions
    cx = (x1 + x2) / 2  # Calculate center x-coordinate
    cy = (y1 + y2) / 2  # Calculate center y-coordinate
    w = x2 - x1  # Calculate width
    h = y2 - y1  # Calculate height
    return torch.stack((cx, cy, w, h), dim=-1)  # Stack center coordinates and dimensions

def box_center_to_corner(boxes):
    """Convert center-format bounding boxes to corner-format.
    """
    # Extract center coordinates and dimensions (cx, cy, w, h)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]  # Get center coordinates and dimensions
    # Calculate corner coordinates
    x1 = cx - 0.5 * w  # Calculate top-left x-coordinate
    y1 = cy - 0.5 * h  # Calculate top-left y-coordinate
    x2 = cx + 0.5 * w  # Calculate bottom-right x-coordinate
    y2 = cy + 0.5 * h  # Calculate bottom-right y-coordinate
    return torch.stack((x1, y1, x2, y2), dim=-1)  # Stack corner coordinates

def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of boxes.
    Chapter 14.5 (Anchor Boxes) [11]
    """
    def box_area(boxes):  # Define function to calculate box area
        return ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))  # Calculate area
    areas1 = box_area(boxes1)  # Calculate areas of first set of boxes
    areas2 = box_area(boxes2)  # Calculate areas of second set of boxes
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # Find intersection upper-left corners
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # Find intersection lower-right corners
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)  # Calculate intersection dimensions (clamp to avoid negative)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]  # Calculate intersection areas
    union_areas = areas1[:, None] + areas2 - inter_areas  # Calculate union areas
    return inter_areas / union_areas  # Return IoU (intersection over union)

def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel.
    Chapter 14.5 (Anchor Boxes) [11]
    """
    in_height, in_width = data.shape[-2:]  # Get input height and width from data tensor
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)  # Get device and count sizes/ratios
    boxes_per_pixel = (num_sizes + num_ratios - 1)  # Calculate total boxes per pixel
    size_tensor = torch.tensor(sizes, device=device, dtype=torch.float32)  # Convert sizes to tensor
    ratio_tensor = torch.tensor(ratios, device=device, dtype=torch.float32)  # Convert ratios to tensor
    offset_h, offset_w = 0.5, 0.5  # Set offset for centering
    steps_h = 1.0 / in_height  # Calculate step size for height
    steps_w = 1.0 / in_width  # Calculate step size for width
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h  # Generate center y-coordinates
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w  # Generate center x-coordinates
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')  # Create coordinate grid
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)  # Flatten coordinate grids
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor),  # Calculate widths for different anchor combinations
                   size_tensor * torch.sqrt(ratio_tensor[1:]))) \
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor),  # Calculate heights for different anchor combinations
                   size_tensor / torch.sqrt(ratio_tensor[1:])))
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2  # Create anchor manipulations
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)  # Create output grid
    output = out_grid + anchor_manipulations  # Add manipulations to grid
    return output.unsqueeze(0)  # Add batch dimension and return

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes.
    Chapter 14.5 (Anchor Boxes) [11]
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]  # Get number of anchors and ground truth boxes
    if num_gt_boxes == 0:  # If no ground truth boxes
        return torch.full((num_anchors,), -1, dtype=torch.long, device=device)  # Return all -1 (no assignment)

    jaccard = box_iou(anchors, ground_truth)  # Calculate IoU between all anchors and ground truth
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)  # Initialize assignment map
    max_ious, indices = torch.max(jaccard, dim=1)  # Find maximum IoU and corresponding indices for each anchor
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)  # Find anchors with IoU above threshold
    box_j = indices[max_ious >= iou_threshold]  # Get corresponding ground truth indices
    anchors_bbox_map[anc_i] = box_j  # Assign ground truth boxes to anchors
    
    # Force assign each gt box to an anchor
    for i in range(num_gt_boxes):  # Iterate through each ground truth box
        max_idx = torch.argmax(jaccard[:, i])  # Find anchor with highest IoU for this ground truth
        anchors_bbox_map[max_idx] = i  # Assign this ground truth to that anchor

    return anchors_bbox_map  # Return assignment map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets.
    Chapter 14.5 (Anchor Boxes) [11]
    """
    c_anc = box_corner_to_center(anchors)  # Convert anchors to center format
    c_assigned_bb = box_corner_to_center(assigned_bb)  # Convert assigned boxes to center format
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]  # Calculate center offset
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])  # Calculate size offset
    offset = torch.cat([offset_xy, offset_wh], axis=1)  # Concatenate offsets
    return offset  # Return offset tensor

def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes.
    Chapter 14.5 (Anchor Boxes) [11]
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)  # Get batch size and anchors
    batch_offset, batch_mask, batch_class_labels = [], [], []  # Initialize lists for batch data
    device, num_anchors = anchors.device, anchors.shape[0]  # Get device and number of anchors
    for i in range(batch_size):  # Iterate through each sample in batch
        label = labels[i, :, :]  # Get labels for current sample
        # Filter out padding boxes
        valid_labels = label[label[:, 0]!= -1]  # Remove padding boxes (marked with -1)

        anchors_bbox_map = assign_anchor_to_bbox(  # Assign anchors to ground truth boxes
            valid_labels[:, 1:], anchors, device)
        
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)  # Create mask for valid anchors
        
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)  # Initialize class labels
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)  # Initialize assigned boxes
        
        indices_true = torch.nonzero(anchors_bbox_map >= 0).reshape(-1)  # Find indices of assigned anchors
        if len(indices_true) > 0:  # If there are assigned anchors
            bb_idx = anchors_bbox_map[indices_true]  # Get corresponding ground truth indices
            class_labels[indices_true] = valid_labels[bb_idx, 0].long() + 1  # Set class labels (add 1 for background)
            assigned_bb[indices_true] = valid_labels[bb_idx, 1:]  # Set assigned boxes
        
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask  # Calculate offsets and apply mask
        
        batch_offset.append(offset)  # Add to batch offsets
        batch_mask.append(bbox_mask.view(-1))  # Add to batch masks
        batch_class_labels.append(class_labels)  # Add to batch class labels
        
    bbox_offset = torch.stack(batch_offset)  # Stack batch offsets
    bbox_mask = torch.stack(batch_mask)  # Stack batch masks
    class_labels = torch.stack(batch_class_labels)  # Stack batch class labels
    return (bbox_offset, bbox_mask, class_labels)  # Return batch data

def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes.
    Chapter 14.6 (Multiscale Object Detection) [11]
    """
    B = torch.argsort(scores, dim=-1, descending=True)  # Sort scores in descending order
    keep = []  # Initialize list to keep boxes
    while B.numel() > 0:  # While there are boxes to process
        i = B[0]  # Get index of highest scoring box
        keep.append(i)  # Add to keep list
        if B.numel() == 1: break  # If only one box left, break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)  # Calculate IoU with remaining boxes
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)  # Find boxes with IoU below threshold
        B = B[inds + 1]  # Update B to remaining boxes
    return torch.tensor(keep, device=boxes.device)  # Return indices of kept boxes

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.1):
    """Predict bounding boxes using non-maximum suppression.
    Chapter 14.6 (Multiscale Object Detection) [11]
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]  # Get device and batch size
    anchors = anchors.squeeze(0)  # Remove batch dimension from anchors
    num_classes, num_anchors = cls_probs.shape[2], cls_probs.shape[1]  # Get number of classes and anchors
    out = []  # Initialize output list
    for i in range(batch_size):  # Iterate through each sample in batch
        cls_prob, offset_pred = cls_probs[i], offset_preds[i]  # Get predictions for current sample
        
        # Apply softmax to get probabilities
        cls_prob = torch.softmax(cls_prob, dim=1)  # Convert logits to probabilities
        
        # Inverse transform offsets to get predicted boxes
        anc = box_corner_to_center(anchors)  # Convert anchors to center format
        pred_bbox_xy = (offset_pred[:, :2] * anc[:, 2:] / 10) + anc[:, :2]  # Calculate predicted center coordinates
        pred_bbox_wh = torch.exp(offset_pred[:, 2:] / 5) * anc[:, 2:]  # Calculate predicted width and height
        pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)  # Concatenate center and size
        predicted_bb = box_center_to_corner(pred_bbox)  # Convert to corner format
        
        output = []  # Initialize output for current sample
        # Skip background class (0)
        for c in range(1, num_classes):  # Iterate through non-background classes
            class_scores = cls_prob[:, c]  # Get scores for current class
            conf_mask = class_scores > pos_threshold  # Find predictions above threshold
            if not conf_mask.any():  # If no predictions above threshold
                continue  # Skip this class
            
            scores_c = class_scores[conf_mask]  # Get scores for confident predictions
            boxes_c = predicted_bb[conf_mask]  # Get boxes for confident predictions
            
            keep = nms(boxes_c, scores_c, nms_threshold)  # Apply non-maximum suppression
            
            # class_id, score, x_min, y_min, x_max, y_max
            detections = torch.cat([  # Concatenate detection information
                torch.full_like(scores_c[keep], c - 1, dtype=torch.float32).unsqueeze(1),  # Class ID
                scores_c[keep].unsqueeze(1),  # Confidence score
                boxes_c[keep]], dim=1)  # Bounding box coordinates
            output.append(detections)  # Add to output
            
        if output:  # If there are detections
            output = torch.cat(output, dim=0)  # Concatenate all detections
            out.append(output)  # Add to batch output
        else:  # If no detections
            out.append(torch.zeros((0, 6), device=device))  # Add empty detection tensor
            
    return out  # Return batch detections

class FiberOpticDataset(Dataset):
    """
    d2l.ai's VOCSegDataset.[9]
    """
    def __init__(self, dataset_path, reference_path, img_size=(256, 256), is_train=True):
        self.img_size = img_size  # Store target image size
        self.is_train = is_train  # Store training mode flag
        
        # 1. Parse dataset directory - find ALL image files recursively
        if os.path.exists(dataset_path):  # Check if dataset path exists
            # Look for ALL image files recursively in the dataset path
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']  # Define supported image extensions
            self.image_files = []  # Initialize list to store image file paths
            
            # Search recursively through all subdirectories
            for ext in image_extensions:  # Iterate through each extension
                pattern = os.path.join(dataset_path, '**', ext)  # Create recursive pattern
                self.image_files.extend(glob.glob(pattern, recursive=True))  # Find all matching files recursively
            
            print(f"Found {len(self.image_files)} images in {dataset_path} (including all subdirectories)")  # Print number of found images
            
            # If no images found, try alternative approach
            if len(self.image_files) == 0:
                print("No images found with standard extensions, checking for mask files...")
                # Look for mask files as fallback
                mask_pattern = os.path.join(dataset_path, '**', 'mask_*.png')
                self.image_files = glob.glob(mask_pattern, recursive=True)
                print(f"Found {len(self.image_files)} mask files as fallback")
        else:  # If dataset path doesn't exist
            print(f"Dataset path {dataset_path} not found!")
            self.image_files = []

        # 2. Parse reference directory - find ALL .pt files recursively
        self.reference_tensors = self._load_reference_tensors(reference_path)  # Load reference tensors
        print(f"Loaded {len(self.reference_tensors)} reference tensors from {reference_path}")

        # 3. Define classes
        self.seg_classes = {'background': 0, 'ferrule': 1, 'cladding': 2, 'core': 3}  # Define segmentation class mapping
        self.defect_classes = {'scratch': 0, 'pit': 1}  # Define defect class mapping
        self.inv_seg_classes = {v: k for k, v in self.seg_classes.items()}  # Create inverse mapping
        
        # 4. Define transformations (inspired by d2l.ai's normalization)
        self.transform = torchvision.transforms.Compose([  # Create transformation pipeline
            torchvision.transforms.ToTensor(),  # Convert PIL Image to tensor
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])

    def _load_reference_tensors(self, reference_path):
        """Load ALL .pt files from reference directory and subdirectories"""
        refs = {}  # Initialize dictionary to store reference tensors
        
        if not os.path.exists(reference_path):  # Check if reference path exists
            print(f"Reference path {reference_path} not found!")
            return refs
        
        # Find ALL .pt files recursively
        pt_files = glob.glob(os.path.join(reference_path, '**', '*.pt'), recursive=True)  # Find all .pt files recursively
        print(f"Found {len(pt_files)} .pt files in reference directory")
        
        for pt_file in pt_files:  # Iterate through each .pt file
            try:  # Try to load the tensor
                # Create a unique key based on the relative path from reference directory
                rel_path = os.path.relpath(pt_file, reference_path)  # Get relative path
                key = os.path.splitext(rel_path)[0]  # Remove extension to get key
                key = key.replace(os.sep, '_')  # Replace path separators with underscores
                
                # Load the tensor
                tensor = torch.load(pt_file, weights_only=False)  # Load tensor (allow loading older tensors)
                refs[key] = tensor  # Store with unique key
                
                if len(refs) % 100 == 0:  # Print progress every 100 files
                    print(f"Loaded {len(refs)} reference tensors...")
                    
            except Exception as e:  # If loading fails
                print(f"Warning: Could not load reference tensor {pt_file}: {e}")  # Print warning message
        
        print(f"Successfully loaded {len(refs)} reference tensors")  # Print final count
        return refs  # Return loaded reference tensors

    def __len__(self):
        return len(self.image_files)  # Return total number of images

    def __getitem__(self, idx):
        img_path = self.image_files[idx]  # Get image path at given index
        
        # Load image - handle both regular images and mask files
        image = cv2.imread(img_path)  # Read image using OpenCV
        if image is None:  # If image couldn't be loaded
            print(f"Warning: Could not load image {img_path}")
            # Create a dummy image as fallback
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB color space
            image = cv2.resize(image, self.img_size)  # Resize image to target size

        # On-the-fly ground truth generation
        seg_mask, defect_boxes = self._generate_ground_truth(image)  # Generate segmentation mask and defect boxes
        
        # Apply transformations
        image_tensor = self.transform(image)  # Apply transformation pipeline
        
        # Pad defect boxes to a fixed size for batching
        max_boxes = 20  # Set maximum number of boxes per image
        padded_boxes = torch.full((max_boxes, 5), -1.0)  # Create padded tensor with -1 values
        num_boxes = min(max_boxes, defect_boxes.shape[0])  # Calculate number of boxes to copy
        if num_boxes > 0:  # If there are boxes to copy
            padded_boxes[:num_boxes] = defect_boxes[:num_boxes]  # Copy boxes to padded tensor

        return image_tensor, seg_mask.long(), padded_boxes  # Return image tensor, segmentation mask, and padded boxes

    def _generate_ground_truth(self, image):
        """
        Generates segmentation mask and defect boxes using template matching
        with reference tensors. This is a crucial step that automates labeling.
        """
        h, w, _ = image.shape  # Get image dimensions
        seg_mask = torch.zeros((h, w), dtype=torch.uint8)  # Initialize segmentation mask
        defect_boxes = []  # Initialize list for defect bounding boxes

        # Convert image to grayscale for template matching
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

        # Generate Segmentation Mask
        for name, class_id in self.seg_classes.items():  # Iterate through segmentation classes
            if name in self.reference_tensors and name != 'background':  # If reference tensor exists and not background
                template = self.reference_tensors[name]  # Get reference template
                # Convert tensor to numpy if needed
                if isinstance(template, torch.Tensor):  # If template is a PyTorch tensor
                    template = template.cpu().numpy().astype(np.uint8)  # Convert to numpy array
                else:  # If already numpy array
                    template = template.astype(np.uint8)  # Ensure uint8 type
                    
                if template.ndim > 2:  # Ensure template is grayscale
                    template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
                
                res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)  # Perform template matching
                _, max_val, _, max_loc = cv2.minMaxLoc(res)  # Find best match location and value
                
                if max_val > 0.7:  # If match confidence is above threshold
                    th, tw = template.shape  # Get template dimensions
                    top_left = max_loc  # Get top-left corner of match
                    bottom_right = (top_left[0] + tw, top_left[1] + th)  # Calculate bottom-right corner
                    # "Paint" the class ID onto the mask
                    seg_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = class_id  # Fill mask region
        
        # Generate Defect Bounding Boxes
        for name, class_id in self.defect_classes.items():  # Iterate through defect classes
            if name in self.reference_tensors:  # If reference tensor exists
                template = self.reference_tensors[name]  # Get reference template
                # Convert tensor to numpy if needed  
                if isinstance(template, torch.Tensor):  # If template is a PyTorch tensor
                    template = template.cpu().numpy().astype(np.uint8)  # Convert to numpy array
                else:  # If already numpy array
                    template = template.astype(np.uint8)  # Ensure uint8 type
                    
                if template.ndim > 2:  # If template has more than 2 dimensions
                    template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
                
                res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)  # Perform template matching
                loc = np.where(res >= 0.8)  # Find locations above threshold for multiple detections
                th, tw = template.shape  # Get template dimensions
                
                for pt in zip(*loc[::-1]):  # Iterate through detected locations
                    x1, y1 = pt[0], pt[1]  # Get top-left coordinates
                    x2, y2 = x1 + tw, y1 + th  # Calculate bottom-right coordinates
                    # Normalize coordinates to [0,1]
                    box = [class_id, x1/w, y1/h, x2/w, y2/h]  # Create normalized bounding box
                    defect_boxes.append(box)  # Add to defect boxes list

        return seg_mask, torch.tensor(defect_boxes, dtype=torch.float32)  # Return mask and boxes tensor

class UnifiedFiberInspector(nn.Module):
    """
    The main model:
    FCN head for segmentation
    SSD head for defect detection.
    """
    def __init__(self, num_seg_classes, num_defect_classes):
        super().__init__()  # Initialize parent class
        self.num_seg_classes = num_seg_classes  # Store number of segmentation classes
        self.num_defect_classes = num_defect_classes + 1  # Add background class for detection

        # --- Shared Backbone (ResNet-18) ---
        # Inspired by d2l.ai FCN example [4]
        pretrained_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Load pretrained ResNet-18
        self.backbone = nn.Sequential(*list(pretrained_net.children())[:-2])  # Remove last two layers (avgpool and fc)
        
        # --- FCN Segmentation Head ---
        # Inspired by d2l.ai FCN example [4]
        self.fcn_head = nn.Sequential(  # Create FCN head for segmentation
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # First upsampling layer
            nn.ReLU(inplace=True),  # ReLU activation
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Second upsampling layer
            nn.ReLU(inplace=True),  # ReLU activation
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Third upsampling layer
            nn.ReLU(inplace=True),  # ReLU activation
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # Fourth upsampling layer
            nn.ReLU(inplace=True),  # ReLU activation
            nn.ConvTranspose2d(32, self.num_seg_classes, kernel_size=2, stride=2)  # Final upsampling to output classes
        )
        
        # --- SSD Defect Detection Head ---
        # Inspired by d2l.ai TinySSD example [6]
        self.defect_sizes = [[0.1, 0.15, 0.2]]  # Define anchor box sizes for defects
        self.defect_ratios = [[1, 2, 0.5]]  # Define anchor box ratios for defects
        num_anchors = len(self.defect_sizes) + len(self.defect_ratios) - 1  # Calculate total number of anchors

        # Use the feature map from the backbone (512 channels)
        self.cls_predictor = nn.Conv2d(512, num_anchors * self.num_defect_classes, kernel_size=3, padding=1)  # Classification predictor
        self.bbox_predictor = nn.Conv2d(512, num_anchors * 4, kernel_size=3, padding=1)  # Bounding box regression predictor

    def forward(self, x):
        # 1. Backbone feature extraction
        features = self.backbone(x)  # Extract features using ResNet-18 backbone
        
        # 2. FCN head for segmentation
        seg_logits = self.fcn_head(features)  # Generate segmentation logits
        
        # 3. Mask-guided detection
        with torch.no_grad():  # Disable gradient computation for mask generation
            seg_pred = torch.argmax(seg_logits, dim=1)  # Get predicted segmentation classes
            # Create mask for 'core' (3) and 'cladding' (2)
            attention_mask = ((seg_pred == 2) | (seg_pred == 3)).float().unsqueeze(1)  # Create attention mask for core/cladding regions
            # Resize mask to match feature map size
            attention_mask_resized = nn.functional.interpolate(attention_mask, size=features.shape[-2:])  # Resize mask to feature map dimensions
        
        # Apply mask to features
        masked_features = features * attention_mask_resized  # Apply attention mask to features
        
        # 4. SSD head for defect detection
        anchors = multibox_prior(features, self.defect_sizes, self.defect_ratios)  # Generate anchor boxes
        cls_preds = self.cls_predictor(masked_features)  # Generate classification predictions
        bbox_preds = self.bbox_predictor(masked_features)  # Generate bounding box regression predictions
        
        # Reshape predictions for loss calculation
        # Inspired by d2l.ai concat_preds and reshape [6]
        def flatten_pred(pred):  # Define function to flatten predictions
            return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)  # Flatten spatial dimensions
        
        cls_preds = flatten_pred(cls_preds).reshape(x.shape[0], -1, self.num_defect_classes)  # Reshape classification predictions
        bbox_preds = flatten_pred(bbox_preds).reshape(x.shape[0], -1, 4)  # Reshape bounding box predictions
        
        return seg_logits, cls_preds, bbox_preds, anchors  # Return all outputs

class Accumulator:
    """For accumulating sums over `n` variables.
    Used in many training loops [6]
    """
    def __init__(self, n):
        self.data = [0.0] * n  # Initialize list with n zeros
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]  # Add corresponding values
    def reset(self):
        self.data = [0.0] * len(self.data)  # Reset all values to zero
    def __getitem__(self, idx):
        return self.data[idx]  # Return value at given index

def calc_total_loss(seg_logits, seg_labels, cls_preds, bbox_preds, anchors, bbox_labels, alpha=1.0, beta=1.0):
    """
    Calculates the grand composite loss for the unified model.
    - alpha: weight for segmentation loss
    - beta: weight for detection loss
    """
    # Segmentation Loss (Cross-Entropy)
    seg_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Create loss function, ignore background class
    seg_loss = seg_loss_fn(seg_logits, seg_labels)  # Calculate segmentation loss
    
    # Detection Loss (Classification + Regression)
    # Inspired by d2l.ai TinySSD loss calculation [5, 6]
    cls_loss_fn = nn.CrossEntropyLoss(reduction='none')  # Create classification loss function
    bbox_loss_fn = nn.SmoothL1Loss(reduction='none')  # Create regression loss function
    
    bbox_offset, bbox_mask, cls_labels = multibox_target(anchors, bbox_labels)  # Generate target labels for anchors
    
    # Class loss
    cls_loss = cls_loss_fn(cls_preds.reshape(-1, cls_preds.shape[-1]), cls_labels.reshape(-1))  # Calculate classification loss
    cls_loss = cls_loss.reshape(cls_labels.shape)  # Reshape loss to match label shape
    # Only consider positive and negative anchors, not ignored ones
    cls_loss = (cls_loss * (cls_labels >= 0).float()).sum() / (cls_labels >= 0).float().sum().clamp(min=1)  # Average loss over valid anchors

    # Bbox regression loss
    bbox_loss = bbox_loss_fn(bbox_preds, bbox_offset) * bbox_mask.float()  # Calculate regression loss and apply mask
    bbox_loss = bbox_loss.sum() / bbox_mask.sum().clamp(min=1)  # Average loss over valid anchors
    
    det_loss = cls_loss + bbox_loss  # Combine classification and regression losses
    
    # Total weighted loss
    total_loss = alpha * seg_loss + beta * det_loss  # Combine segmentation and detection losses
    
    return total_loss, seg_loss, det_loss  # Return total loss and individual components

def train_epoch(net, train_iter, loss_fn, optimizer, device):
    """The training loop for one epoch."""
    net.train()  # Set model to training mode
    metric = Accumulator(4)  # Initialize accumulator for total_loss, seg_loss, det_loss, num_examples
    for batch in train_iter:  # Iterate through training batches
        optimizer.zero_grad()  # Clear gradients from previous iteration
        X, seg_y, bbox_y = [v.to(device) for v in batch]  # Move batch data to device
        
        seg_logits, cls_preds, bbox_preds, anchors = net(X)  # Forward pass through model
        
        l, seg_l, det_l = loss_fn(seg_logits, seg_y, cls_preds, bbox_preds, anchors, bbox_y)  # Calculate loss
        
        l.backward()  # Backward pass to compute gradients
        optimizer.step()  # Update model parameters
        
        metric.add(l.item(), seg_l.item(), det_l.item(), X.shape[0])  # Add batch metrics to accumulator
    
    return metric[0]/metric[3], metric[1]/metric[3], metric[2]/metric[3]  # Return average losses

def run_inference(model_path, image_path, device, seg_classes, defect_classes):
    """
    Loads a trained model and runs inference on a single image,
    saving a visualized output.
    """
    print(f"Running inference on {image_path} with model {model_path}...")  # Print inference start message
    
    # Load model
    num_seg_classes = len(seg_classes)  # Get number of segmentation classes
    num_defect_classes = len(defect_classes)  # Get number of defect classes
    net = UnifiedFiberInspector(num_seg_classes, num_defect_classes)  # Create model instance
    net.load_state_dict(torch.load(model_path, map_location=device))  # Load trained weights
    net.to(device)  # Move model to device
    net.eval()  # Set model to evaluation mode

    # Load and preprocess image
    image = cv2.imread(image_path)  # Read image using OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    orig_image = cv2.resize(image, (256, 256))  # Resize image to 256x256
    
    transform = torchvision.transforms.Compose([  # Create transformation pipeline
        torchvision.transforms.ToTensor(),  # Convert to tensor
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    img_tensor = transform(orig_image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension
    
    # Perform inference
    with torch.no_grad():  # Disable gradient computation
        seg_logits, cls_preds, bbox_preds, anchors = net(img_tensor)  # Forward pass

    # Post-process segmentation
    seg_pred = torch.argmax(seg_logits.squeeze(0), dim=0).cpu().numpy()  # Get predicted segmentation classes
    
    # Post-process detections
    detections = multibox_detection(cls_preds, bbox_preds, anchors)  # Apply detection post-processing
    
    # Visualize results
    # Inspired by d2l.ai show_bboxes [11]
    fig, ax = plt.subplots(1, figsize=(8, 8))  # Create figure and axis
    ax.imshow(orig_image)  # Display original image
    
    # Overlay segmentation mask
    seg_colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)  # Define colors for classes
    seg_overlay = seg_colors[seg_pred]  # Create colored overlay from predictions
    ax.imshow(seg_overlay, alpha=0.3)  # Overlay with transparency
    
    # Draw detection boxes
    for det in detections[0]:  # Iterate through detections (detections is a list, get first item)
        class_id, score, x1, y1, x2, y2 = det.cpu().numpy()  # Extract detection information
        if score < 0.5: continue  # Skip low-confidence detections
        
        class_name = list(defect_classes.keys())[int(class_id)]  # Get class name from class ID
        
        box_w = (x2 - x1) * 256  # Calculate box width in pixels
        box_h = (y2 - y1) * 256  # Calculate box height in pixels
        rect = patches.Rectangle((x1 * 256, y1 * 256), box_w, box_h,  # Create rectangle patch
                                 linewidth=2, edgecolor='yellow', facecolor='none')  # Set rectangle properties
        ax.add_patch(rect)  # Add rectangle to plot
        ax.text(x1 * 256, y1 * 256 - 5, f'{class_name}: {score:.2f}',  # Add text label
                bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')  # Set text properties
    
    plt.axis('off')  # Hide axes
    output_path = os.path.join(os.path.dirname(image_path), "inspection_result.png")  # Set output path
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save figure
    print(f"Inference result saved to {output_path}")  # Print confirmation message
    plt.close()  # Close figure to free memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fiber Optic End-Face Inspector Training Script")  # Create argument parser
    parser.add_argument('--dataset_path', type=str, default='./dataset', help="Path to the dataset directory")  # Dataset path argument
    parser.add_argument('--reference_path', type=str, default='./reference', help="Path to the reference tensors directory")  # Reference path argument
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs")  # Epochs argument
    parser.add_argument('--batch_size', type=int, default=2, help="Training batch size")  # Batch size argument
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Optimizer learning rate")  # Learning rate argument
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help="Run mode: train or inference")  # Mode argument
    parser.add_argument('--model_path', type=str, default='./fiber_inspector.pth', help="Path to save/load the model")  # Model path argument
    parser.add_argument('--image_path', type=str, help="Path to an image for inference mode")  # Image path argument
    args = parser.parse_args()  # Parse command line arguments

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device (GPU if available, else CPU)
    print(f"Using device: {device}")  # Print which device is being used

    # Convert relative paths to absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    if not os.path.isabs(args.dataset_path):  # If dataset path is relative
        args.dataset_path = os.path.join(script_dir, args.dataset_path)  # Make it absolute
    if not os.path.isabs(args.reference_path):  # If reference path is relative
        args.reference_path = os.path.join(script_dir, args.reference_path)  # Make it absolute
    
    print(f"Dataset path: {args.dataset_path}")  # Print dataset path
    print(f"Reference path: {args.reference_path}")  # Print reference path

    # Instantiate dataset and dataloader
    fiber_dataset = FiberOpticDataset(args.dataset_path, args.reference_path)  # Create dataset instance
    
    # Validate dataset
    if len(fiber_dataset) == 0:  # If no images found
        print("Error: No images found in the dataset directory!")  # Print error message
        print("Please check that the dataset path contains valid image files.")  # Print help message
        exit(1)  # Exit with error code
    
    print(f"Dataset loaded successfully with {len(fiber_dataset)} images")  # Print success message
    
    if args.mode == 'train':  # If training mode
        # Adjust batch size if dataset is smaller
        effective_batch_size = min(args.batch_size, len(fiber_dataset))  # Calculate effective batch size
        if effective_batch_size != args.batch_size:  # If batch size was adjusted
            print(f"Adjusted batch size from {args.batch_size} to {effective_batch_size} based on dataset size")  # Print adjustment message
            
        train_loader = DataLoader(fiber_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=0)  # Create training data loader
        
        # Instantiate model
        net = UnifiedFiberInspector(  # Create model instance
            num_seg_classes=len(fiber_dataset.seg_classes),  # Number of segmentation classes
            num_defect_classes=len(fiber_dataset.defect_classes)  # Number of defect classes
        ).to(device)  # Move model to device

        # --- TensorBoard Visualization Code Start ---
        # 1. Create a SummaryWriter to write to the 'runs' directory
        writer = SummaryWriter('runs/unified_fiber_inspector')

        # 2. Get a sample batch of images to trace the graph
        images, _, _ = next(iter(train_loader))

        # 3. Add the model graph to TensorBoard
        writer.add_graph(net, images)
        writer.close()  # Close the writer

        print("\n✅ Model graph saved for TensorBoard.")
        print("Run `tensorboard --logdir=runs` in your terminal to view it.")
        # --- TensorBoard Visualization Code End ---

        # Optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)  # Create Adam optimizer

        print("Starting training...")  # Print training start message
        start_time = time.time()  # Record start time
        
        stats = []  # Initialize list to store training statistics
        for epoch in range(args.epochs):  # Iterate through epochs
            train_loss, seg_loss, det_loss = train_epoch(net, train_loader, calc_total_loss, optimizer, device)  # Train one epoch
            print(f"Epoch {epoch+1}/{args.epochs} | "  # Print epoch progress
                  f"Total Loss: {train_loss:.4f} | "  # Print total loss
                  f"Seg Loss: {seg_loss:.4f} | "  # Print segmentation loss
                  f"Det Loss: {det_loss:.4f}")  # Print detection loss
            stats.append([epoch+1, train_loss, seg_loss, det_loss])  # Store epoch statistics
        
        end_time = time.time()  # Record end time
        total_training_time = end_time - start_time  # Calculate total training time
        
        # Save model
        torch.save(net.state_dict(), args.model_path)  # Save model weights
        print(f"Training complete. Model saved to {args.model_path}")  # Print completion message

        # --- Statistical Report ---
        print("\n--- STATISTICAL REPORT ---")  # Print report header
        print(f"Total Training Time: {total_training_time:.2f} seconds")  # Print training time
        print(f"Images Processed: {len(fiber_dataset) * args.epochs}")  # Print total images processed
        print(f"Throughput: {(len(fiber_dataset) * args.epochs) / total_training_time:.2f} images/sec")  # Print throughput
        
        df = pd.DataFrame(stats, columns=['Epoch', 'Total_Loss', 'Seg_Loss', 'Det_Loss'])  # Create DataFrame from statistics
        print("\nLoss Progression:")  # Print loss progression header
        print(df.to_string(index=False))  # Print DataFrame
        
        # Plot losses
        plt.figure(figsize=(10, 6))  # Create figure
        plt.plot(df['Epoch'], df['Total_Loss'], label='Total Loss')  # Plot total loss
        plt.plot(df['Epoch'], df['Seg_Loss'], label='Segmentation Loss')  # Plot segmentation loss
        plt.plot(df['Epoch'], df['Det_Loss'], label='Detection Loss')  # Plot detection loss
        plt.xlabel('Epoch')  # Set x-axis label
        plt.ylabel('Loss')  # Set y-axis label
        plt.title('Training Loss Over Epochs')  # Set title
        plt.legend()  # Add legend
        plt.grid(True)  # Add grid
        plt.savefig('training_loss_plot.png')  # Save plot
        print("\nTraining loss plot saved to training_loss_plot.png")  # Print confirmation message
        
    elif args.mode == 'inference':  # If inference mode
        if not args.image_path:  # If no image path provided
            raise ValueError("Must provide --image_path for inference mode.")  # Raise error
        if not os.path.exists(args.model_path):  # If model file doesn't exist
            raise FileNotFoundError(f"Model file not found at {args.model_path}. Please train a model first.")  # Raise error
            
        run_inference(args.model_path, args.image_path, device, fiber_dataset.seg_classes, fiber_dataset.defect_classes)  # Run inference