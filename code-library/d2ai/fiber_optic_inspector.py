# fiber_optic_inspector.py

# =============================================================================
# Section 1: Imports and Setup
# Based on standard PyTorch and d2l.ai notebook setups [2, 3]
# =============================================================================
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import glob
import time
import pandas as pd

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

# =============================================================================
# Section 2: Custom Dataset for Fiber Optic Imagery
# Logic derived from d2l.ai custom dataset examples [2, 9, 10]
# =============================================================================
class FiberOpticDataset(Dataset):
    """Custom Dataset for loading fiber optic end-face images, masks, and reference features."""
    def __init__(self, dataset_dir, reference_dir, crop_size, is_train=True):
        self.crop_size = crop_size
        self.is_train = is_train
        self.image_paths = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True))
        
        # In a real scenario, masks would be loaded from a corresponding 'masks' directory.
        # For this example, we assume they are generated or exist with a similar naming convention.
        # self.mask_paths = [p.replace(dataset_dir, mask_dir).replace('.png', '_mask.png') for p in self.image_paths]

        self.reference_tensors = self._load_reference_tensors(reference_dir)
        
        # Image normalization transform, standard for models pretrained on ImageNet [17]
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _load_reference_tensors(self, reference_dir):
        # Loads reference tensors from.pt files [13]
        ref_tensors = {}
        pt_files = glob.glob(os.path.join(reference_dir, '*.pt'))
        for pt_file in pt_files:
            key = os.path.basename(pt_file).replace('.pt', '')
            ref_tensors[key] = torch.load(pt_file)
        print(f"Loaded {len(ref_tensors)} reference tensors: {list(ref_tensors.keys())}")
        return ref_tensors

    def _rand_crop(self, feature, label):
        # Synchronized random cropping for image and mask [2]
        rect = torchvision.transforms.RandomCrop.get_params(feature, self.crop_size)
        feature = F.crop(feature, *rect)
        label = F.crop(label, *rect)
        return feature, label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Placeholder for ground truth mask and bounding boxes
        # In a real implementation, these would be loaded from annotation files.
        # For segmentation, the mask should have integer class labels per pixel.
        gt_mask = Image.new('L', image.size, 0) # Example: background mask
        # For detection, boxes are [x1, y1, x2, y2] and labels are class indices.
        gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
        gt_labels = torch.zeros((0,), dtype=torch.int64)

        # Convert PIL images to tensors
        image_tensor = torchvision.transforms.ToTensor()(image)
        gt_mask_tensor = torch.from_numpy(np.array(gt_mask)).long()

        if self.is_train:
            image_tensor, gt_mask_tensor = self._rand_crop(image_tensor, gt_mask_tensor)

        # Normalize the image tensor
        image_tensor = self.transform(image_tensor)

        # Package the sample
        sample = {
            'image': image_tensor,
            'gt_mask': gt_mask_tensor,
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels,
            'reference_features': self.reference_tensors
        }
        return sample

# =============================================================================
# Section 3: Unified FCN-SSD Model Architecture
# Combines FCN [3] and SSD [5] principles into a single model.
# =============================================================================
class FCN_SSD(nn.Module):
    """A unified model for segmentation (FCN) and masked detection (SSD)."""
    def __init__(self, num_seg_classes, num_defect_classes):
        super().__init__()
        # 1. Shared Backbone (Feature Extractor)
        # Using a pretrained ResNet-18 as described in d2l.ai [3, 17]
        pretrained_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(pretrained_net.children())[:-2])
        
        # 2. Stage 1: FCN Head for Segmentation
        self.fcn_head = nn.Sequential(
            nn.Conv2d(512, num_seg_classes, kernel_size=1), # 1x1 conv to map channels to classes [3]
            # Transposed convolution for upsampling [3]
            nn.ConvTranspose2d(num_seg_classes, num_seg_classes, kernel_size=64, padding=16, stride=32)
        )
        # Initialize transposed conv with bilinear kernel [3]
        self.fcn_head.weight.data.copy_(self._bilinear_kernel(num_seg_classes, num_seg_classes, 64))

        # 3. Stage 2: SSD Head for Defect Detection
        # This is a simplified SSD head for demonstration. A full SSD has multiple heads.
        # Based on SSD principles from d2l.ai [4, 5]
        self.ssd_cls_head = nn.Conv2d(512, (num_defect_classes + 1) * 4, kernel_size=3, padding=1) # 4 anchors per location
        self.ssd_bbox_head = nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1)

    def _bilinear_kernel(self, in_channels, out_channels, kernel_size):
        # Helper function to create bilinear interpolation weights [3]
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = (torch.arange(kernel_size).reshape(-1, 1),
              torch.arange(kernel_size).reshape(1, -1))
        filt = (1 - torch.abs(og - center) / factor) * \
               (1 - torch.abs(og - center) / factor)
        weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
        weight[range(in_channels), range(out_channels), :, :] = filt
        return weight

    def forward(self, x):
        img_size = x.shape[-2:]
        
        # 1. Pass through shared backbone
        features = self.backbone(x) # Shape: (N, 512, H/32, W/32)
        
        # 2. FCN Head forward pass for segmentation
        seg_logits = self.fcn_head(features) # Shape: (N, num_seg_classes, H, W)
        
        # 3. Create mask from segmentation prediction for Stage 2
        # We assume class 0 is background, and we want to detect in other regions.
        with torch.no_grad():
            seg_pred = torch.argmax(seg_logits, dim=1) # Shape: (N, H, W)
            # Create a binary mask where regions of interest are 1, background is 0
            interest_mask = (seg_pred > 0).float().unsqueeze(1) # Shape: (N, 1, H, W)
            # Downsample mask to match feature map size for SSD
            ssd_mask = F.interpolate(interest_mask, size=features.shape[-2:], mode='nearest')

        # 4. Apply mask to features before SSD heads
        masked_features = features * ssd_mask
        
        # 5. SSD Head forward pass for detection
        cls_preds = self.ssd_cls_head(masked_features)
        bbox_preds = self.ssd_bbox_head(masked_features)
        
        # Reshape SSD outputs for loss calculation [5]
        #... (reshaping logic would go here)
        
        return seg_logits, cls_preds, bbox_preds, features

# =============================================================================
# Section 4: Loss Functions and Training Loop
# Based on d2l.ai training patterns [17, 19, 25]
# =============================================================================
def train_model(model, dataloader, num_epochs=10, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)

    # Loss functions
    seg_loss_fn = nn.CrossEntropyLoss()
    # Feature similarity loss [20]
    feature_sim_loss_fn = nn.CosineEmbeddingLoss()
    # SSD losses [4, 5]
    ssd_cls_loss_fn = nn.CrossEntropyLoss() # Placeholder, often Focal Loss is used
    ssd_bbox_loss_fn = nn.SmoothL1Loss() # Placeholder

    # Optimizer with different learning rates for backbone and heads [17]
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': learning_rate * 0.1},
        {'params': model.fcn_head.parameters()},
        {'params': model.ssd_cls_head.parameters()},
        {'params': model.ssd_bbox_head.parameters()}
    ], lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        total_loss = 0

        for i, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            gt_masks = batch['gt_mask'].to(device)
            # gt_boxes = batch['gt_boxes'].to(device) # Not used in this simplified loop
            # gt_labels = batch['gt_labels'].to(device) # Not used in this simplified loop
            ref_features = batch['reference_features']

            # Forward pass
            seg_logits, cls_preds, bbox_preds, features = model(images)

            # --- Calculate Composite Loss ---
            # 1. Segmentation Loss
            loss_seg = seg_loss_fn(seg_logits, gt_masks)
            
            # 2. Feature Similarity Loss (Example for 'core')
            # This requires ground truth info about which areas are 'core'
            # For simplicity, we'll create a dummy target
            core_ref_tensor = ref_features['core_ref'].to(device)
            # This is a conceptual example. A real implementation needs to extract
            # features only from the 'core' region of the `features` tensor.
            # Here we just compare the mean feature to the reference.
            mean_img_feature = torch.mean(features, dim=(-1, -2))
            target = torch.ones(images.size(0)).to(device) # Target is 1 for similarity
            loss_sim = feature_sim_loss_fn(mean_img_feature, core_ref_tensor.expand_as(mean_img_feature), target)

            # 3. Detection Losses (Placeholder)
            # This requires complex anchor box matching logic not shown here [23]
            loss_cls = torch.tensor(0.0, device=device) # ssd_cls_loss_fn(...)
            loss_bbox = torch.tensor(0.0, device=device) # ssd_bbox_loss_fn(...)

            # Total weighted loss
            # Weights (w1, w2, w3, w4) are hyperparameters
            w1, w2, w3, w4 = 1.0, 0.5, 1.0, 1.0
            loss = w1 * loss_seg + w2 * loss_sim + w3 * loss_cls + w4 * loss_bbox

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} summary: Average Loss: {total_loss / len(dataloader):.4f}, Time: {epoch_time:.2f}s")

    print("Training finished.")
    return model

# =============================================================================
# Section 5: Main Execution Block
# =============================================================================
if __name__ == '__main__':
    # Configuration
    DATASET_DIR = 'dataset'
    REFERENCE_DIR = 'reference'
    CROP_SIZE = (224, 224)
    BATCH_SIZE = 4
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    NUM_SEG_CLASSES = 4  # core, cladding, ferrule, background
    NUM_DEFECT_CLASSES = 2 # scratch, blob

    # Create dummy directories and data for demonstration
    os.makedirs(os.path.join(DATASET_DIR, 'chunk1'), exist_ok=True)
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    for i in range(20):
        Image.new('RGB', (512, 512), color = 'red').save(os.path.join(DATASET_DIR, 'chunk1', f'img_{i}.png'))
    torch.save(torch.randn(512), os.path.join(REFERENCE_DIR, 'core_ref.pt'))
    torch.save(torch.randn(512), os.path.join(REFERENCE_DIR, 'cladding_ref.pt'))

    # Create Dataset and DataLoader
    fiber_dataset = FiberOpticDataset(
        dataset_dir=DATASET_DIR,
        reference_dir=REFERENCE_DIR,
        crop_size=CROP_SIZE,
        is_train=True
    )
    
    # Check if dataset is empty
    if len(fiber_dataset) == 0:
        print("Error: Dataset is empty. Please check the DATASET_DIR path and contents.")
    else:
        data_loader = DataLoader(
            dataset=fiber_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # Initialize model
        model = FCN_SSD(
            num_seg_classes=NUM_SEG_CLASSES,
            num_defect_classes=NUM_DEFECT_CLASSES
        )

        # Train model
        trained_model = train_model(model, data_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

        # Save the trained model parameters [26]
        torch.save(trained_model.state_dict(), 'fiber_inspector_model.pth')
        print("Trained model saved to fiber_inspector_model.pth")

        # --- Statistics Reporting (Conceptual) ---
        # A full evaluation would require a separate test loop and test dataset.
        # Here we just outline the process.
        print("\n--- Performance Statistics Report ---")
        # Evaluation metrics would be calculated here on a test set.
        # For segmentation: Pixel Accuracy, mIoU
        # For detection: Precision, Recall, mAP@0.5
        
        # Example of creating a results DataFrame
        seg_results = {
            'Class': ['Core', 'Cladding', 'Ferrule', 'Mean'],
            'Pixel Accuracy': [0.98, 0.95, 0.99, 0.973], # Dummy data
            'IoU': [0.92, 0.88, 0.96, 0.92] # Dummy data
        }
        df_seg = pd.DataFrame(seg_results)
        
        det_results = {
            'Defect Class':,
            'Precision': [0.85, 0.91, 0.88], # Dummy data
            'Recall': [0.82, 0.88, 0.85], # Dummy data
            'mAP@0.5': [0.84, 0.90, 0.87] # Dummy data
        }
        df_det = pd.DataFrame(det_results)
        
        print("\nTable 1: Regional Segmentation Performance")
        print(df_seg.to_string(index=False))
        
        print("\nTable 2: Defect Detection Performance")
        print(df_det.to_string(index=False))