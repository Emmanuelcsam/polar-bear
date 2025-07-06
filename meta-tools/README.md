# Fiber Optic Core Defect Detection Dataset

## Dataset Structure
```
dataset/
├── train/
│   ├── scratch/      (48 images: 3 original + 45 augmented)
│   ├── clean/        (50 images: 1 original + 49 augmented)
│   └── contaminated/ (50 images: 5 original + 45 augmented)
├── val/
│   └── contaminated/ (3 images: 1 original + 2 augmented)
├── test/
│   ├── scratch/      (2 images)
│   ├── clean/        (1 image)
│   └── contaminated/ (2 images)
└── masks/            (4 processing masks)
```

## Class Definitions
- **scratch (label: 0)**: Fiber cores with visible linear scratch defects
- **clean (label: 1)**: Clean fiber cores with minimal visible defects
- **contaminated (label: 2)**: Fiber cores with contamination, spots, or other non-scratch defects

## File Naming Convention
`{class}_{split}_{index:04d}_{hash}.png`
- Example: `scratch_train_0001_3a2f2cff.png`

Augmented files include augmentation details:
`{base_name}_aug{index:03d}_{augmentations}.png`
- Example: `scratch_train_0001_3a2f2cff_aug001_rot15_bright120_noise5.png`

## Augmentation Techniques Applied
- Rotation: ±30 degrees
- Brightness: 70-130% adjustment
- Gaussian noise: σ = 2-8
- Gaussian blur: 3x3 or 5x5 kernel
- Elastic deformation (scratch class only)

## Usage with PyTorch
```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('dataset/train', transform=transform)
val_dataset = datasets.ImageFolder('dataset/val', transform=transform)
test_dataset = datasets.ImageFolder('dataset/test', transform=transform)
```

## Metadata Files
- `dataset_metadata.json`: Original image mappings and labels
- `augmented_metadata.json`: Augmentation details and counts
- `distribution_summary.json`: Class distribution statistics
- `training_config.json`: Recommended neural network training parameters

## Notes
- Dataset is augmented due to limited original samples
- Test set contains only original images (no augmentation)
- Masks directory contains processing masks (not for training)
- Class imbalance addressed through augmentation to ~50 images per class in training