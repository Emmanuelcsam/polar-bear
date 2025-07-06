# Ferrule Dataset Organization Summary

Generated: 2025-07-05 10:24:51

## Dataset Overview
Total images processed: 14
Total images reorganized: 14

## Original File Analysis

### Defect Type Distribution
- Contaminated: 8 (57.1%)
- Scratch: 6 (42.9%)

## Reorganized Dataset Structure

### Train/Val/Test Split

**Train Set**: 9 images
  - scratch: 4
  - contaminated: 5

**Val Set**: 2 images
  - scratch: 1
  - contaminated: 1

**Test Set**: 3 images
  - scratch: 1
  - contaminated: 2

## File Naming Convention
Files have been renamed according to the pattern:
`{defect_type}_{split}_{index:04d}_{hash}.png`

Examples:
- `scratch_train_0001_eee5eaa7.png`
- `contaminated_val_0001_f018732a.png`
- `clean_test_0001_abc12345.png`

## Original to New File Mapping

### Scratch Images

### Scratch Images
- `19700101000214-scratch_ferrule.png` → `scratch_train_0001_eee5eaa7.png` (severity: 1.000)
- `19700101000223-scratch_ferrule.png` → `scratch_train_0002_1f19b0e8.png` (severity: 1.000)
- `ferrule.png` → `scratch_train_0003_a0bfc514.png` (severity: 1.000)
- `19700101000222-scratch_ferrule.png` → `scratch_train_0004_3ef888de.png` (severity: 1.000)
- `19700101000232-scratch_ferrule.png` → `scratch_val_0001_f2c69065.png` (severity: 1.000)
  ... and 1 more

### Contaminated Images
- `19700101000102-ubet_ferrule.png` → `contaminated_train_0001_0049853f.png` (severity: 0.313)
- `img (74)_ferrule.png` → `contaminated_train_0002_33115772.png` (severity: 0.307)
- `19700101000054-ubet_ferrule.png` → `contaminated_train_0003_33115772.png` (severity: 0.307)
- `img (49)_ferrule.png` → `contaminated_train_0004_c2b95fd6.png` (severity: 0.306)
- `19700101000030-_ferrule.png` → `contaminated_train_0005_c2b95fd6.png` (severity: 0.306)
  ... and 3 more

## Key Insights

1. **Scratch Detection**: The analysis successfully identified all images with 'scratch' in their filename as having scratch defects
2. **UBET Images**: Images labeled 'ubet' were classified as contaminated, suggesting they may have surface contamination
3. **Unnamed Images**: Generic ferrule images (without specific labels) showed varied defect patterns
4. **No Clean Images**: No images were classified as completely clean, indicating all samples have some form of defect

## Recommendations

1. **Collect Clean Samples**: The dataset lacks clean ferrule images for comparison
2. **Manual Verification**: Review the contaminated classification for UBET images
3. **Augmentation**: Consider data augmentation to balance the dataset
4. **Severity Thresholds**: Fine-tune severity thresholds based on domain expertise