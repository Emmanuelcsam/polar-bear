# Fiber Optic End Face Core Region Classification Report

## Dataset Overview
Total images analyzed: 19 fiber optic core region images

## Classification Categories

### 1. Scratch Defects (5 images - 26.3%)
**Files:** 
- 19700101000214-scratch_core.png
- 19700101000222-scratch_core.png
- 19700101000223-scratch_core.png
- 19700101000232-scratch_core.png
- 19700101000233-scratch_core.png

**Characteristics:**
- Visible linear marks across the fiber core surface
- Scratches appear as thin dark lines on the bright core region
- Some images show multiple scratch patterns
- Scratches vary in orientation and severity

### 2. UBET Defects (2 images - 10.5%)
**Files:**
- 19700101000054-ubet_core.png
- 19700101000102-ubet_core.png

**Characteristics:**
- Clean core appearance with minimal visible defects
- Well-defined circular core boundary
- Uniform illumination across the core region
- Dark surrounding cladding area

### 3. Unlabeled/General Core Images (7 images - 36.8%)
**Files:**
- 19700101000030-_core.png
- 19700101000031-_core.png
- core.png
- core_51.png
- img (49)_core.png
- img (50)_core.png
- img (74)_core.png

**Characteristics:**
- Variable quality and defect patterns
- Some show contamination or surface irregularities
- Central dark spots visible in some samples (possible contamination)
- Varying degrees of core clarity

### 4. Processing/Analysis Images (5 images - 26.3%)
**Files:**
- Davids_circle_extract.png
- circle_split_inner_circle.png
- core_mask.png (binary mask)
- inner_white_mask.png (smaller binary mask)
- white_region_original.png

**Characteristics:**
- Binary masks for image processing
- Extracted circular regions for analysis
- Pure white circles on black backgrounds (masks)
- Used for segmentation and analysis workflows

## Key Observations

1. **Image Standardization**: All images are centered and cropped to show only the core region with consistent black backgrounds

2. **Defect Distribution**:
   - Scratch defects are the most common labeled defect type (26.3%)
   - Many images remain unlabeled, suggesting need for further classification
   - UBET category shows the cleanest cores

3. **Image Quality**: Generally high contrast between bright core and dark background, facilitating automated analysis

4. **Processing Artifacts**: The presence of mask images indicates active image processing and segmentation work

## Recommendations

1. **Expand Classification**: The unlabeled images should be reviewed and classified into appropriate defect categories
2. **Defect Severity**: Consider adding severity levels (minor/moderate/severe) to defect classifications
3. **Additional Categories**: Consider adding categories for contamination, pits, or other observed defects
4. **Standardized Naming**: Implement consistent naming convention for all images including defect type and severity