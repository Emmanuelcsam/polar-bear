# Comprehensive Statistical Analysis Report

## Overview
- Total files analyzed: 6116
- Total features: 19
- Numeric features: 18

## Key Statistics Summary

### center_x
- Mean: 595.8718
- Std: 116.3471
- Min: 229.0000
- Max: 1337.0000
- Median: 582.8645

### center_y
- Mean: 447.1887
- Std: 87.0680
- Min: 218.0000
- Max: 913.0000
- Median: 426.0000

### core_radius
- Mean: 114.4197
- Std: 78.4117
- Min: 5.0000
- Max: 639.0000
- Median: 91.7830

### cladding_radius
- Mean: 208.8852
- Std: 100.5034
- Min: 14.0000
- Max: 777.0000
- Median: 158.8024

### core_cladding_ratio
- Mean: 0.5302
- Std: 0.1822
- Min: 0.0628
- Max: 1.3714
- Median: 0.4238

### num_valid_results
- Mean: 6.8365
- Std: 0.5569
- Min: 2.0000
- Max: 7.0000
- Median: 7.0000

## Method Accuracy Statistics

### adaptive_intensity
- Mean accuracy: 0.2325
- Std: 0.2724

### computational_separation
- Mean accuracy: 0.2687
- Std: 0.3134

### geometric_approach
- Mean accuracy: 0.7485
- Std: 0.3222

### guess_approach
- Mean accuracy: 0.6086
- Std: 0.3289

### hough_separation
- Mean accuracy: 0.5812
- Std: 0.3557

### threshold_separation
- Mean accuracy: 0.4539
- Std: 0.4094

### unified_core_cladding_detector
- Mean accuracy: 0.3277
- Std: 0.1911

## Correlation Analysis

### High Correlations (|r| > 0.5)
- center_x vs center_y: 0.9088
- center_x vs center_distance_from_origin: 0.9878
- center_y vs center_distance_from_origin: 0.9623
- core_radius vs cladding_radius: 0.7964
- core_radius vs accuracy_hough_separation: -0.6945
- core_radius vs core_cladding_ratio: 0.6972
- core_radius vs core_area: 0.9335
- core_radius vs cladding_area: 0.7395
- cladding_radius vs core_area: 0.7419
- cladding_radius vs cladding_area: 0.9584
- cladding_radius vs cladding_core_area_diff: 0.8366
- accuracy_hough_separation vs core_cladding_ratio: -0.6679
- accuracy_hough_separation vs core_area: -0.5496
- core_cladding_ratio vs core_area: 0.5657
- core_area vs cladding_area: 0.7543
- cladding_area vs cladding_core_area_diff: 0.8869

## Regression Models

### Target: core_radius
- R²: 0.5745
- Equation: `181.278193 - 25.380857*accuracy_adaptive_intensity + 77.136231*accuracy_computational_separation - 11.234236*accuracy_geometric_approach + 16.665568*accuracy_guess_approach - 129.820486*accuracy_hough_separation + 10.694743*accuracy_threshold_separation - 39.121880*accuracy_unified_core_cladding_detector`

### Target: cladding_radius
- R²: 0.4290
- Equation: `313.894572 - 5.826730*accuracy_adaptive_intensity + 102.567323*accuracy_computational_separation - 70.932718*accuracy_geometric_approach - 14.220535*accuracy_guess_approach - 75.585671*accuracy_hough_separation - 5.723188*accuracy_threshold_separation - 70.009830*accuracy_unified_core_cladding_detector`

### Target: core_cladding_ratio
- R²: 0.5231
- Equation: `0.600998 - 0.051924*accuracy_adaptive_intensity + 0.079397*accuracy_computational_separation + 0.128683*accuracy_geometric_approach + 0.026918*accuracy_guess_approach - 0.383285*accuracy_hough_separation + 0.042756*accuracy_threshold_separation + 0.032219*accuracy_unified_core_cladding_detector`

## Master Equation

```
S(I, D) = exp(-sqrt(0.362261*(I_center_x - D_center_x)^2 + 0.202874*(I_center_y - D_center_y)^2 + 0.164540*(I_core_radius - D_core_radius)^2 + 0.270316*(I_cladding_radius - D_cladding_radius)^2 + 0.000001*(I_core_cladding_ratio - D_core_cladding_ratio)^2 + 0.000008*(I_num_valid_results - D_num_valid_results)^2))
```
