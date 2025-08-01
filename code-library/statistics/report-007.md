# Full Mathematical Expressions

## Linear Regression Equations

### Target Variable: core_radius

#### Full Equation:
```
core_radius = 181.2781928261
        - 25.3808572886 * accuracy_adaptive_intensity
        + 77.1362310817 * accuracy_computational_separation
        - 11.2342355741 * accuracy_geometric_approach
        + 16.6655679368 * accuracy_guess_approach
        - 129.8204862289 * accuracy_hough_separation
        + 10.6947431319 * accuracy_threshold_separation
        - 39.1218796198 * accuracy_unified_core_cladding_detector
```

#### Coefficient Table:
| Feature | Coefficient |
|---------|-------------|
| Intercept | 181.2781928261 |
| accuracy_adaptive_intensity | -25.3808572886 |
| accuracy_computational_separation | 77.1362310817 |
| accuracy_geometric_approach | -11.2342355741 |
| accuracy_guess_approach | 16.6655679368 |
| accuracy_hough_separation | -129.8204862289 |
| accuracy_threshold_separation | 10.6947431319 |
| accuracy_unified_core_cladding_detector | -39.1218796198 |

#### Model Performance:
- R² Score: 0.574518

### Target Variable: cladding_radius

#### Full Equation:
```
cladding_radius = 313.8945720830
        - 5.8267299173 * accuracy_adaptive_intensity
        + 102.5673233058 * accuracy_computational_separation
        - 70.9327176361 * accuracy_geometric_approach
        - 14.2205353203 * accuracy_guess_approach
        - 75.5856711438 * accuracy_hough_separation
        - 5.7231882339 * accuracy_threshold_separation
        - 70.0098300644 * accuracy_unified_core_cladding_detector
```

#### Coefficient Table:
| Feature | Coefficient |
|---------|-------------|
| Intercept | 313.8945720830 |
| accuracy_adaptive_intensity | -5.8267299173 |
| accuracy_computational_separation | 102.5673233058 |
| accuracy_geometric_approach | -70.9327176361 |
| accuracy_guess_approach | -14.2205353203 |
| accuracy_hough_separation | -75.5856711438 |
| accuracy_threshold_separation | -5.7231882339 |
| accuracy_unified_core_cladding_detector | -70.0098300644 |

#### Model Performance:
- R² Score: 0.428996

### Target Variable: core_cladding_ratio

#### Full Equation:
```
core_cladding_ratio = 0.6009977081
        - 0.0519236152 * accuracy_adaptive_intensity
        + 0.0793973339 * accuracy_computational_separation
        + 0.1286830543 * accuracy_geometric_approach
        + 0.0269181091 * accuracy_guess_approach
        - 0.3832852637 * accuracy_hough_separation
        + 0.0427559783 * accuracy_threshold_separation
        + 0.0322189437 * accuracy_unified_core_cladding_detector
```

#### Coefficient Table:
| Feature | Coefficient |
|---------|-------------|
| Intercept | 0.6009977081 |
| accuracy_adaptive_intensity | -0.0519236152 |
| accuracy_computational_separation | 0.0793973339 |
| accuracy_geometric_approach | 0.1286830543 |
| accuracy_guess_approach | 0.0269181091 |
| accuracy_hough_separation | -0.3832852637 |
| accuracy_threshold_separation | 0.0427559783 |
| accuracy_unified_core_cladding_detector | 0.0322189437 |

#### Model Performance:
- R² Score: 0.523102

## Master Similarity Equation

### Full Expression:
```
S(I, D) = exp(-sqrt(
    0.3622606831 * (I_center_x - D_center_x)^2 +
    0.2028743335 * (I_center_y - D_center_y)^2 +
    0.1645400381 * (I_core_radius - D_core_radius)^2 +
    0.2703157582 * (I_cladding_radius - D_cladding_radius)^2 +
    0.0000008887 * (I_core_cladding_ratio - D_core_cladding_ratio)^2 +
    0.0000082984 * (I_num_valid_results - D_num_valid_results)^2
))
```

### Weight Table:
| Feature | Weight |
|---------|--------|
| center_x | 0.3622606831 |
| center_y | 0.2028743335 |
| core_radius | 0.1645400381 |
| cladding_radius | 0.2703157582 |
| core_cladding_ratio | 0.0000008887 |
| num_valid_results | 0.0000082984 |

### Interpretation:
- S(I, D): Similarity score between images I and D (range: 0 to 1)
- I_feature: Value of feature in input image I
- D_feature: Value of feature in database image D
- Weights are normalized variance-based importance factors
- The exponential of negative distance ensures similarity decreases with distance
