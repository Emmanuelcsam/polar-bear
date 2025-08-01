# Advanced Statistical Analysis Report

## Distribution Analysis

### core_radius
- Skewness: 1.7737
- Kurtosis: 4.3947
- Jarque-Bera statistic: 8128.4177
- Jarque-Bera p-value: 0.0000
- Standard Error of Mean: 1.0027
- Median Absolute Deviation: 33.7993
- Trimmed Mean (10%): 101.1934
- Gini Coefficient: 0.3382

### cladding_radius
- Skewness: 1.7510
- Kurtosis: 4.1602
- Jarque-Bera statistic: 7535.8333
- Jarque-Bera p-value: 0.0000
- Standard Error of Mean: 1.2852
- Median Absolute Deviation: 16.2299
- Trimmed Mean (10%): 191.6975
- Gini Coefficient: 0.2320

### core_cladding_ratio
- Skewness: 0.5531
- Kurtosis: -0.3716
- Jarque-Bera statistic: 347.0538
- Jarque-Bera p-value: 0.0000
- Standard Error of Mean: 0.0023
- Median Absolute Deviation: 0.0829
- Trimmed Mean (10%): 0.5163
- Gini Coefficient: 0.1871

## Correlation Matrix Analysis

### Correlation Comparison (Pearson vs Spearman)
Differences indicate non-linear relationships:

- center_x vs center_y:
  - Pearson: 0.9088
  - Spearman: -0.3652
  - Difference: 1.2740

- center_x vs core_radius:
  - Pearson: 0.2570
  - Spearman: -0.1723
  - Difference: 0.4293

- center_x vs cladding_radius:
  - Pearson: 0.3122
  - Spearman: -0.1370
  - Difference: 0.4492

- center_x vs accuracy_computational_separation:
  - Pearson: -0.0557
  - Spearman: -0.2288
  - Difference: 0.1731

- center_x vs accuracy_guess_approach:
  - Pearson: 0.0007
  - Spearman: 0.1575
  - Difference: 0.1569

- center_x vs accuracy_hough_separation:
  - Pearson: -0.0360
  - Spearman: 0.2096
  - Difference: 0.2456

- center_x vs num_contributing_methods:
  - Pearson: -0.0851
  - Spearman: 0.0356
  - Difference: 0.1207

- center_x vs core_cladding_ratio:
  - Pearson: -0.0112
  - Spearman: -0.2355
  - Difference: 0.2243

- center_x vs core_area:
  - Pearson: 0.3358
  - Spearman: -0.1723
  - Difference: 0.5081

- center_x vs cladding_area:
  - Pearson: 0.3257
  - Spearman: -0.1370
  - Difference: 0.4628

- center_x vs center_distance_from_origin:
  - Pearson: 0.9878
  - Spearman: 0.8419
  - Difference: 0.1459

- center_y vs cladding_radius:
  - Pearson: 0.2753
  - Spearman: 0.1512
  - Difference: 0.1241

- center_y vs core_cladding_ratio:
  - Pearson: 0.0285
  - Spearman: 0.1357
  - Difference: 0.1073

- center_y vs core_area:
  - Pearson: 0.2797
  - Spearman: 0.1489
  - Difference: 0.1308

- center_y vs cladding_area:
  - Pearson: 0.2909
  - Spearman: 0.1512
  - Difference: 0.1398

- center_y vs cladding_core_area_diff:
  - Pearson: 0.2156
  - Spearman: -0.1659
  - Difference: 0.3815

- center_y vs center_distance_from_origin:
  - Pearson: 0.9623
  - Spearman: -0.0365
  - Difference: 0.9988

- core_radius vs accuracy_computational_separation:
  - Pearson: 0.4637
  - Spearman: 0.6662
  - Difference: 0.2024

- core_radius vs accuracy_geometric_approach:
  - Pearson: -0.3624
  - Spearman: -0.2448
  - Difference: 0.1176

- core_radius vs accuracy_guess_approach:
  - Pearson: -0.1744
  - Spearman: -0.2779
  - Difference: 0.1035

- core_radius vs accuracy_threshold_separation:
  - Pearson: -0.2218
  - Spearman: 0.0112
  - Difference: 0.2330

- core_radius vs num_contributing_methods:
  - Pearson: -0.0218
  - Spearman: -0.1268
  - Difference: 0.1050

- core_radius vs cladding_core_area_diff:
  - Pearson: 0.3915
  - Spearman: 0.2236
  - Difference: 0.1680

- core_radius vs center_distance_from_origin:
  - Pearson: 0.2511
  - Spearman: -0.1759
  - Difference: 0.4271

- cladding_radius vs num_valid_results:
  - Pearson: -0.2998
  - Spearman: -0.1737
  - Difference: 0.1262

- cladding_radius vs accuracy_computational_separation:
  - Pearson: 0.4540
  - Spearman: 0.6102
  - Difference: 0.1562

- cladding_radius vs accuracy_geometric_approach:
  - Pearson: -0.4836
  - Spearman: -0.3514
  - Difference: 0.1321

- cladding_radius vs accuracy_threshold_separation:
  - Pearson: -0.2878
  - Spearman: -0.1556
  - Difference: 0.1322

- cladding_radius vs num_contributing_methods:
  - Pearson: 0.0310
  - Spearman: -0.1468
  - Difference: 0.1778

- cladding_radius vs core_cladding_ratio:
  - Pearson: 0.2008
  - Spearman: 0.3770
  - Difference: 0.1761

- cladding_radius vs cladding_core_area_diff:
  - Pearson: 0.8366
  - Spearman: 0.5873
  - Difference: 0.2493

- cladding_radius vs center_distance_from_origin:
  - Pearson: 0.3069
  - Spearman: -0.1148
  - Difference: 0.4216

- num_valid_results vs core_area:
  - Pearson: -0.2420
  - Spearman: -0.0994
  - Difference: 0.1427

- num_valid_results vs cladding_area:
  - Pearson: -0.3695
  - Spearman: -0.1737
  - Difference: 0.1958

- num_valid_results vs cladding_core_area_diff:
  - Pearson: -0.3535
  - Spearman: -0.1725
  - Difference: 0.1809

- accuracy_adaptive_intensity vs accuracy_threshold_separation:
  - Pearson: 0.1351
  - Spearman: -0.0258
  - Difference: 0.1610

- accuracy_computational_separation vs accuracy_hough_separation:
  - Pearson: -0.2989
  - Spearman: -0.4293
  - Difference: 0.1304

- accuracy_computational_separation vs core_cladding_ratio:
  - Pearson: 0.2681
  - Spearman: 0.4920
  - Difference: 0.2240

- accuracy_computational_separation vs core_area:
  - Pearson: 0.3818
  - Spearman: 0.6662
  - Difference: 0.2843

- accuracy_computational_separation vs cladding_area:
  - Pearson: 0.4142
  - Spearman: 0.6102
  - Difference: 0.1959

- accuracy_computational_separation vs center_distance_from_origin:
  - Pearson: -0.0617
  - Spearman: -0.2312
  - Difference: 0.1695

- accuracy_geometric_approach vs accuracy_guess_approach:
  - Pearson: 0.3203
  - Spearman: 0.0755
  - Difference: 0.2448

- accuracy_geometric_approach vs accuracy_threshold_separation:
  - Pearson: 0.3903
  - Spearman: 0.2899
  - Difference: 0.1004

- accuracy_guess_approach vs accuracy_threshold_separation:
  - Pearson: 0.1436
  - Spearman: 0.0086
  - Difference: 0.1350

- accuracy_guess_approach vs core_area:
  - Pearson: -0.0915
  - Spearman: -0.2779
  - Difference: 0.1864

- accuracy_guess_approach vs cladding_core_area_diff:
  - Pearson: -0.2250
  - Spearman: -0.1051
  - Difference: 0.1199

- accuracy_guess_approach vs center_distance_from_origin:
  - Pearson: -0.0216
  - Spearman: 0.1400
  - Difference: 0.1616

- accuracy_hough_separation vs accuracy_threshold_separation:
  - Pearson: 0.3053
  - Spearman: 0.1745
  - Difference: 0.1307

- accuracy_hough_separation vs num_contributing_methods:
  - Pearson: -0.0269
  - Spearman: 0.1073
  - Difference: 0.1342

- accuracy_hough_separation vs core_area:
  - Pearson: -0.5496
  - Spearman: -0.6672
  - Difference: 0.1176

- accuracy_hough_separation vs cladding_area:
  - Pearson: -0.4086
  - Spearman: -0.5762
  - Difference: 0.1676

- accuracy_hough_separation vs center_distance_from_origin:
  - Pearson: -0.0531
  - Spearman: 0.1583
  - Difference: 0.2114

- accuracy_threshold_separation vs core_cladding_ratio:
  - Pearson: -0.0719
  - Spearman: 0.1055
  - Difference: 0.1773

- accuracy_threshold_separation vs core_area:
  - Pearson: -0.1794
  - Spearman: 0.0112
  - Difference: 0.1906

- accuracy_threshold_separation vs center_distance_from_origin:
  - Pearson: 0.1224
  - Spearman: 0.0172
  - Difference: 0.1051

- num_contributing_methods vs cladding_area:
  - Pearson: -0.0068
  - Spearman: -0.1468
  - Difference: 0.1399

- num_contributing_methods vs cladding_core_area_diff:
  - Pearson: 0.0202
  - Spearman: 0.1407
  - Difference: 0.1204

- num_contributing_methods vs center_distance_from_origin:
  - Pearson: -0.0975
  - Spearman: 0.0294
  - Difference: 0.1269

- core_cladding_ratio vs core_area:
  - Pearson: 0.5657
  - Spearman: 0.7657
  - Difference: 0.2000

- core_cladding_ratio vs cladding_area:
  - Pearson: 0.1489
  - Spearman: 0.3770
  - Difference: 0.2281

- core_cladding_ratio vs cladding_core_area_diff:
  - Pearson: -0.1870
  - Spearman: -0.3040
  - Difference: 0.1170

- core_cladding_ratio vs center_distance_from_origin:
  - Pearson: 0.0040
  - Spearman: -0.2473
  - Difference: 0.2513

- core_area vs cladding_core_area_diff:
  - Pearson: 0.3656
  - Spearman: 0.2236
  - Difference: 0.1421

- core_area vs center_distance_from_origin:
  - Pearson: 0.3233
  - Spearman: -0.1759
  - Difference: 0.4993

- cladding_area vs cladding_core_area_diff:
  - Pearson: 0.8869
  - Spearman: 0.5873
  - Difference: 0.2996

- cladding_area vs center_distance_from_origin:
  - Pearson: 0.3214
  - Spearman: -0.1148
  - Difference: 0.4362

