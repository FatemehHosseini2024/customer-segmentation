# Final evaluation: K-Means (k=3 primary, k=4 alternative)

## K-Means k=3
| metric | value |
|---|---|
| silhouette (mean) | 0.314 |
| share silhouette < 0.05 | 0.059 |
| share silhouette < 0 | 0.021 |

### Per-cluster silhouette
| cluster | n | mean silhouette |
|---|---|---|
| 0 | 1321 | 0.304 |
| 1 | 1513 | 0.328 |
| 2 | 1504 | 0.310 |

### Profiles (original scale) - selected features
        recency        frequency        monetary          distinct_products         tenure        has_cancellation        avg_order_value        
           mean median      mean median     mean   median              mean median    mean median             mean median            mean  median
cluster                                                                                                                                          
0         44.06   36.0      1.65    1.0   562.68   393.54             32.96   25.0   74.72   64.0             0.19    0.0          367.89  270.80
1         29.54   17.0      8.98    6.0  4807.83  2241.10            121.91   94.0  298.93  318.0             0.64    1.0          528.50  352.21
2        195.61  196.0      1.84    2.0   578.23   370.15             25.42   20.0  274.86  277.0             0.23    0.0          349.83  226.48

### Standardized deviation from global mean (selected features)
         recency  frequency  monetary  distinct_products  tenure
cluster                                                         
0          -0.47      -0.34     -0.17              -0.33   -1.25
1          -0.62       0.61      0.31               0.71    0.65
2           1.04      -0.32     -0.16              -0.42    0.45

Bootstrap (B=30, frac=0.8): monetary cluster-mean spread std across resamples = 0.870

## K-Means k=4
| metric | value |
|---|---|
| silhouette (mean) | 0.283 |
| share silhouette < 0.05 | 0.079 |
| share silhouette < 0 | 0.024 |

### Per-cluster silhouette
| cluster | n | mean silhouette |
|---|---|---|
| 0 | 1082 | 0.341 |
| 1 | 1352 | 0.244 |
| 2 | 1176 | 0.297 |
| 3 | 728 | 0.245 |

### Profiles (original scale) - selected features
        recency        frequency        monetary          distinct_products         tenure        has_cancellation        avg_order_value        
           mean median      mean median     mean   median              mean median    mean median             mean median            mean  median
cluster                                                                                                                                          
0        226.91  233.0      1.42    1.0   333.75   266.12             16.25   13.0  262.40  266.0             0.18    0.0          256.16  193.45
1         69.19   49.0      3.91    4.0  1584.66  1132.08             69.17   57.0  275.60  283.0             0.44    0.0          490.66  324.83
2         41.97   35.0      1.57    1.0   526.27   374.85             31.23   24.0   65.33   59.0             0.18    0.0          363.34  271.05
3         11.90    7.0     13.55   10.0  7918.60  3651.42            162.62  124.5  317.33  357.0             0.75    1.0          609.79  366.86

### Standardized deviation from global mean (selected features)
         recency  frequency  monetary  distinct_products  tenure
cluster                                                         
0           1.35      -0.37     -0.19              -0.53    0.34
1          -0.22      -0.05     -0.05               0.09    0.45
2          -0.50      -0.35     -0.17              -0.35   -1.33
3          -0.80       1.21      0.65               1.19    0.81

Bootstrap (B=30, frac=0.8): monetary cluster-mean spread std across resamples = 1.004

## Marketing labels (k=3; interpretive, not findings)
- Cluster 1 (n=1513): High-value active/loyal (low recency, high frequency/monetary, long tenure).
- Cluster 0 (n=1321): Mid-value recent (moderate recency/frequency/monetary, short tenure).
- Cluster 2 (n=1504): At-risk/inactive (high recency, low frequency/monetary, long tenure).