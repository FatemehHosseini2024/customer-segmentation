# Algorithm selection: K-Means vs Agglomerative vs DBSCAN

Features: ['recency', 'frequency', 'monetary', 'distinct_products', 'tenure']
Customers: 4,338

Preprocessing: log1p on skewed features + StandardScaler.

## 1) K-Means: elbow + metrics (k=2..10)
| k | inertia | silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| 2 | 12,825 | 0.353 | 1.096 | 2,997 |
| 3 | 9,940 | 0.314 | 1.145 | 2,562 |
| 4 | 8,219 | 0.283 | 1.141 | 2,368 |
| 5 | 7,306 | 0.271 | 1.176 | 2,133 |
| 6 | 6,618 | 0.244 | 1.233 | 1,973 |
| 7 | 6,159 | 0.234 | 1.266 | 1,820 |
| 8 | 5,800 | 0.207 | 1.333 | 1,695 |
| 9 | 5,481 | 0.211 | 1.338 | 1,600 |
| 10 | 5,220 | 0.207 | 1.354 | 1,517 |

## 2) Bootstrap stability: K-Means (B=30, frac=0.8)
| k | mean ARI | std |
|---|---|---|
| 2 | 0.990 | 0.014 |
| 3 | 0.982 | 0.012 |
| 4 | 0.963 | 0.022 |

## 3) Agglomerative clustering: metrics by linkage (k=2..10)
| linkage | k | silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| ward | 2 | 0.334 | 1.047 | 2,389 |
| ward | 3 | 0.260 | 1.346 | 2,088 |
| ward | 4 | 0.214 | 1.317 | 1,797 |
| ward | 5 | 0.219 | 1.243 | 1,736 |
| ward | 6 | 0.209 | 1.197 | 1,583 |
| ward | 7 | 0.165 | 1.392 | 1,434 |
| ward | 8 | 0.142 | 1.413 | 1,330 |
| ward | 9 | 0.141 | 1.528 | 1,261 |
| ward | 10 | 0.147 | 1.484 | 1,202 |
| average | 2 | 0.516 | 0.615 | 157 |
| average | 3 | 0.400 | 0.805 | 86 |
| average | 4 | 0.338 | 0.726 | 59 |
| average | 5 | 0.322 | 0.576 | 45 |
| average | 6 | 0.271 | 0.750 | 560 |
| average | 7 | 0.230 | 0.775 | 471 |
| average | 8 | 0.227 | 0.745 | 405 |
| average | 9 | 0.210 | 0.802 | 361 |
| average | 10 | 0.200 | 0.881 | 469 |
| complete | 2 | 0.401 | 0.725 | 676 |
| complete | 3 | 0.111 | 1.747 | 750 |
| complete | 4 | 0.112 | 1.591 | 510 |
| complete | 5 | 0.184 | 1.382 | 1,057 |
| complete | 6 | 0.172 | 1.277 | 1,011 |
| complete | 7 | 0.166 | 1.316 | 895 |
| complete | 8 | 0.162 | 1.231 | 773 |
| complete | 9 | 0.137 | 1.298 | 806 |
| complete | 10 | 0.138 | 1.330 | 786 |

## 4) Bootstrap stability: Agglomerative (B=30, frac=0.8)
| linkage | k | mean ARI | std |
|---|---|---|---|
| ward | 2 | 0.368 | 0.208 |
| ward | 3 | 0.369 | 0.146 |
| ward | 4 | 0.423 | 0.086 |
| average | 2 | 0.418 | 0.313 |
| average | 3 | 0.460 | 0.261 |
| average | 4 | 0.460 | 0.276 |
| complete | 2 | 0.457 | 0.225 |
| complete | 3 | 0.077 | 0.031 |
| complete | 4 | 0.096 | 0.050 |

## 5) DBSCAN: grid search over eps x min_samples
| eps | min_samples | n_clusters | n_noise | silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|---|---|
| 0.30 | 3 | 130 | 2095 | -0.217 | 1.025 | 29 |
| 0.30 | 5 | 41 | 2716 | -0.194 | 1.005 | 32 |
| 0.30 | 10 | 4 | 3315 | -0.065 | 1.116 | 13 |
| 0.51 | 3 | 14 | 483 | -0.246 | 1.008 | 161 |
| 0.51 | 5 | 12 | 687 | -0.262 | 1.014 | 183 |
| 0.51 | 10 | 3 | 1156 | 0.208 | 1.018 | 909 |
| 0.72 | 3 | 5 | 133 | -0.016 | 0.891 | 5 |
| 0.72 | 5 | 2 | 171 | 0.073 | 1.066 | 3 |
| 0.72 | 10 | 1 | 254 | - | - | - |
| 0.94 | 3 | 3 | 49 | 0.447 | 0.469 | 24 |
| 0.94 | 5 | 1 | 72 | - | - | - |
| 0.94 | 10 | 1 | 102 | - | - | - |
| 1.15 | 3 | 1 | 20 | - | - | - |
| 1.15 | 5 | 1 | 25 | - | - | - |
| 1.15 | 10 | 1 | 38 | - | - | - |
| 1.36 | 3 | 1 | 9 | - | - | - |
| 1.36 | 5 | 1 | 9 | - | - | - |
| 1.36 | 10 | 1 | 16 | - | - | - |
| 1.57 | 3 | 1 | 7 | - | - | - |
| 1.57 | 5 | 1 | 7 | - | - | - |
| 1.57 | 10 | 1 | 10 | - | - | - |
| 1.79 | 3 | 1 | 3 | - | - | - |
| 1.79 | 5 | 1 | 5 | - | - | - |
| 1.79 | 10 | 1 | 5 | - | - | - |
| 2.00 | 3 | 1 | 2 | - | - | - |
| 2.00 | 5 | 1 | 2 | - | - | - |
| 2.00 | 10 | 1 | 4 | - | - | - |

## 6) Bootstrap stability: DBSCAN (B=30, frac=0.8)
| eps | min_samples | n_clusters | mean ARI | std |
|---|---|---|---|---|
| 0.30 | 3 | 130 | 0.885 | 0.010 |
| 0.30 | 5 | 41 | 0.872 | 0.016 |
| 0.30 | 10 | 4 | 0.894 | 0.013 |
| 0.51 | 3 | 14 | 0.933 | 0.011 |
| 0.51 | 5 | 12 | 0.913 | 0.009 |
| 0.51 | 10 | 3 | 0.855 | 0.013 |
| 0.72 | 3 | 5 | 0.932 | 0.023 |
| 0.72 | 5 | 2 | 0.905 | 0.017 |
| 0.94 | 3 | 3 | 0.897 | 0.035 |

## 7) Best silhouette by model
- K-Means: k=2, silhouette=0.353, DB=1.096, CH=2,997
- Agglomerative: linkage=average, k=2, silhouette=0.516, DB=0.615, CH=157
- DBSCAN: eps=0.94, min_samples=3, n_clusters=3, n_noise=49, silhouette=0.447, DB=0.469, CH=24

## 8) Comparison: solutions with more than 2 clusters
| model | n_clusters | min cluster size | silhouette | Davies-Bouldin | Calinski-Harabasz | noise | bootstrap ARI |
|---|---|---|---|---|---|---|---|
| DBSCAN eps=0.94, min_samples=3 | 3 | 3 | 0.447 | 0.469 | 24 | 49 | 0.897 |
| Agglomerative (average) | 3 | 4 | 0.400 | 0.805 | 86 | 0 | 0.460 |
| Agglomerative (average) | 4 | 1 | 0.338 | 0.726 | 59 | 0 | 0.460 |
| Agglomerative (average) | 5 | 1 | 0.322 | 0.576 | 45 | 0 | n/a |
| K-Means | 3 | 1321 | 0.314 | 1.145 | 2,562 | 0 | 0.982 |
| K-Means | 4 | 728 | 0.283 | 1.141 | 2,368 | 0 | 0.963 |
| Agglomerative (average) | 6 | 1 | 0.271 | 0.750 | 560 | 0 | n/a |
| K-Means | 5 | 584 | 0.271 | 1.176 | 2,133 | 0 | n/a |
| Agglomerative (ward) | 3 | 1086 | 0.260 | 1.346 | 2,088 | 0 | 0.369 |
| K-Means | 6 | 426 | 0.244 | 1.233 | 1,973 | 0 | n/a |
| K-Means | 7 | 346 | 0.234 | 1.266 | 1,820 | 0 | n/a |
| Agglomerative (average) | 7 | 1 | 0.230 | 0.775 | 471 | 0 | n/a |
| Agglomerative (average) | 8 | 1 | 0.227 | 0.745 | 405 | 0 | n/a |
| Agglomerative (ward) | 5 | 618 | 0.219 | 1.243 | 1,736 | 0 | n/a |
| Agglomerative (ward) | 4 | 618 | 0.214 | 1.317 | 1,797 | 0 | 0.423 |
| K-Means | 9 | 199 | 0.211 | 1.338 | 1,600 | 0 | n/a |
| Agglomerative (average) | 9 | 1 | 0.210 | 0.802 | 361 | 0 | n/a |
| Agglomerative (ward) | 6 | 83 | 0.209 | 1.197 | 1,583 | 0 | n/a |
| DBSCAN eps=0.51, min_samples=10 | 3 | 9 | 0.208 | 1.018 | 909 | 1156 | 0.855 |
| K-Means | 8 | 265 | 0.207 | 1.333 | 1,695 | 0 | n/a |
| K-Means | 10 | 219 | 0.207 | 1.354 | 1,517 | 0 | n/a |
| Agglomerative (average) | 10 | 1 | 0.200 | 0.881 | 469 | 0 | n/a |
| Agglomerative (complete) | 5 | 6 | 0.184 | 1.382 | 1,057 | 0 | n/a |
| Agglomerative (complete) | 6 | 6 | 0.172 | 1.277 | 1,011 | 0 | n/a |
| Agglomerative (complete) | 7 | 6 | 0.166 | 1.316 | 895 | 0 | n/a |
| Agglomerative (ward) | 7 | 83 | 0.165 | 1.392 | 1,434 | 0 | n/a |
| Agglomerative (complete) | 8 | 4 | 0.162 | 1.231 | 773 | 0 | n/a |
| Agglomerative (ward) | 10 | 83 | 0.147 | 1.484 | 1,202 | 0 | n/a |
| Agglomerative (ward) | 8 | 83 | 0.142 | 1.413 | 1,330 | 0 | n/a |
| Agglomerative (ward) | 9 | 83 | 0.141 | 1.528 | 1,261 | 0 | n/a |
| Agglomerative (complete) | 10 | 4 | 0.138 | 1.330 | 786 | 0 | n/a |
| Agglomerative (complete) | 9 | 4 | 0.137 | 1.298 | 806 | 0 | n/a |
| Agglomerative (complete) | 4 | 6 | 0.112 | 1.591 | 510 | 0 | 0.096 |
| Agglomerative (complete) | 3 | 145 | 0.111 | 1.747 | 750 | 0 | 0.077 |
| DBSCAN eps=0.72, min_samples=3 | 5 | 3 | -0.016 | 0.891 | 5 | 133 | 0.932 |
| DBSCAN eps=0.30, min_samples=10 | 4 | 10 | -0.065 | 1.116 | 13 | 3315 | 0.894 |
| DBSCAN eps=0.30, min_samples=5 | 41 | 2 | -0.194 | 1.005 | 32 | 2716 | 0.872 |
| DBSCAN eps=0.30, min_samples=3 | 130 | 3 | -0.217 | 1.025 | 29 | 2095 | 0.885 |
| DBSCAN eps=0.51, min_samples=3 | 14 | 3 | -0.246 | 1.008 | 161 | 483 | 0.933 |
| DBSCAN eps=0.51, min_samples=5 | 12 | 1 | -0.262 | 1.014 | 183 | 687 | 0.913 |

Final decision: K-Means with k=3 (chosen per min cluster ≥5%, stability ≥0.85, interpretability, and top-2 internal metrics). k=2 reported as a coarse split; k=4 retained only for finer analysis. DBSCAN is used to flag outlier/wholesale customers, not as the general segmentation method.

## 9) K-Means vs Ward agreement (adjusted Rand index)
| k | ARI |
|---|---|
| 3 | 0.571 |
| 4 | 0.429 |

## 10) Interpretability: cluster profiles (original scale means)
Profiles shown for K-Means k=3 and k=4, and DBSCAN eps=0.94/min_samples=3.

### K-Means k=3
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| 0 | 44.06 | 1.65 | 562.68 | 32.96 | 74.72 |
| 1 | 29.54 | 8.98 | 4807.83 | 121.91 | 298.93 |
| 2 | 195.61 | 1.84 | 578.23 | 25.42 | 274.86 |

### K-Means k=4
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| 0 | 226.91 | 1.42 | 333.75 | 16.25 | 262.4 |
| 1 | 69.19 | 3.91 | 1584.66 | 69.17 | 275.6 |
| 2 | 41.97 | 1.57 | 526.27 | 31.23 | 65.33 |
| 3 | 11.9 | 13.55 | 7918.6 | 162.62 | 317.33 |

### DBSCAN eps=0.94, min_samples=3 (k=3, incl. noise)
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| noise | 47.25 | 28.15 | 43076.19 | 232.98 | 275.46 |
| 0 | 92.16 | 3.91 | 1517.58 | 58.89 | 221.51 |
| 1 | 1.0 | 104.0 | 53304.89 | 841.67 | 371.33 |
| 2 | 3.0 | 41.0 | 52772.77 | 73.0 | 366.0 |