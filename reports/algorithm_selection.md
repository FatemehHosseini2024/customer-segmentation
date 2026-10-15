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
| eps | min_samples | mean ARI | std |
|---|---|---|---|
| 0.30 | 3 | 0.885 | 0.010 |
| 0.30 | 5 | 0.872 | 0.016 |
| 0.30 | 10 | 0.894 | 0.013 |
| 0.51 | 3 | 0.933 | 0.011 |
| 0.51 | 5 | 0.913 | 0.009 |
| 0.51 | 10 | 0.855 | 0.013 |

## 7) Best silhouette by model
- K-Means: k=2, silhouette=0.353, DB=1.096, CH=2,997
- Agglomerative: linkage=average, k=2, silhouette=0.516, DB=0.615, CH=157
- DBSCAN: eps=0.94, min_samples=3, n_clusters=3, n_noise=49, silhouette=0.447, DB=0.469, CH=24

## 8) Interpretability: cluster profiles (original scale means)
Profiles shown for K-Means k=2 (best overall: metrics + stability + interpretability).

### K-Means k=2
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| 0 | 33.75 | 8.36 | 4422.04 | 116.08 | 289.04 |
| 1 | 128.84 | 1.63 | 516.27 | 26.05 | 179.22 |

- C0: frequent, recent, high-spend customers (active/high-value).
- C1: infrequent, older purchases, low spend (one-off/inactive).

### DBSCAN eps=0.94, min_samples=3
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| noise | 47.18 | 27.63 | 42199.04 | 228.27 | 273.04 |
| 0 | 92.17 | 3.91 | 1517.91 | 58.91 | 221.52 |
| 1 | 1.0 | 104.0 | 53304.89 | 841.67 | 371.33 |
| 2 | 3.0 | 41.0 | 52772.77 | 73.0 | 366.0 |