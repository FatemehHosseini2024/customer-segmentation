# Algorithm selection: K-Means vs Agglomerative clustering

Features: ['recency', 'frequency', 'monetary', 'distinct_products', 'tenure']
Customers: 4,338

Preprocessing: log1p on skewed features + StandardScaler on the scaled data.

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

## 5) Best silhouette by model
- K-Means: k=2, silhouette=0.353
- Agglomerative: linkage=average, k=2, silhouette=0.516

## 6) Interpretability: cluster profiles (original scale means)

### K-Means k=2
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| 0 | 33.75 | 8.36 | 4422.04 | 116.08 | 289.04 |
| 1 | 128.84 | 1.63 | 516.27 | 26.05 | 179.22 |

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