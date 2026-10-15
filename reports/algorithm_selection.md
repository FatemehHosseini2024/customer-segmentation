# Algorithm selection: K-Means

Features: ['recency', 'frequency', 'monetary', 'distinct_products', 'tenure']
Customers: 4,338

## 1) Elbow, inertia, silhouette, DB, CH (k=2..10)
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
## 2) Bootstrap stability (ARI vs full ref), k=2,3,4 (B=30, frac=0.8)
| k | mean ARI | std |
|---|---|---|
| 2 | 0.990 | 0.014 |
| 3 | 0.982 | 0.012 |
| 4 | 0.962 | 0.024 |

## 3) Interpretability: profiles + characteristics

### k=2
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| 0 | 33.75 | 8.36 | 4422.04 | 116.08 | 289.04 |
| 1 | 128.84 | 1.63 | 516.27 | 26.05 | 179.22 |

- C0 (n=1702): recency=34d, freq=8.4, monetary=£4422, distinct=116, tenure=289d
- C1 (n=2636): recency=129d, freq=1.6, monetary=£516, distinct=26, tenure=179d

### k=3
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| 0 | 44.06 | 1.65 | 562.68 | 32.96 | 74.72 |
| 1 | 29.54 | 8.98 | 4807.83 | 121.91 | 298.93 |
| 2 | 195.61 | 1.84 | 578.23 | 25.42 | 274.86 |

- C0 (n=1321): recency=44d, freq=1.6, monetary=£563, distinct=33, tenure=75d
- C1 (n=1513): recency=30d, freq=9.0, monetary=£4808, distinct=122, tenure=299d
- C2 (n=1504): recency=196d, freq=1.8, monetary=£578, distinct=25, tenure=275d

### k=4
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| 0 | 226.91 | 1.42 | 333.75 | 16.25 | 262.4 |
| 1 | 69.19 | 3.91 | 1584.66 | 69.17 | 275.6 |
| 2 | 41.97 | 1.57 | 526.27 | 31.23 | 65.33 |
| 3 | 11.9 | 13.55 | 7918.6 | 162.62 | 317.33 |

- C0 (n=1082): recency=227d, freq=1.4, monetary=£334, distinct=16, tenure=262d
- C1 (n=1352): recency=69d, freq=3.9, monetary=£1585, distinct=69, tenure=276d
- C2 (n=1176): recency=42d, freq=1.6, monetary=£526, distinct=31, tenure=65d
- C3 (n=728): recency=12d, freq=13.6, monetary=£7919, distinct=163, tenure=317d
