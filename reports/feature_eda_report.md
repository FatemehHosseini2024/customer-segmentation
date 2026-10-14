# Feature-level EDA — customer feature table

> Customers: **4,338** | Features: **7**

## 1. Distributions

| feature | mean | std | median | skew |
|---|---|---|---|---|
| recency | 91.54 | 100.01 | 50.00 | 1.25 |
| frequency | 4.27 | 7.70 | 2.00 | 12.07 |
| monetary | 2,048.69 | 8,985.23 | 668.57 | 19.34 |
| avg_order_value | 417.65 | 1,796.51 | 291.94 | 41.69 |
| distinct_products | 61.37 | 85.29 | 35.00 | 6.92 |
| has_cancellation | 0.36 | 0.48 | 0.00 | 0.59 |
| tenure | 222.31 | 117.89 | 248.00 | -0.37 |
| basket_size | 252.46 | 1,312.77 | 161.00 | 48.01 |
| cadence | 98.02 | 87.59 | 66.71 | 1.46 |
| cancellation_rate | 0.11 | 0.17 | 0.00 | 1.40 |
| return_rate | 0.00 | 0.00 | 0.00 | 0.00 |
| cancellation_count | 0.83 | 2.15 | 0.00 | 8.83 |
| cancellation_value | 137.92 | 2,957.92 | 0.00 | 47.47 |

- **Right-skewed (|skew| > 1): recency, frequency, monetary, avg_order_value, distinct_products, basket_size, cadence, cancellation_rate, cancellation_count, cancellation_value** — log-transform these before distance-based clustering.
- `tenure` is the only near-symmetric feature (skew -0.37).
- `return_rate` is **constant 0** — every negative-quantity line in the raw data sat on a C-invoice or a guest checkout, so no returns survive the cleaning policy. Excluded from the clustering feature set: a constant cannot separate clusters.

## 2. Correlation matrix

Full matrix written to `reports\feature_corr.csv`. Strongest absolute pairs (|r| > 0.5):

- `frequency` ↔ `distinct_products`: **+0.69**
- `frequency` ↔ `monetary`: **+0.55**

## 3. Outliers

- `monetary`: p99 = 19,780.49, max = 280,206.02 (44 customers above p99) — wholesale/bulk buyers.
- `frequency`: p99 = 30.00, max = 209.00 (42 customers above p99) — wholesale/bulk buyers.
- `avg_order_value`: p99 = 2,031.16, max = 84,236.25 (44 customers above p99) — wholesale/bulk buyers.
- Recommendation: log-transform + StandardScaler; optionally cap at p99 before clustering.

## 4. k-selection (elbow + silhouette)

| k | inertia | silhouette |
|---|---|---|
| 2 | 20,040 | 0.305 |
| 3 | 17,144 | 0.239 |
| 4 | 15,081 | 0.210 |
| 5 | 13,310 | 0.236 |
| 6 | 12,237 | 0.221 |
| 7 | 11,460 | 0.216 |
| 8 | 10,712 | 0.220 |
| 9 | 10,170 | 0.216 |
| 10 | 9,669 | 0.214 |

- **Best silhouette: k = 2** (score 0.305).
- Inertia keeps falling with k (expected); silhouette peaks at the natural cluster count. Confirm with business interpretability of the resulting segments.

## 5. Next steps

- Cluster on the log-transformed, scaled feature table (`reports/customer_features.csv`).
- Profile segments by mean R/F/M + cancellation_rate.
- `cancellation_rate` covers 1,589 of 4,371 customers (36%) — expect it to split high-value churn-risk customers from clean ones.
