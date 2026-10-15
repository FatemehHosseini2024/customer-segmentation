# Feature evaluation report (no ground truth)
Customers: 4,338 | k=4

## 1) Correlation
- frequency <-> monetary: r=0.808
- monetary <-> avg_order_value: r=0.709
- monetary <-> distinct_products: r=0.691
- frequency <-> distinct_products: r=0.621
- recency <-> frequency: r=0.571
- frequency <-> tenure: r=0.507
- Redundancy: frequency-monetary strongest; monetary correlates with avg_order_value/distinct_products.
## 2) Zero-inflation / variance (raw)
- recency: var(raw)=10002.8339, zero=0.021
- frequency: var(raw)=59.2592, zero=0.000
- monetary: var(raw)=80734362.1066, zero=0.000
- avg_order_value: var(raw)=3227453.0052, zero=0.000
- distinct_products: var(raw)=7274.1237, zero=0.001
- tenure: var(raw)=13897.2200, zero=0.001

## 3) Leave-one-out ablation (vs full)
| drop | ARI | sil | DB | CH |
|---|---|---|---|---|
| recency | 0.466 | 0.258 | 1.202 | 2,018 |
| frequency | 0.518 | 0.246 | 1.230 | 1,658 |
| monetary | 0.633 | 0.239 | 1.273 | 1,616 |
| avg_order_value | 0.724 | 0.283 | 1.141 | 2,368 |
| distinct_products | 0.757 | 0.262 | 1.220 | 1,864 |
| tenure | 0.342 | 0.236 | 1.302 | 2,007 |

## 4) Stability (bootstrap ARI)
| feature set | mean ARI | std |
|---|---|---|
| all | 0.919 | 0.087 |
| -recency | 0.966 | 0.023 |
| -frequency | 0.961 | 0.038 |
| -monetary | 0.911 | 0.085 |
| -avg_order_value | 0.961 | 0.021 |
| -distinct_products | 0.903 | 0.108 |
| -tenure | 0.918 | 0.061 |
