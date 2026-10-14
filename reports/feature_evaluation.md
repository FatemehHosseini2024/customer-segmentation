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
- has_cancellation: var(raw)=0.2301, zero=0.641
- tenure: var(raw)=13897.2200, zero=0.001

## 3) Leave-one-out ablation (vs full)
| drop | ARI | sil | DB | CH |
|---|---|---|---|---|
| recency | 0.319 | 0.250 | 1.362 | 1,682 |
| frequency | 0.616 | 0.237 | 1.409 | 1,418 |
| monetary | 0.548 | 0.232 | 1.448 | 1,368 |
| avg_order_value | 0.677 | 0.241 | 1.337 | 1,845 |
| distinct_products | 0.677 | 0.229 | 1.436 | 1,528 |
| has_cancellation | 0.761 | 0.236 | 1.302 | 1,731 |
| tenure | 0.424 | 0.262 | 1.310 | 1,789 |

## 4) Stability (bootstrap ARI vs full-data reference)
| feature set | mean ARI | std |
|---|---|---|
| all | 0.915 | 0.044 |
| -recency | 0.956 | 0.099 |
| -frequency | 0.949 | 0.037 |
| -monetary | 0.975 | 0.013 |
| -avg_order_value | 0.938 | 0.093 |
| -distinct_products | 0.757 | 0.178 |
| -has_cancellation | 0.918 | 0.107 |
| -tenure | 0.970 | 0.017 |
