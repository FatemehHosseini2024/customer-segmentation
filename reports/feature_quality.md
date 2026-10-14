# Feature quality for segmentation

> Customers: 4,338 | Full-model k=4

## Results (sorted by discriminative power)

| feature | eta^2 | single-feature silhouette | best k | zero share |
|---|---|---|---|---|
| monetary | 0.671 | 0.566 | 2 | 0% |
| frequency | 0.668 | 0.781 | 5 | 0% |
| tenure | 0.629 | 0.672 | 2 | 0% |
| distinct_products | 0.501 | 0.556 | 2 | 0% |
| recency | 0.489 | 0.583 | 4 | 2% |
| has_cancellation | 0.333 | 1.000 | 2 | 64% |
| avg_order_value | 0.232 | 0.540 | 2 | 0% |

## How to read this

- **eta^2** = share of the feature's variance explained by the 4 segments. >0.14 large, 0.06-0.14 medium, <0.06 small (Cohen's conventions).
- **single-feature silhouette** = how cleanly the feature alone splits customers (best k in 2..5). >0.5 strong, 0.25-0.5 reasonable, <0.25 weak.
- **zero share** = fraction of customers with value 0 (zero-inflated features split into 'has vs has-not' groups).

## Verdict

- Strongest segment drivers (eta^2): monetary, frequency, tenure, distinct_products.
- Weakest: recency, has_cancellation, avg_order_value.
- Redundant pairs to collapse: avg_order_value/basket_size (r=0.93), recency/cadence (r=0.89).
