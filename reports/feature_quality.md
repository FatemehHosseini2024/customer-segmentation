# Feature quality for segmentation

> Customers: 4,338 | Full-model k=4

## Results (sorted by discriminative power)

| feature | eta^2 | single-feature silhouette | best k | zero share |
|---|---|---|---|---|
| frequency | 0.723 | 0.781 | 5 | 0% |
| tenure | 0.682 | 0.672 | 2 | 0% |
| monetary | 0.619 | 0.566 | 2 | 0% |
| recency | 0.576 | 0.583 | 4 | 2% |
| distinct_products | 0.505 | 0.556 | 2 | 0% |

## How to read this

- **eta^2** = share of the feature's variance explained by the 4 segments. >0.14 large, 0.06-0.14 medium, <0.06 small (Cohen's conventions).
- **single-feature silhouette** = how cleanly the feature alone splits customers (best k in 2..5). >0.5 strong, 0.25-0.5 reasonable, <0.25 weak.
- **zero share** = fraction of customers with value 0 (zero-inflated features split into 'has vs has-not' groups).

## Verdict

- Strongest segment drivers (eta^2): frequency, tenure, monetary, recency.
- Weakest: monetary, recency, distinct_products.
- Redundant pairs to collapse: avg_order_value/basket_size (r=0.93), recency/cadence (r=0.89).
