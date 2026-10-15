# Comparison of 4 states (monetary vs avg_order_value)
| state | sil | DB | CH | stability_mean_ARI | stability_std |
|---|---|---|---|---|---|
| 1. +monetary +avg_order_value | 0.236 | 1.302 | 1,731 | 0.919 | 0.087 |
| 2. +monetary -avg_order_value | 0.283 | 1.141 | 2,368 | 0.963 | 0.022 |
| 3. -monetary +avg_order_value | 0.239 | 1.273 | 1,616 | 0.863 | 0.105 |
| 4. -monetary -avg_order_value | 0.304 | 1.095 | 2,487 | 0.964 | 0.024 |
## Profile interpretability (cluster means, original scale)

**2. +monetary -avg_order_value**
| cluster | recency | frequency | distinct_products | tenure | monetary |
|---|---|---|---|---|---|
| 0 | 226.91 | 1.42 | 16.25 | 262.4 | 333.75 |
| 1 | 69.19 | 3.91 | 69.17 | 275.6 | 1584.66 |
| 2 | 41.97 | 1.57 | 31.23 | 65.33 | 526.27 |
| 3 | 11.9 | 13.55 | 162.62 | 317.33 | 7918.6 |

**3. -monetary +avg_order_value**
| cluster | recency | frequency | distinct_products | tenure | avg_order_value |
|---|---|---|---|---|---|
| 0 | 187.23 | 1.61 | 10.55 | 235.77 | 149.78 |
| 1 | 38.85 | 1.79 | 37.41 | 71.06 | 462.99 |
| 2 | 145.69 | 2.63 | 50.96 | 281.41 | 545.85 |
| 3 | 17.36 | 10.76 | 137.05 | 311.63 | 429.96 |

**4. -monetary -avg_order_value**
| cluster | recency | frequency | distinct_products | tenure |
|---|---|---|---|---|
| 0 | 42.15 | 1.67 | 34.02 | 68.74 |
| 1 | 223.84 | 1.5 | 17.23 | 264.06 |
| 2 | 63.4 | 4.25 | 74.15 | 287.49 |
| 3 | 9.16 | 13.98 | 165.1 | 318.5 |

**1. +monetary +avg_order_value**
| cluster | recency | frequency | distinct_products | tenure | monetary | avg_order_value |
|---|---|---|---|---|---|---|
| 0 | 92.73 | 3.45 | 62.6 | 276.46 | 1452.01 | 533.16 |
| 1 | 41.18 | 1.64 | 33.78 | 66.65 | 549.63 | 365.53 |
| 2 | 206.22 | 1.5 | 14.44 | 248.46 | 245.78 | 180.98 |
| 3 | 13.97 | 12.53 | 154.75 | 312.2 | 7237.85 | 600.02 |
