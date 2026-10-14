# Decision tests for `cancellation_rate` in clustering


KMeans k=4, B=20 bootstraps, log+StandardScaler on 4,338 customers.


## 1. ARI: clustering WITH vs WITHOUT the cancellation feature

`adjusted_rand_score = **0.761**

- Moderate — the feature reshapes a meaningful share of boundaries.


## 2. Per-cluster means (k=4, with cancellation feature)

| cluster | n | recency | frequency | monetary | avg_order_value | distinct_products | has_cancel | cancel_rate | cancel_value | tenure |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1316 | 88 | 3.56 | 1468 | 527 | 64 | 0.40 | 0.129 | 138 | 277 |
| 1 | 838 | 17 | 12.46 | 7244 | 609 | 152 | 0.88 | 0.188 | 475 | 309 |
| 2 | 1129 | 41 | 1.64 | 549 | 364 | 34 | 0.14 | 0.060 | 9 | 68 |
| 3 | 1055 | 209 | 1.48 | 251 | 187 | 15 | 0.13 | 0.060 | 9 | 250 |

## 3. Is the high-cancellation cluster an RFM segment or just 'the cancelers'?

- Highest-cancellation cluster = **cluster 1** (838 customers)
- Share with ≥1 cancellation: **87.6%** (global has_cancellation mean = 35.9%)
- RFM vs global: recency -82%, frequency +192%, monetary +254%
- The cluster mixes cancelers and non-cancelers; cancellation is a sub-property, not the cluster's defining trait.

## Test 2 — stability under resampling (ARI vs full-data reference)

| version | mean pairwise ARI | std |
|---|---|---|
| **with cancellation_rate** | 0.907 | ±0.041 |
| without cancellation_rate | 0.925 | ±0.123 |
- Verdict: **stable** — adding the feature does not destabilise clustering.

## Test 3 — internal metrics

| version | silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|
| with cancel | 0.210 | 1.476 | 1,464 |
| without | 0.236 | 1.302 | 1,731 |
- *Caveat:* metrics are computed on different-dimensional spaces, so absolute comparison is indicative, not definitive.

- *Decision rule:* lower DB and higher CH are better.

## Test 1 — small-denominator sensitivity

A continuous `cancellation_rate` is noisy at low frequency (the original confound that motivated removal):
- **2,328** of 4,338 customers have < 3 regular invoices.
- With the raw continuous rate (cancellations ÷ frequency), **49** customers exceed 1 (cancelled more than purchased; e.g. 1 purchase + 2 credit notes → 2.0).
- **Fix used here:** cluster on the **binary** `has_cancellation` flag (cancellation_count > 0), which is denominator-independent — no clamping or thresholding needed.

### Re-cluster on 2,010 customers (≥3 real invoices)

- High-cancellation cluster in subset = **cluster 1** (418 customers), share with cancellation = 0.885
| RFM | high-cancel cluster | subset global | delta |
|---|---|---|---|
| recency | 8.6 | 43.3 | -80% |
| frequency | 18.2 | 7.6 | +139% |
| monetary | 11,404.5 | 3,749.1 | +204% |

- Membership overlap with full-data high-cancel cluster: **Jaccard = 0.50** (418/838 original members present in subset, 418 in subset high-cancel)
- Verdict: **artifact of low frequency** — cluster dissolves on the stable-denominator subset.


## Overall decision

- Test 1 (small-denominator): FAIL — high-cancel cluster is a low-frequency artifact (Jaccard 0.50)
- Test 2 (stability): PASS — mean pairwise ARI with=0.907 vs without=0.925
- Test 3 (metrics): FAIL — silhouette 0.210/0.236, DB 1.48/1.30, CH 1,464/1,731

**Decision: REMOVE** `has_cancellation` from clustering. Use it only to **profile** clusters afterward (e.g. tag the high-cancellation segment as 'at-risk'). Document the failed tests so the choice is not data-dredged.
