# Final Report: Customer Segmentation

## Problem and Data

- Dataset: UCI Online Retail (customer-level segmentation from transaction data).
- Customers after cleaning: **4,338**.
- Full feature set engineered per customer: `recency`, `frequency`, `monetary`, `distinct_products`, `tenure`, plus descriptive `has_cancellation`, `avg_order_value`, etc.

Cleaning steps (from `src/data_loader.py`): invoice-based CustomerID propagation where structure allows, Description backfill from StockCode, drop exact duplicates, drop rows with missing/unattributable CustomerID, drop non-positive UnitPrice, normalize StockCodes, flag non-product codes.

## Features: What was kept, removed, and evidence

**Primary features (for clustering):** `recency`, `frequency`, `monetary`, `distinct_products`, `tenure`.  

**Not used in the primary clustering space:** `has_cancellation` and `avg_order_value` (AOV). The decision tests (`reports/decision_tests.md`) show `has_cancellation` as a binary flag passes stability tests but the continuous rate suffers from small-denominator confounds; the binary flag was retained for post-hoc profiling/interpretation rather than driving the core partition. AOV is highly redundant with monetary/frequency. The feature quality analysis (`reports/feature_quality.md`) also highlights redundancy patterns (e.g. avg_order_value/basket_size).

## Algorithm comparison and k selection

- K-Means (n_init=10, random_state=42) vs Agglomerative (ward/average/complete) vs DBSCAN (eps × min_samples grid). Preprocessing: log1p on skewed features (`recency`, `frequency`, `monetary`, `distinct_products`) + StandardScaler.
- Metrics: inertia, silhouette, Davies–Bouldin, Calinski–Harabasz, bootstrap ARI stability (B=30, frac=0.8).
- Decision rule (written in advance): prefer solution with min cluster ≥ 5% of customers where feasible, stability ≥ 0.85, and top-2 on at least two internal metrics; avoid declaring a single "winner" in advance. Report k=2 as coarse split alongside others.
- K-Means: best k=3 by the pre-set rule (stability 0.982, silhouette 0.314, all clusters reasonably sized). Agglomerative (Ward) gives lower ARI vs K-Means at k=3 (ARI ~0.571), lower stability; average linkage tends to form very small clusters. 
- Chosen primary: **K-Means k=3** on original preprocessed space. k=4 kept as alternative.

## PCA: results and its role

PCA is used as a diagnostic, not a performance booster. With five features, it is not meaningful dimensionality reduction; its value is structure/loadings/visualization.

- Threshold (pre-registered): retain smallest number of components reaching 90% cumulative variance → **3 components** retained (91.7% cumulative). 
- Explained variance: PC1 59.9%, PC2 22.2%, PC3 9.6%; PC1+PC2 capture 82.2% of variance.
- Loadings (see `reports/pca_analysis.md`): PC1 captures overall value/activity with frequency, monetary, distinct_products sharing the component (the M/F redundancy observed earlier); PC2 contrasts recency vs. tenure.
- **ARI between original-space and PCA-space K-Means partitions (k=3): 0.980**. This high agreement means PCA changes little; the original feature space is sufficient.
- Bootstrap stability (k=3): original 0.982 vs PCA (reduced) 0.979 (nearly identical).
- PC1 vs PC2 plot shows overlapping clusters with no clear gaps — consistent with Silhouette ~0.3 and a continuum of customers rather than well-separated spherical clusters.
- **Conclusion:** PCA is neutral; we keep **original-space K-Means (k=3)** as the primary model. PCA is reported as supporting analysis.

## Cluster profiles and labels (3 layers: data, interpretation, suggestion)

**Primary (K-Means k=3, original space):** See `reports/final_evaluation.md` for full details.

| cluster | n | % | recency (mean/med) | frequency (mean/med) | monetary (mean/med) | distinct_products (mean/med) | tenure (mean/med) | has_cancellation (mean) | avg_order_value (mean/med) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1321 | 30.5% | 44.1/36.0 | 1.65/1.0 | 562.7/393.5 | 33.0/25.0 | 74.7/64.0 | 0.19 | 367.9/270.8 |
| 1 | 1513 | 34.9% | 29.5/17.0 | 8.98/6.0 | 4807.8/2241.1 | 121.9/94.0 | 298.9/318.0 | 0.64 | 528.5/352.2 |
| 2 | 1504 | 34.7% | 195.6/196.0 | 1.84/2.0 | 578.2/370.2 | 25.4/20.0 | 274.9/277.0 | 0.23 | 349.8/226.5 |

Interpretive marketing labels (interpretation, not findings):
- Cluster 1 (n=1513): High-value active/loyal (low recency, high frequency/monetary, long tenure).
- Cluster 0 (n=1321): Mid-value recent (moderate recency/frequency/monetary, short tenure).
- Cluster 2 (n=1504): At-risk/inactive (high recency, low frequency/monetary, long tenure).

Standardized deviations from global mean (original space, k=3): Cluster 0 shows negative recency/frequency/distinct_products/tenure; Cluster 1 shows positive frequency/monetary/distinct_products/tenure with lower recency; Cluster 2 shows strong positive recency and low activity.

Silhouette (k=3): mean 0.314; per-cluster 0.304, 0.328, 0.310; share Silhouette < 0.05: 5.9%; < 0: 2.1%.

Bootstrap stability: monetary cluster-mean spread std across resamples B=30 (frac=0.8) = 0.870.

## Ward linkage (k=3-8) and ARI vs K-Means
- Ward k=3 gives silhouette 0.260, stability 0.369, ARI vs K-Means k=3 = 0.571. 
- Agreement is moderate, not strong; Ward tends to form more even-sized clusters but internal metrics/stability differ.
- Full Ward metrics k=2-10 are in `reports/algorithm_selection.md`.

## DBSCAN: outlier and wholesale-customer detection
DBSCAN (eps=0.94, min_samples=3): n_clusters=3, **n_noise=49**. Noise points are high-value/very high activity (mean recency low, frequency/monetary/distinct_products very high). Two small clusters (1 and 2) also show extremely high monetary/frequency. 
- DBSCAN best used to flag outliers/wholesale customers, not as the general segmentation method.
- Profiles for noise and clusters in `reports/algorithm_selection.md` (section 10).

## Limitations
- Weak-to-moderate cluster structure (Silhouette ~0.3); overlapping clusters reflect a continuum rather than discrete, well-separated groups.
- Moderate agreement between K-Means and Ward (ARI 0.571 at k=3).
- Gross Monetary includes effects of returns/cancellations in the aggregation (not netted to a "pure sales" monetary in the same way across all cases); cancellation behavior is used for post-hoc profiling only.
- Influence of wholesale customers/outliers is visible; DBSCAN identifies them separately.
- Evaluation without ground-truth labels; results rely on internal metrics, stability, and interpretability.

## Reproducibility
- Fixed random seeds: RANDOM_STATE=42, BOOTSTRAP_SEED=0. n_init=10 for K-Means. Bootstrap B=30, frac=0.8.
- Python 3.12.4; key libraries: scikit-learn 1.7.2, pandas, numpy, matplotlib.
- End-to-end pipeline: `src/data_loader.py` → `src/features.py` → `feature_quality.py` → `feature_evaluation.py` → `algorithm_selection.py` → `make_charts.py` → `decision_tests.py` → `final_evaluation.py` → `pca_analysis.py`. Generated reports/charts in `reports/` and `reports/charts/`.
- All outputs committed in repo history.
