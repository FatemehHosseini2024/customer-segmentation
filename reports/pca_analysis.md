# PCA diagnostic for customer segmentation

PCA is a diagnostic and comparison tool here, not a performance booster. With only five features it is not meaningful dimensionality reduction; its value is structure insight (loadings), visualization (PC1 vs PC2), and a controlled test of whether clustering in a reduced space changes the segmentation.

## Decision rule (pre-registered)
- Retention rule: retain the smallest number of components whose cumulative explained variance reaches **90%** (threshold fixed before looking at results).
- Use the PCA-space clustering as the main result only if it is clearly more stable and equally interpretable; otherwise keep original-space K-Means (k=3) as the primary model and report PCA as supporting analysis.
- Silhouette, Davies–Bouldin and Calinski–Harabasz are read only within one fixed space and are **not** compared across spaces.

## Data and preprocessing
- Features: recency, frequency, monetary, distinct_products, tenure
- Preprocessing: log1p on recency/frequency/monetary/distinct_products, then StandardScaler (same preprocessed data the clustering used).
- PCA fitted on this same preprocessed data (PCA is scale-sensitive).
- Customers: 4,338

## Step 1 — Explained variance (scree plot)
| component | explained variance ratio | cumulative variance |
|---|---|---|
| PC1 | 0.5993 | 0.5993 |
| PC2 | 0.2222 | 0.8215 |
| PC3 | 0.0959 | 0.9174 |
| PC4 | 0.0556 | 0.9731 |
| PC5 | 0.0269 | 1.0000 |

Retained components: **3** (cumulative variance 91.7%, meeting the 90% rule).
PC1 is expected to absorb a large share because frequency, monetary and distinct_products are correlated — this is the redundancy already observed, now seen from a different angle.
![PCA scree plot](reports/charts/pca_scree.png)

## Step 2 — Loadings (feature x component)
Loadings = component eigenvector x sqrt(eigenvalue); magnitude shows how strongly each feature contributes to a component. Signs are arbitrary up to a flip.

| feature | PC1 | PC2 | PC3 | PC4 | PC5 |
|---|---|---|---|---|---|
| recency | -0.638 | +0.659 | +0.319 | +0.206 | +0.121 |
| frequency | +0.928 | +0.065 | -0.224 | +0.074 | +0.281 |
| monetary | +0.908 | +0.021 | +0.078 | +0.375 | -0.166 |
| distinct_products | +0.812 | -0.076 | +0.534 | -0.220 | +0.029 |
| tenure | +0.493 | +0.816 | -0.191 | -0.203 | -0.113 |

- **PC1:** strongest contributors frequency (+0.928) and monetary (+0.908).
- **PC2:** strongest contributors tenure (+0.816) and recency (+0.659).
- **PC3:** strongest contributors distinct_products (+0.534) and recency (+0.319).
- **PC4:** strongest contributors monetary (+0.375) and distinct_products (-0.220).
- **PC5:** strongest contributors frequency (+0.281) and monetary (-0.166).
- PC1 is read as overall customer value/activity (frequency, monetary, distinct_products share the same component — the M/F redundancy).
- PC2 is read as a recency vs. tenure contrast.
- Component interpretations are descriptive summaries of correlation structure, not causal findings.
![PCA loadings table above; see reports\charts\pca_scree.png for variance context.]

## Step 3 — Visualization (PC1 vs PC2)
The 2D view (PC1 vs PC2) captures **82.2%** of the total variance. Points are colored by K-Means labels computed in the **original** space; gray x marks DBSCAN noise (eps=0.94, min_samples=3, n=48) on the k=3 panel.
Expect overlapping clusters with no clear gaps — this matches the weak-to-moderate Silhouette (~0.3) and describes a continuum of customers, not a failure.
![PC1 vs PC2](reports/charts/pca_pc1_pc2.png)

## Step 4 — Clustering in the reduced space
K-Means re-run on the 3 retained components, same k range ([3, 4]), same seed (42), same n_init (10).

### Partition agreement (adjusted Rand index)
| k | ARI: original space vs PCA space |
|---|---|
| 3 | 0.980 |
| 4 | 0.976 |

### Bootstrap stability (B=30, frac=0.8), within each space
| space | k | mean ARI | std |
|---|---|---|---|
| original | 3 | 0.982 | ±0.012 |
| PCA (reduced) | 3 | 0.979 | ±0.013 |
| original | 4 | 0.963 | ±0.022 |
| PCA (reduced) | 4 | 0.967 | ±0.029 |

### Silhouette (within one fixed space only — not compared across spaces)
| space | k=3 | k=4 |
|---|---|---|
| original | 0.314 | 0.283 |
| PCA (reduced) | 0.354 | 0.326 |

### Cluster profiles mapped back to original features (original scale means)
#### Original space, k=3
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| 0 | 44.06 | 1.65 | 562.68 | 32.96 | 74.72 |
| 1 | 29.54 | 8.98 | 4807.83 | 121.91 | 298.93 |
| 2 | 195.61 | 1.84 | 578.23 | 25.42 | 274.86 |

#### PCA (reduced) space, k=3
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| 0 | 43.31 | 1.66 | 559.52 | 33.44 | 75.38 |
| 1 | 195.42 | 1.84 | 581.51 | 25.55 | 274.13 |
| 2 | 29.46 | 9.03 | 4841.18 | 122.1 | 299.8 |

#### Standardized deviation from global mean — original space, k=3
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| 0 | -0.47 | -0.34 | -0.17 | -0.33 | -1.25 |
| 1 | -0.62 | 0.61 | 0.31 | 0.71 | 0.65 |
| 2 | 1.04 | -0.32 | -0.16 | -0.42 | 0.45 |

#### Standardized deviation from global mean — PCA space, k=3
| cluster | recency | frequency | monetary | distinct_products | tenure |
|---|---|---|---|---|---|
| 0 | -0.48 | -0.34 | -0.17 | -0.33 | -1.25 |
| 1 | 1.04 | -0.32 | -0.16 | -0.42 | 0.44 |
| 2 | -0.62 | 0.62 | 0.31 | 0.71 | 0.66 |

## Step 5 — Cautions in comparing
- Do not compare Silhouette, Davies–Bouldin or Calinski–Harabasz directly between the original and reduced spaces: Silhouette tends to rise in lower dimensions without the segmentation being better.
- Base the comparison on partition agreement (ARI), stability and interpretability; read internal metrics only within one fixed space.
- If PCA retains nearly all variance (e.g. four of five components), expect a very high ARI with the original result — a valid finding that PCA changes little because the feature set is already compact.

## Step 6 — Conclusion
- ARI between original-space and PCA-space partitions at k=3: **0.980**
- Bootstrap stability at k=3: original 0.982 vs PCA 0.979
- **Verdict: PCA is neutral.** It does not clearly improve stability or interpretability; original-space K-Means (k=3) remains the primary model and PCA is reported as supporting analysis.
- Limitations: PCA is linear, assumes variance equals importance, and with five features its main value is visualization and structural insight, not dimensionality reduction.

