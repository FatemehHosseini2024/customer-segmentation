"""Algorithm selection and evaluation for customer segmentation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

REPORT_DIR = Path("reports")
CHART_DIR = REPORT_DIR / "charts"
REPORT_PATH = REPORT_DIR / "algorithm_selection.md"

FEATURES = ["recency", "frequency", "monetary", "distinct_products", "tenure"]
LOG_FEATURES = ["recency", "frequency", "monetary", "distinct_products"]
K_RANGE = range(2, 11)
RANDOM_STATE = 42
B = 30
FRAC = 0.8

def transform(df, cols):
    X = df[cols].copy()
    for f in LOG_FEATURES:
        if f in cols:
            X[f] = np.log1p(X[f].clip(lower=0))
    return StandardScaler().fit_transform(X[cols])

def cluster(X, k=K_RANGE.start if False else 4):
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    feat = pd.read_csv(REPORT_DIR / 'customer_features.csv', index_col='CustomerID')

    # Elbow + metrics
    lines = []
    lines.append('# Algorithm selection: K-Means')
    lines.append(f'\nFeatures: {FEATURES}')
    lines.append(f'Customers: {len(feat):,}\n')
    lines.append('## 1) Elbow, inertia, silhouette, DB, CH (k=2..10)')
    lines.append('| k | inertia | silhouette | Davies-Bouldin | Calinski-Harabasz |')
    lines.append('|---|---|---|---|---|')
    inertias, sils = [], []
    ks_list = list(K_RANGE)
    for k in ks_list:
        X = transform(feat, FEATURES)
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        lab = km.fit_predict(X)
        Xk = X
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xk, lab))
        db = davies_bouldin_score(Xk, lab)
        ch = calinski_harabasz_score(Xk, lab)
        lines.append(f'| {k} | {km.inertia_:,.0f} | {sils[-1]:.3f} | {db:.3f} | {ch:,.0f} |')
    Path(REPORT_PATH).write_text('\n'.join(lines), encoding='utf-8')
    print('wrote')

if __name__ == "__main__":
    main()
