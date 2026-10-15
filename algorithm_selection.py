"""Algorithm selection and evaluation for customer segmentation.

Models compared:
- K-Means (random_state=42, n_init=10)
- Agglomerative clustering (linkage: ward, average, complete), k=2..10

Preprocessing: log1p on skew features + StandardScaler.
Metrics: inertia/elbow (K-Means), silhouette, Davies-Bouldin, Calinski-Harabasz,
bootstrap ARI stability, cluster-profile interpretability.

Final feature set: recency, frequency, monetary, distinct_products, tenure
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

REPORT_DIR = Path("reports")
CHART_DIR = REPORT_DIR / "charts"
REPORT_PATH = REPORT_DIR / "algorithm_selection.md"

FEATURES = ["recency", "frequency", "monetary", "distinct_products", "tenure"]
LOG_FEATURES = ["recency", "frequency", "monetary", "distinct_products"]
K_RANGE = list(range(2, 11))
RANDOM_STATE = 42
B = 30
FRAC = 0.8

LINKAGES = ["ward", "average", "complete"]


def transform(df, cols):
    X = df[cols].copy()
    for f in LOG_FEATURES:
        if f in cols:
            X[f] = np.log1p(X[f].clip(lower=0))
    return StandardScaler().fit_transform(X[cols])


def labels_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)


def labels_agg(X, k, linkage):
    return AgglomerativeClustering(
        n_clusters=k, linkage=linkage
    ).fit_predict(X)


def metrics(X, lab):
    return (
        silhouette_score(X, lab),
        davies_bouldin_score(X, lab),
        calinski_harabasz_score(X, lab),
    )


def bootstrap_ari(X, lab_ref, model_fn, B=B, frac=FRAC):
    rng = np.random.RandomState(0)
    n = len(X)
    m = int(frac * n)
    aris = []
    for _ in range(B):
        idx = rng.choice(n, size=m, replace=False)
        lab_b = model_fn(X[idx])
        ref = lab_ref[idx]
        if len(set(lab_b)) > 1 and len(set(ref)) > 1:
            aris.append(adjusted_rand_score(ref, lab_b))
    return (float(np.mean(aris)) if aris else 0.0, float(np.std(aris)) if aris else 0.0)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    feat = pd.read_csv(REPORT_DIR / "customer_features.csv", index_col="CustomerID")
    X = transform(feat, FEATURES)

    lines = []
    lines.append("# Algorithm selection: K-Means vs Agglomerative clustering")
    lines.append(f"\nFeatures: {FEATURES}")
    lines.append(f"Customers: {len(feat):,}\n")
    lines.append(
        "Preprocessing: log1p on skewed features + StandardScaler on the scaled data.\n"
    )

    # ---------- Section 1: K-Means ----------
    lines.append("## 1) K-Means: elbow + metrics (k=2..10)")
    lines.append(
        "| k | inertia | silhouette | Davies-Bouldin | Calinski-Harabasz |"
    )
    lines.append("|---|---|---|---|---|")
    inertias = []
    for k in K_RANGE:
        lab = labels_kmeans(X, k)
        sil, db, ch = metrics(X, lab)
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        inertias.append(km.inertia_)
        lines.append(
            f"| {k} | {km.inertia_:,.0f} | {sil:.3f} | {db:.3f} | {ch:,.0f} |"
        )

    lines.append("\n## 2) Bootstrap stability: K-Means (B=30, frac=0.8)")
    lines.append("| k | mean ARI | std |")
    lines.append("|---|---|---|")
    for k in [2, 3, 4]:
        lab_ref = labels_kmeans(X, k)
        mean_a, std_a = bootstrap_ari(X, lab_ref, lambda x: labels_kmeans(x, k))
        lines.append(f"| {k} | {mean_a:.3f} | {std_a:.3f} |")

    # ---------- Section 3: Agglomerative ----------
    lines.append("\n## 3) Agglomerative clustering: metrics by linkage (k=2..10)")
    lines.append(
        "| linkage | k | silhouette | Davies-Bouldin | Calinski-Harabasz |"
    )
    lines.append("|---|---|---|---|---|")
    agg_scores = {}  # (linkage, k) -> (sil, db, ch)
    for linkage in LINKAGES:
        for k in K_RANGE:
            lab = labels_agg(X, k, linkage)
            sil, db, ch = metrics(X, lab)
            agg_scores[(linkage, k)] = (sil, db, ch)
            lines.append(f"| {linkage} | {k} | {sil:.3f} | {db:.3f} | {ch:,.0f} |")

    lines.append("\n## 4) Bootstrap stability: Agglomerative (B=30, frac=0.8)")
    lines.append("| linkage | k | mean ARI | std |")
    lines.append("|---|---|---|---|")
    for linkage in LINKAGES:
        for k in [2, 3, 4]:
            lab_ref = labels_agg(X, k, linkage)
            mean_a, std_a = bootstrap_ari(
                X, lab_ref, lambda x: labels_agg(x, k, linkage)
            )
            lines.append(f"| {linkage} | {k} | {mean_a:.3f} | {std_a:.3f} |")

    # ---------- Section 5: Best picks ----------
    # best silhouette per model
    best_km = max(K_RANGE, key=lambda k: metrics(X, labels_kmeans(X, k))[0])
    best_agg = max(
        [(l, k) for l in LINKAGES for k in K_RANGE],
        key=lambda lk: agg_scores[lk][0],
    )
    lines.append("\n## 5) Best silhouette by model")
    lines.append(f"- K-Means: k={best_km}, silhouette={metrics(X, labels_kmeans(X, best_km))[0]:.3f}")
    l, kk = best_agg
    lines.append(f"- Agglomerative: linkage={l}, k={kk}, silhouette={agg_scores[best_agg][0]:.3f}")

    # ---------- Section 6: Interpretability ----------
    lines.append("\n## 6) Interpretability: cluster profiles (original scale means)")
    for k in [2, 3, 4]:
        lab_km = labels_kmeans(X, k)
        dfp = feat[FEATURES].copy()
        dfp["cluster"] = lab_km
        m = dfp.groupby("cluster")[FEATURES].mean().round(2)
        lines.append(f"\n### K-Means k={k}")
        lines.append("| cluster | " + " | ".join(FEATURES) + " |")
        lines.append("|" + "---|" * (len(FEATURES) + 1))
        for idx, row in m.iterrows():
            vals = " | ".join(str(v) for v in row[FEATURES].values)
            lines.append(f"| {idx} | {vals} |")

    Path(REPORT_PATH).write_text("\n".join(lines), encoding="utf-8")
    print("wrote", REPORT_PATH)


if __name__ == "__main__":
    main()
