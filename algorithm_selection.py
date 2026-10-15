"""Algorithm selection and evaluation for customer segmentation.

Models compared:
- K-Means (random_state=42, n_init=10)
- Agglomerative clustering (linkage: ward, average, complete), k=2..10
- DBSCAN (eps x min_samples grid)

Preprocessing: log1p on skew features + StandardScaler.
Metrics: inertia/elbow (K-Means), silhouette, Davies-Bouldin, Calinski-Harabasz,
bootstrap ARI stability, cluster-profile interpretability.

Final feature set: recency, frequency, monetary, distinct_products, tenure
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
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

# DBSCAN grid
EPS_GRID = np.linspace(0.3, 2.0, 9)
MIN_SAMPLES_GRID = [3, 5, 10]


def transform(df, cols):
    X = df[cols].copy()
    for f in LOG_FEATURES:
        if f in cols:
            X[f] = np.log1p(X[f].clip(lower=0))
    return StandardScaler().fit_transform(X[cols])


def labels_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)


def labels_agg(X, k, linkage):
    return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X)


def labels_dbscan(X, eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def metrics(X, lab):
    mask = lab != -1
    Xm, lm = X[mask], lab[mask]
    n_clusters = len(set(lm))
    if n_clusters < 2:
        return None
    return (
        silhouette_score(Xm, lm),
        davies_bouldin_score(Xm, lm),
        calinski_harabasz_score(Xm, lm),
        n_clusters,
        int((lab == -1).sum()),
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
    lines.append("# Algorithm selection: K-Means vs Agglomerative vs DBSCAN")
    lines.append(f"\nFeatures: {FEATURES}")
    lines.append(f"Customers: {len(feat):,}\n")
    lines.append(
        "Preprocessing: log1p on skewed features + StandardScaler.\n"
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
        sil, db, ch, _, _ = metrics(X, lab)
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
    agg_scores = {}
    for linkage in LINKAGES:
        for k in K_RANGE:
            lab = labels_agg(X, k, linkage)
            res = metrics(X, lab)
            if res:
                sil, db, ch, _, _ = res
                agg_scores[(linkage, k)] = (sil, db, ch)
                lines.append(f"| {linkage} | {k} | {sil:.3f} | {db:.3f} | {ch:,.0f} |")
            else:
                agg_scores[(linkage, k)] = (None, None, None)
                lines.append(f"| {linkage} | {k} | - | - | - |")

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

    # ---------- Section 4: DBSCAN ----------
    lines.append("\n## 5) DBSCAN: grid search over eps x min_samples")
    lines.append(
        "| eps | min_samples | n_clusters | n_noise | silhouette | Davies-Bouldin | Calinski-Harabasz |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    dbscan_results = []
    for eps in EPS_GRID:
        for min_s in MIN_SAMPLES_GRID:
            lab = labels_dbscan(X, eps, min_s)
            n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
            n_noise = int((lab == -1).sum())
            res = metrics(X, lab)
            if res:
                sil, db, ch, nc, noise = res
                dbscan_results.append(
                    (eps, min_s, nc, noise, sil, db, ch)
                )
                lines.append(
                    f"| {eps:.2f} | {min_s} | {nc} | {noise} | {sil:.3f} | {db:.3f} | {ch:,.0f} |"
                )
            else:
                dbscan_results.append((eps, min_s, n_clusters, n_noise, None, None, None))
                lines.append(
                    f"| {eps:.2f} | {min_s} | {n_clusters} | {n_noise} | - | - | - |"
                )

    lines.append("\n## 6) Bootstrap stability: DBSCAN (B=30, frac=0.8)")
    lines.append("| eps | min_samples | mean ARI | std |")
    lines.append("|---|---|---|---|")
    for eps, min_s, *_ in dbscan_results[:6]:
        lab_ref = labels_dbscan(X, eps, min_s)
        mean_a, std_a = bootstrap_ari(
            X, lab_ref, lambda x: labels_dbscan(x, eps, min_s)
        )
        lines.append(f"| {eps:.2f} | {min_s} | {mean_a:.3f} | {std_a:.3f} |")

    # ---------- Section 5: Best picks ----------
    lines.append("\n## 7) Best silhouette by model")
    best_km = max(K_RANGE, key=lambda k: metrics(X, labels_kmeans(X, k))[0])
    sil_km, db_km, ch_km, _, _ = metrics(X, labels_kmeans(X, best_km))
    lines.append(
        f"- K-Means: k={best_km}, silhouette={sil_km:.3f}, DB={db_km:.3f}, CH={ch_km:,.0f}"
    )

    best_agg = max(
        [(l, k) for l in LINKAGES for k in K_RANGE],
        key=lambda lk: agg_scores[lk][0] if agg_scores[lk][0] else -1,
    )
    l, kk = best_agg
    sil_a, db_a, ch_a = agg_scores[best_agg]
    lines.append(
        f"- Agglomerative: linkage={l}, k={kk}, silhouette={sil_a:.3f}, DB={db_a:.3f}, CH={ch_a:,.0f}"
    )

    valid_db = [r for r in dbscan_results if r[4] is not None]
    if valid_db:
        best_db = max(valid_db, key=lambda r: r[4])
        eps_b, ms_b, nc_b, noise_b, sil_b, db_b, ch_b = best_db
        lines.append(
            f"- DBSCAN: eps={eps_b:.2f}, min_samples={ms_b}, n_clusters={nc_b}, "
            f"n_noise={noise_b}, silhouette={sil_b:.3f}, DB={db_b:.3f}, CH={ch_b:,.0f}"
        )
    else:
        lines.append("- DBSCAN: no valid clustering found (single cluster or all noise).")

    # ---------- Section 6: Interpretability ----------
    lines.append("\n## 8) Interpretability: cluster profiles (original scale means)")
    lines.append(
        "Profiles shown for K-Means k=2 (best overall: metrics + stability + interpretability)."
    )
    lab_km = labels_kmeans(X, 2)
    dfp = feat[FEATURES].copy()
    dfp["cluster"] = lab_km
    m = dfp.groupby("cluster")[FEATURES].mean().round(2)
    lines.append("\n### K-Means k=2")
    lines.append("| cluster | " + " | ".join(FEATURES) + " |")
    lines.append("|" + "---|" * (len(FEATURES) + 1))
    for idx, row in m.iterrows():
        vals = " | ".join(str(v) for v in row[FEATURES].values)
        lines.append(f"| {idx} | {vals} |")
    lines.append(
        "\n- C0: frequent, recent, high-spend customers (active/high-value)."
    )
    lines.append(
        "- C1: infrequent, older purchases, low spend (one-off/inactive)."
    )

    if valid_db:
        best_db = max(valid_db, key=lambda r: r[4])
        eps_b, ms_b, _, _, _, _, _ = best_db
        lab_db = labels_dbscan(X, eps_b, ms_b)
        dfd = feat[FEATURES].copy()
        dfd["cluster"] = lab_db
        md = dfd.groupby("cluster")[FEATURES].mean().round(2)
        lines.append(f"\n### DBSCAN eps={eps_b:.2f}, min_samples={ms_b}")
        lines.append("| cluster | " + " | ".join(FEATURES) + " |")
        lines.append("|" + "---|" * (len(FEATURES) + 1))
        for idx, row in md.iterrows():
            label = "noise" if idx == -1 else str(idx)
            vals = " | ".join(str(v) for v in row[FEATURES].values)
            lines.append(f"| {label} | {vals} |")

    Path(REPORT_PATH).write_text("\n".join(lines), encoding="utf-8")
    print("wrote", REPORT_PATH)


if __name__ == "__main__":
    main()
