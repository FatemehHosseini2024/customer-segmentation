"""PCA diagnostic for customer segmentation.

PCA is a diagnostic and comparison tool here, NOT a performance booster. With
only five features it is not meaningful dimensionality reduction; its value is
structure insight (loadings), visualization (PC1 vs PC2), and a controlled test
of whether clustering in a reduced space changes the segmentation.

Retention rule (pre-registered BEFORE looking at results): retain the smallest
number of components whose cumulative explained variance reaches 90%.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_DIR = Path("reports")
CHART_DIR = REPORT_DIR / "charts"
REPORT_PATH = REPORT_DIR / "pca_analysis.md"

FEATURES = ["recency", "frequency", "monetary", "distinct_products", "tenure"]
LOG_FEATURES = ["recency", "frequency", "monetary", "distinct_products"]
K_VALUES = [3, 4]
RANDOM_STATE = 42
N_INIT = 10
B = 30
FRAC = 0.8
BOOTSTRAP_SEED = 0
VAR_THRESHOLD = 0.90  # pre-registered retention rule (90%, not 95%)

DBSCAN_EPS = 0.94
DBSCAN_MIN_SAMPLES = 3


def transform(feat, cols):
    X = feat[cols].copy()
    for f in LOG_FEATURES:
        if f in cols:
            X[f] = np.log1p(X[f].clip(lower=0))
    return StandardScaler().fit_transform(X[cols])


def kmeans_labels(X, k):
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit_predict(X)


def bootstrap_ari(X, ref_labels, model_fn, B=B, frac=FRAC, seed=BOOTSTRAP_SEED):
    """Mean ARI of each bootstrap clustering vs the full-data reference,
    restricted to the subsample (within one fixed space)."""
    rng = np.random.RandomState(seed)
    n = len(X)
    m = int(frac * n)
    aris = []
    for _ in range(B):
        idx = rng.choice(n, size=m, replace=False)
        lab_b = model_fn(X[idx])
        ref = ref_labels[idx]
        if len(set(lab_b)) > 1 and len(set(ref)) > 1:
            aris.append(adjusted_rand_score(ref, lab_b))
    return (float(np.mean(aris)) if aris else 0.0,
            float(np.std(aris)) if aris else 0.0)


def profile_table(feat, labels, features):
    dfp = feat[features].copy()
    dfp["cluster"] = labels
    return dfp.groupby("cluster")[features].mean().round(2)


def std_deviation_table(feat, labels, features):
    dfp = feat[features].copy()
    dfp["cluster"] = labels
    means = dfp.groupby("cluster")[features].mean()
    glob = feat[features].mean()
    std = feat[features].std()
    return ((means - glob) / std).round(2)


def df_to_md(df):
    lines = ["| cluster | " + " | ".join(df.columns) + " |",
             "|" + "---|" * (len(df.columns) + 1)]
    for idx, row in df.iterrows():
        vals = " | ".join(str(v) for v in row.values)
        lines.append(f"| {idx} | {vals} |")
    return "\n".join(lines)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(REPORT_DIR / "customer_features.csv", index_col="CustomerID")
    X = transform(feat, FEATURES)
    n_customers = len(feat)

    # ---------- Step 1: PCA with all components ----------
    pca = PCA(n_components=len(FEATURES), random_state=RANDOM_STATE)
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_retained = int(np.searchsorted(cum, VAR_THRESHOLD) + 1)
    n_retained = min(n_retained, len(FEATURES))
    X_reduced = pca.transform(X)[:, :n_retained]

    # Scree plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comps = np.arange(1, len(FEATURES) + 1)
    ax.bar(comps, evr, width=0.6, label="explained variance ratio", color="#4C72B0")
    ax.plot(comps, cum, "o-", label="cumulative", color="#C0392B")
    ax.axhline(VAR_THRESHOLD, color="#2E86C1", ls="--",
               label=f"retention threshold {VAR_THRESHOLD:.0%}")
    ax.axvline(n_retained, color="#2E86C1", ls=":", label=f"retain {n_retained}")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title("PCA scree plot (log1p + standardized features)")
    ax.set_xticks(comps)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "pca_scree.png", dpi=110)
    plt.close(fig)

    # ---------- Step 2: Loadings ----------
    # Loading = component eigenvector * sqrt(eigenvalue): feature x component
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_df = pd.DataFrame(loadings, index=FEATURES,
                              columns=[f"PC{i+1}" for i in range(len(FEATURES))])

    # ---------- Cluster labels ----------
    labels_orig = {k: kmeans_labels(X, k) for k in K_VALUES}
    labels_red = {k: kmeans_labels(X_reduced, k) for k in K_VALUES}

    # DBSCAN noise overlay (original space) for visualization
    try:
        lab_db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X)
        noise_mask = lab_db == -1
        n_noise = int(noise_mask.sum())
    except Exception:
        noise_mask = np.zeros(n_customers, dtype=bool)
        n_noise = 0

    # ---------- Step 3: Visualization ----------
    pca2 = pca.transform(X)[:, :2]
    var2 = cum[1]  # variance captured by PC1 + PC2

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True, sharey=True)
    for ax, k in zip(axes, K_VALUES):
        lab = labels_orig[k]
        ax.scatter(pca2[:, 0], pca2[:, 1], c=lab, cmap="tab10",
                   s=12, alpha=0.7, edgecolors="none", label=f"k={k}")
        if k == 3 and n_noise:
            ax.scatter(pca2[noise_mask, 0], pca2[noise_mask, 1],
                       marker="x", c="gray", s=22, linewidths=0.8,
                       label=f"DBSCAN noise (n={n_noise})")
        ax.set_xlabel(f"PC1 ({evr[0]:.1%} of variance)")
        ax.set_ylabel(f"PC2 ({evr[1]:.1%} of variance)")
        ax.set_title(f"K-Means k={k} on PC1 vs PC2\n"
                     f"(2D view captures {var2:.1%} of total variance)")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "pca_pc1_pc2.png", dpi=110)
    plt.close(fig)

    # ---------- Step 4: Clustering in reduced space ----------
    ari_k3 = adjusted_rand_score(labels_orig[3], labels_red[3])
    ari_k4 = adjusted_rand_score(labels_orig[4], labels_red[4])

    stab_orig_k3 = bootstrap_ari(X, labels_orig[3], lambda x: kmeans_labels(x, 3))
    stab_red_k3 = bootstrap_ari(X_reduced, labels_red[3], lambda x: kmeans_labels(x, 3))
    stab_orig_k4 = bootstrap_ari(X, labels_orig[4], lambda x: kmeans_labels(x, 4))
    stab_red_k4 = bootstrap_ari(X_reduced, labels_red[4], lambda x: kmeans_labels(x, 4))

    sil_orig_k3 = silhouette_score(X, labels_orig[3])
    sil_red_k3 = silhouette_score(X_reduced, labels_red[3])
    sil_orig_k4 = silhouette_score(X, labels_orig[4])
    sil_red_k4 = silhouette_score(X_reduced, labels_red[4])

    profiles_orig_k3 = profile_table(feat, labels_orig[3], FEATURES)
    profiles_red_k3 = profile_table(feat, labels_red[3], FEATURES)
    z_orig_k3 = std_deviation_table(feat, labels_orig[3], FEATURES)
    z_red_k3 = std_deviation_table(feat, labels_red[3], FEATURES)

    # ---------- Build report ----------
    md = []
    md.append("# PCA diagnostic for customer segmentation")
    md.append("")
    md.append("PCA is a diagnostic and comparison tool here, not a performance booster. "
              "With only five features it is not meaningful dimensionality reduction; its value "
              "is structure insight (loadings), visualization (PC1 vs PC2), and a controlled test "
              "of whether clustering in a reduced space changes the segmentation.")
    md.append("")
    md.append("## Decision rule (pre-registered)")
    md.append("- Retention rule: retain the smallest number of components whose cumulative "
              "explained variance reaches **90%** (threshold fixed before looking at results).")
    md.append("- Use the PCA-space clustering as the main result only if it is clearly more "
              "stable and equally interpretable; otherwise keep original-space K-Means (k=3) as "
              "the primary model and report PCA as supporting analysis.")
    md.append("- Silhouette, Davies–Bouldin and Calinski–Harabasz are read only within one "
              "fixed space and are **not** compared across spaces.")
    md.append("")
    md.append("## Data and preprocessing")
    md.append(f"- Features: {', '.join(FEATURES)}")
    md.append("- Preprocessing: log1p on recency/frequency/monetary/distinct_products, then "
              "StandardScaler (same preprocessed data the clustering used).")
    md.append(f"- PCA fitted on this same preprocessed data (PCA is scale-sensitive).")
    md.append(f"- Customers: {n_customers:,}")
    md.append("")

    # Step 1
    md.append("## Step 1 — Explained variance (scree plot)")
    md.append("| component | explained variance ratio | cumulative variance |")
    md.append("|---|---|---|")
    for i in range(len(FEATURES)):
        md.append(f"| PC{i+1} | {evr[i]:.4f} | {cum[i]:.4f} |")
    md.append("")
    md.append(f"Retained components: **{n_retained}** (cumulative variance "
              f"{cum[n_retained-1]:.1%}, meeting the {VAR_THRESHOLD:.0%} rule).")
    md.append("PC1 is expected to absorb a large share because frequency, monetary and "
              "distinct_products are correlated — this is the redundancy already observed, "
              "now seen from a different angle.")
    md.append(f"![PCA scree plot]({(CHART_DIR / 'pca_scree.png').as_posix()})")
    md.append("")

    # Step 2
    md.append("## Step 2 — Loadings (feature x component)")
    md.append("Loadings = component eigenvector x sqrt(eigenvalue); magnitude shows how strongly "
              "each feature contributes to a component. Signs are arbitrary up to a flip.")
    md.append("")
    md.append("| feature | " + " | ".join(loading_df.columns) + " |")
    md.append("|" + "---|" * (len(loading_df.columns) + 1))
    for f, row in loading_df.iterrows():
        vals = " | ".join(f"{v:+.3f}" for v in row.values)
        md.append(f"| {f} | {vals} |")
    md.append("")
    # dynamic interpretation
    for pc in loading_df.columns:
        s = loading_df[pc].abs().sort_values(ascending=False)
        top = s.index[:2].tolist()
        md.append(f"- **{pc}:** strongest contributors {top[0]} ({loading_df.loc[top[0], pc]:+.3f}) "
                  f"and {top[1]} ({loading_df.loc[top[1], pc]:+.3f}).")
    md.append("- PC1 is read as overall customer value/activity (frequency, monetary, "
              "distinct_products share the same component — the M/F redundancy).")
    md.append("- PC2 is read as a recency vs. tenure contrast.")
    md.append("- Component interpretations are descriptive summaries of correlation structure, "
              "not causal findings.")
    md.append(f"![PCA loadings table above; see {CHART_DIR / 'pca_scree.png'} for variance context.]")
    md.append("")

    # Step 3
    md.append("## Step 3 — Visualization (PC1 vs PC2)")
    md.append(f"The 2D view (PC1 vs PC2) captures **{var2:.1%}** of the total variance. "
              f"Points are colored by K-Means labels computed in the **original** space; "
              f"gray x marks DBSCAN noise (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}, "
              f"n={n_noise}) on the k=3 panel.")
    md.append("Expect overlapping clusters with no clear gaps — this matches the weak-to-moderate "
              "Silhouette (~0.3) and describes a continuum of customers, not a failure.")
    md.append(f"![PC1 vs PC2]({(CHART_DIR / 'pca_pc1_pc2.png').as_posix()})")
    md.append("")

    # Step 4
    md.append("## Step 4 — Clustering in the reduced space")
    md.append(f"K-Means re-run on the {n_retained} retained components, same k range "
              f"({K_VALUES}), same seed ({RANDOM_STATE}), same n_init ({N_INIT}).")
    md.append("")
    md.append("### Partition agreement (adjusted Rand index)")
    md.append("| k | ARI: original space vs PCA space |")
    md.append("|---|---|")
    md.append(f"| 3 | {ari_k3:.3f} |")
    md.append(f"| 4 | {ari_k4:.3f} |")
    md.append("")
    md.append("### Bootstrap stability (B=30, frac=0.8), within each space")
    md.append("| space | k | mean ARI | std |")
    md.append("|---|---|---|---|")
    md.append(f"| original | 3 | {stab_orig_k3[0]:.3f} | ±{stab_orig_k3[1]:.3f} |")
    md.append(f"| PCA (reduced) | 3 | {stab_red_k3[0]:.3f} | ±{stab_red_k3[1]:.3f} |")
    md.append(f"| original | 4 | {stab_orig_k4[0]:.3f} | ±{stab_orig_k4[1]:.3f} |")
    md.append(f"| PCA (reduced) | 4 | {stab_red_k4[0]:.3f} | ±{stab_red_k4[1]:.3f} |")
    md.append("")
    md.append("### Silhouette (within one fixed space only — not compared across spaces)")
    md.append("| space | k=3 | k=4 |")
    md.append("|---|---|---|")
    md.append(f"| original | {sil_orig_k3:.3f} | {sil_orig_k4:.3f} |")
    md.append(f"| PCA (reduced) | {sil_red_k3:.3f} | {sil_red_k4:.3f} |")
    md.append("")
    md.append("### Cluster profiles mapped back to original features (original scale means)")
    md.append("#### Original space, k=3")
    md.append(df_to_md(profiles_orig_k3))
    md.append("")
    md.append("#### PCA (reduced) space, k=3")
    md.append(df_to_md(profiles_red_k3))
    md.append("")
    md.append("#### Standardized deviation from global mean — original space, k=3")
    md.append(df_to_md(z_orig_k3))
    md.append("")
    md.append("#### Standardized deviation from global mean — PCA space, k=3")
    md.append(df_to_md(z_red_k3))
    md.append("")

    # Step 5
    md.append("## Step 5 — Cautions in comparing")
    md.append("- Do not compare Silhouette, Davies–Bouldin or Calinski–Harabasz directly "
              "between the original and reduced spaces: Silhouette tends to rise in lower "
              "dimensions without the segmentation being better.")
    md.append("- Base the comparison on partition agreement (ARI), stability and "
              "interpretability; read internal metrics only within one fixed space.")
    md.append("- If PCA retains nearly all variance (e.g. four of five components), expect a "
              "very high ARI with the original result — a valid finding that PCA changes little "
              "because the feature set is already compact.")
    md.append("")

    # Step 6
    md.append("## Step 6 — Conclusion")
    more_stable = stab_red_k3[0] >= stab_orig_k3[0]
    uses_pca = more_stable and ari_k3 > 0.85
    md.append(f"- ARI between original-space and PCA-space partitions at k=3: **{ari_k3:.3f}**")
    md.append(f"- Bootstrap stability at k=3: original {stab_orig_k3[0]:.3f} vs PCA "
              f"{stab_red_k3[0]:.3f}")
    if uses_pca:
        md.append("- **Verdict: PCA-space clustering is neutral-to-positive.** It is at least as "
                  "stable and equally interpretable; either space is defensible.")
    else:
        md.append("- **Verdict: PCA is neutral.** It does not clearly improve stability or "
                  "interpretability; original-space K-Means (k=3) remains the primary model and "
                  "PCA is reported as supporting analysis.")
    md.append("- Limitations: PCA is linear, assumes variance equals importance, and with five "
              "features its main value is visualization and structural insight, not "
              "dimensionality reduction.")
    md.append("")

    REPORT_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote report -> {REPORT_PATH}")
    print(f"Charts -> {CHART_DIR / 'pca_scree.png'}, {CHART_DIR / 'pca_pc1_pc2.png'}")
    print(f"Retained components: {n_retained} (cumulative {cum[n_retained-1]:.1%})")
    print(f"ARI k=3 original vs PCA: {ari_k3:.3f} | k=4: {ari_k4:.3f}")
    print(f"Stability k=3: orig {stab_orig_k3[0]:.3f} vs pca {stab_red_k3[0]:.3f}")


if __name__ == "__main__":
    main()
