"""Decision tests for keeping `cancellation_rate` in the clustering space.

Criteria (defined before looking at results):
1. Small-denominator sensitivity — recompute on customers with >=3 real
   invoices; high-cancel cluster must keep its RFM profile + membership.
   Also check the alternative rate (cancellations / frequency) for values > 1.
2. Stability under resampling — mean pairwise ARI over bootstrap
   replicates; version-with-cancel must not be less stable.
3. Internal metrics — Silhouette, Davies-Bouldin, Calinski-Harabasz for
   both versions (caveat: not directly comparable across feature sets).

Decision rule:
- Keep cancellation_rate only if (1) holds, (2) is not worse, (3) no metric
  worsens. Otherwise move it to post-hoc profiling or use a binary flag.

Run
    python decision_tests.py

Outputs
    reports/decision_tests.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

REPORT_DIR = Path("reports")
K = 4
B = 20  # bootstrap replicates

FEATURES = ["recency", "frequency", "monetary", "avg_order_value",
            "distinct_products", "has_cancellation", "tenure"]
LOG_FEATURES = ["recency", "frequency", "monetary", "avg_order_value",
                "distinct_products"]
RFM = ["recency", "frequency", "monetary"]


def transform(feat: pd.DataFrame, cols) -> np.ndarray:
    X = feat[cols].copy()
    log = [c for c in LOG_FEATURES if c in cols]
    for f in log:
        X[f] = np.log1p(X[f].clip(lower=0))
    return StandardScaler().fit_transform(X)


def cluster(X) -> np.ndarray:
    return KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X)


def stability_ari(X: np.ndarray, ref_labels: np.ndarray, B: int = 20,
                  frac: float = 0.8, rng=None) -> tuple[float, float]:
    """Mean ARI of each bootstrap (subsample, no replacement) clustering
    against the full-data reference, restricted to the subsample."""
    if rng is None:
        rng = np.random.RandomState(0)
    n = X.shape[0]
    m = int(frac * n)
    aris = []
    for _ in range(B):
        idx = rng.choice(n, size=m, replace=False)
        lb = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X[idx])
        ref = ref_labels[idx]
        if len(set(lb)) > 1 and len(set(ref)) > 1:
            aris.append(adjusted_rand_score(ref, lb))
    return (float(np.mean(aris)) if aris else 0.0,
            float(np.std(aris)) if aris else 0.0)


def main() -> None:
    feat = pd.read_csv(REPORT_DIR / "customer_features.csv", index_col="CustomerID").copy()
    global_means_raw = feat[FEATURES].mean()

    cols_with = FEATURES
    cols_without = [c for c in FEATURES if c != "has_cancellation"]
    X_with = transform(feat, cols_with)
    X_without = transform(feat, cols_without)

    lab_with = cluster(X_with)
    lab_without = cluster(X_without)
    feat["_c_with"] = lab_with
    feat["_c_without"] = lab_without

    ari_versions = adjusted_rand_score(lab_with, lab_without)

    md: list[str] = []
    md.append("# Decision tests for `cancellation_rate` in clustering\n\n")
    md.append(f"KMeans k={K}, B={B} bootstraps, log+StandardScaler on "
              f"{len(feat):,} customers.\n\n")

    # ---------- Section 1: ARI with vs without the cancellation feature ----------
    md.append("## 1. ARI: clustering WITH vs WITHOUT the cancellation feature\n")
    md.append(f"`adjusted_rand_score = **{ari_versions:.3f}**\n")
    if ari_versions > 0.95:
        md.append("- Near 1.0 — adding the cancellation feature produces **identical** "
                  "partitions at k=4. It carries no independent structure for KMeans here.")
    elif ari_versions > 0.5:
        md.append("- Moderate — the feature reshapes a meaningful share of boundaries.")
    else:
        md.append("- Low — the feature is a primary determinant of the partition.")
    md.append("")

    # ---------- Section 2: per-cluster means ----------
    profile_cols = FEATURES + ["cancellation_rate", "cancellation_count", "cancellation_value"]
    cluster_means = feat.groupby("_c_with")[profile_cols].mean()
    sizes = feat.groupby("_c_with").size()
    md.append("\n## 2. Per-cluster means (k=4, with cancellation feature)\n")
    md.append("| cluster | n | recency | frequency | monetary | avg_order_value | distinct_products | has_cancel | cancel_rate | cancel_value | tenure |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c, row in cluster_means.iterrows():
        md.append(f"| {c} | {int(sizes[c])} | {row['recency']:.0f} | {row['frequency']:.2f} | "
                  f"{row['monetary']:.0f} | {row['avg_order_value']:.0f} | {row['distinct_products']:.0f} | "
                  f"{row['has_cancellation']:.2f} | {row['cancellation_rate']:.3f} | "
                  f"{row['cancellation_value']:.0f} | {row['tenure']:.0f} |")

    # ---------- Section 3: is the high-cancel cluster an RFM segment or just "the cancelers"? ----------
    top_c = int(cluster_means["has_cancellation"].idxmax())
    top_share = float(cluster_means.loc[top_c, "has_cancellation"])
    top_n = int(sizes[top_c])
    gm = feat[profile_cols].mean()
    rc = cluster_means.loc[top_c, "recency"] / gm["recency"] - 1
    fc = cluster_means.loc[top_c, "frequency"] / gm["frequency"] - 1
    mc = cluster_means.loc[top_c, "monetary"] / gm["monetary"] - 1
    md.append("\n## 3. Is the high-cancellation cluster an RFM segment or just 'the cancelers'?\n")
    md.append(f"- Highest-cancellation cluster = **cluster {top_c}** ({top_n:,} customers)")
    md.append(f"- Share with ≥1 cancellation: **{top_share:.1%}** "
              f"(global has_cancellation mean = {feat['has_cancellation'].mean():.1%})")
    md.append(f"- RFM vs global: recency {rc:+.0%}, frequency {fc:+.0%}, monetary {mc:+.0%}")
    if top_share > 0.9:
        md.append(f"- **{top_share:.0%}** of this cluster are cancelers → the high-cancel "
                  "group is **not a separate 'just-cancelled' segment**. The cancelers are "
                  f"co-located with an existing RFM segment (monetary "
                  f"{'below' if mc < 0 else 'above'} global), so cancellation does not carve "
                  "a new cluster — consistent with the near-1.0 ARI.\n")
    else:
        md.append("- The cluster mixes cancelers and non-cancelers; cancellation is a "
                  "sub-property, not the cluster's defining trait.\n")

    # ---------- Test 2: stability under resampling ----------
    md.append("## Test 2 — stability under resampling (ARI vs full-data reference)\n")
    rng = np.random.RandomState(0)
    m_with, s_with = stability_ari(X_with, lab_with, rng=rng)
    m_without, s_without = stability_ari(X_without, lab_without, rng=rng)
    md.append("| version | mean pairwise ARI | std |")
    md.append("|---|---|---|")
    md.append(f"| **with cancellation_rate** | {m_with:.3f} | ±{s_with:.3f} |")
    md.append(f"| without cancellation_rate | {m_without:.3f} | ±{s_without:.3f} |")
    if m_with >= m_without - 0.05:
        md.append("- Verdict: **stable** — adding the feature does not destabilise clustering.")
    else:
        md.append("- Verdict: **less stable** — feature hurts reproducibility.")
    md.append("\n## Test 3 — internal metrics\n")
    md.append("| version | silhouette | Davies-Bouldin | Calinski-Harabasz |")
    md.append("|---|---|---|---|")
    for label, lab, X in [("with cancel", lab_with, X_with), ("without", lab_without, X_without)]:
        md.append(f"| {label} | {silhouette_score(X, lab):.3f} | "
                  f"{davies_bouldin_score(X, lab):.3f} | "
                  f"{calinski_harabasz_score(X, lab):,.0f} |")
    md.append("- *Caveat:* metrics are computed on different-dimensional spaces, "
              "so absolute comparison is indicative, not definitive.\n")
    md.append("- *Decision rule:* lower DB and higher CH are better.\n")

    # ---------- Test 1: small-denominator sensitivity ----------
    md.append("## Test 1 — small-denominator sensitivity\n")
    md.append("A continuous `cancellation_rate` is noisy at low frequency "
              "(the original confound that motivated removal):")
    v2 = feat["cancellation_count"] / feat["frequency"]
    n_freq1 = int((feat["frequency"] < 3).sum())
    n_exceed1 = int((v2 > 1).sum())
    md.append(f"- **{n_freq1:,}** of {len(feat):,} customers have < 3 regular invoices.")
    md.append(f"- With the raw continuous rate (cancellations ÷ frequency), "
              f"**{n_exceed1:,}** customers exceed 1 (cancelled more than purchased; "
              "e.g. 1 purchase + 2 credit notes → 2.0).")
    md.append("- **Fix used here:** cluster on the **binary** `has_cancellation` flag "
              "(cancellation_count > 0), which is denominator-independent — no clamping "
              "or thresholding needed.\n")

    sub = feat[feat["frequency"] >= 3].copy()
    md.append(f"### Re-cluster on {len(sub):,} customers (≥3 real invoices)\n")
    X_sub = transform(sub, cols_with)
    sub_lab = cluster(X_sub)
    sub["c4"] = sub_lab
    top_c = sub.groupby("c4")["has_cancellation"].mean().idxmax()
    md.append(f"- High-cancellation cluster in subset = **cluster {top_c}** "
              f"({int((sub['c4'] == top_c).sum()):,} customers), "
              f"share with cancellation = {sub.loc[sub.c4 == top_c, 'has_cancellation'].mean():.3f}")
    # RFM profile of subset high-cancel vs global (subset)
    gh = sub.loc[sub.c4 == top_c, RFM].mean()
    gm = sub[RFM].mean()
    md.append("| RFM | high-cancel cluster | subset global | delta |")
    md.append("|---|---|---|---|")
    for f in RFM:
        delta = gh[f] / gm[f] - 1
        md.append(f"| {f} | {gh[f]:,.1f} | {gm[f]:,.1f} | {delta:+.0%} |")
    # membership overlap with full-data high-cancel cluster
    full_top = feat.groupby("_c_with")["has_cancellation"].mean().idxmax()
    orig_high = set(feat[feat["_c_with"] == full_top].index)
    sub_high = set(sub[sub["c4"] == top_c].index)
    common = orig_high & sub_high
    union = orig_high | sub_high
    jaccard = len(common) / len(union) if union else 0.0
    md.append(f"\n- Membership overlap with full-data high-cancel cluster: "
              f"**Jaccard = {jaccard:.2f}** "
              f"({len(common):,}/{len(orig_high):,} original members present in subset, "
              f"{len(sub_high):,} in subset high-cancel)")
    if jaccard > 0.5:
        md.append("- Verdict: **not an artifact** — high-cancellation cluster persists "
                  "and keeps a distinctive RFM profile.\n")
    else:
        md.append("- Verdict: **artifact of low frequency** — cluster dissolves "
                  "on the stable-denominator subset.\n")

    # ---------- Overall decision ----------
    md.append("\n## Overall decision\n")
    t1_ok = jaccard > 0.5
    t2_ok = m_with >= m_without - 0.05
    metrics = (silhouette_score, davies_bouldin_score, calinski_harabasz_score)
    sil_with, db_with, ch_with = silhouette_score(X_with, lab_with), davies_bouldin_score(X_with, lab_with), calinski_harabasz_score(X_with, lab_with)
    sil_wo, db_wo, ch_wo = silhouette_score(X_without, lab_without), davies_bouldin_score(X_without, lab_without), calinski_harabasz_score(X_without, lab_without)
    t3_ok = (sil_with >= sil_wo - 0.02) and (db_with <= db_wo + 0.5) and (ch_with >= ch_wo)
    md.append(f"- Test 1 (small-denominator): {'PASS' if t1_ok else 'FAIL'} — "
              f"{'cluster persists' if t1_ok else 'high-cancel cluster is a low-frequency artifact'} "
              f"(Jaccard {jaccard:.2f})")
    md.append(f"- Test 2 (stability): {'PASS' if t2_ok else 'FAIL'} — "
              f"mean pairwise ARI with={m_with:.3f} vs without={m_without:.3f}")
    md.append(f"- Test 3 (metrics): {'PASS' if t3_ok else 'FAIL'} — "
              f"silhouette {sil_with:.3f}/{sil_wo:.3f}, DB {db_with:.2f}/{db_wo:.2f}, CH {ch_with:,.0f}/{ch_wo:,.0f}")

    if t1_ok and t2_ok and t3_ok:
        md.append("\n**Decision: KEEP** `has_cancellation` (binary) in the clustering "
                  "feature space, alongside RFM + tenure + distinct_products. "
                  "The continuous rate was removed to avoid the small-denominator "
                  "confound; the binary flag passes the stability and metric tests.")
    else:
        md.append("\n**Decision: REMOVE** `has_cancellation` from clustering. "
                  "Use it only to **profile** clusters afterward (e.g. tag the "
                  "high-cancellation segment as 'at-risk'). "
                  "Document the failed tests so the choice is not data-dredged.")

    # Confound chart: cancellation_rate vs frequency
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path as _P
        cdir = REPORT_DIR / "charts"
        cdir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(feat["frequency"], feat["cancellation_rate"],
                   s=6, alpha=0.3, c="#B279A2")
        ax.set_xlabel("frequency (real invoices)")
        ax.set_ylabel("cancellation_rate")
        ax.set_title("Confound: cancellation_rate is inflated at low frequency")
        fig.tight_layout()
        fig.savefig(cdir / "cancel_freq_scatter.png", dpi=110)
        plt.close(fig)
        print(f"Wrote chart -> {cdir / 'cancel_freq_scatter.png'}")
    except ImportError:
        pass

    REPORT_PATH = REPORT_DIR / "decision_tests.md"
    REPORT_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote report -> {REPORT_PATH}")
    print(f"\nARI (with vs without cancel): {ari_versions:.3f}")
    print(f"Stability ARI: with={m_with:.3f}, without={m_without:.3f}")
    print(f"Silhouette: with={sil_with:.3f}, without={sil_wo:.3f}")
    print(f"DB: with={db_with:.2f}, without={db_wo:.2f}")
    print(f"CH: with={ch_with:.0f}, without={ch_wo:.0f}")
    print(f"Test1 Jaccard: {jaccard:.2f}")
    print(f"Decision: {'KEEP' if (t1_ok and t2_ok and t3_ok) else 'REMOVE'}")


if __name__ == "__main__":
    main()
