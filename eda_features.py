"""Deeper EDA on the customer-level feature table.

Analyses:
1. Feature distributions (skew, log-transform need)
2. Correlation matrix
3. Outlier analysis (Monetary/Frequency tails)
4. k-selection (elbow + silhouette) on log-transformed, scaled features

Run
    python eda_features.py

Inputs
    data/Online Retail.xlsx (via src.data_loader.load_clean)

Outputs
    reports/feature_eda_report.md
    reports/feature_corr.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_clean
from src.features import build_customer_features

REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "feature_eda_report.md"
CORR_CSV = REPORT_DIR / "feature_corr.csv"

FEATURES = [
    "recency", "frequency", "monetary",
    "distinct_products", "tenure",
]
# Dropped from clustering: basket_size (r=0.93 with avg_order_value),
# cadence (r=0.89 with recency), cancellation_rate (confounded with low
# frequency — see reports/decision_tests.md). Kept in DISPLAY_FEATURES
# for documentation / post-hoc profiling.
DROPPED_FEATURES = ["basket_size", "cadence", "cancellation_rate"]
# Display-only: non-clustering features shown in the report.
DISPLAY_FEATURES = FEATURES + DROPPED_FEATURES + ["return_rate",
                                                  "cancellation_count",
                                                  "cancellation_value"]
LOG_FEATURES = [
    "recency", "frequency", "monetary",
    "distinct_products",
]


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def describe_stats(feat: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "mean": feat[DISPLAY_FEATURES].mean(),
        "std": feat[DISPLAY_FEATURES].std(),
        "median": feat[DISPLAY_FEATURES].median(),
        "skew": feat[DISPLAY_FEATURES].skew(),
    })
    return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean()
    feat = build_customer_features(df)
    feat.to_csv(REPORT_DIR / "customer_features.csv")

    md: list[str] = []
    md.append("# Feature-level EDA — customer feature table\n")
    md.append(f"> Customers: **{len(feat):,}** | Features: **{len(FEATURES)}**\n")

    # 1. Distributions
    stats = describe_stats(feat)
    md.append("## 1. Distributions\n")
    md.append("| feature | mean | std | median | skew |")
    md.append("|---|---|---|---|---|")
    for name, row in stats.iterrows():
        md.append(f"| {name} | {row['mean']:,.2f} | {row['std']:,.2f} | "
                  f"{row['median']:,.2f} | {row['skew']:,.2f} |")
    skewed = stats[stats["skew"].abs() > 1].index.tolist()
    md.append(f"\n- **Right-skewed (|skew| > 1): {', '.join(skewed)}** — "
              "log-transform these before distance-based clustering.")
    md.append("- `tenure` is the only near-symmetric feature (skew -0.37).")
    md.append("- `return_rate` is **constant 0** — every negative-quantity line "
              "in the raw data sat on a C-invoice or a guest checkout, so no "
              "returns survive the cleaning policy. Excluded from the clustering "
              "feature set: a constant cannot separate clusters.")

    # 2. Correlation
    corr = feat[FEATURES].corr()
    corr.to_csv(CORR_CSV)
    md.append("\n## 2. Correlation matrix\n")
    md.append(f"Full matrix written to `{CORR_CSV}`. Strongest absolute pairs (|r| > 0.5):\n")
    pairs = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.5:
                pairs.append((a, b, r))
    pairs.sort(key=lambda t: -abs(t[2]))
    for a, b, r in pairs[:12]:
        md.append(f"- `{a}` ↔ `{b}`: **{r:+.2f}**")
    if not pairs:
        md.append("- None above 0.5 — features are largely independent.")

    # 3. Outliers
    md.append("\n## 3. Outliers\n")
    for f in ["monetary", "frequency"]:
        q99 = feat[f].quantile(0.99)
        top = feat[f].max()
        n_over = int((feat[f] > q99).sum())
        md.append(f"- `{f}`: p99 = {q99:,.2f}, max = {top:,.2f} "
                  f"({n_over:,} customers above p99) — wholesale/bulk buyers.")
    md.append("- Recommendation: log-transform + StandardScaler; "
              "optionally cap at p99 before clustering.")

    # 4. k-selection
    md.append("\n## 4. k-selection (elbow + silhouette)\n")
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        X = feat[FEATURES].copy()
        for f in LOG_FEATURES:
            X[f] = np.log1p(X[f].clip(lower=0))
        X = StandardScaler().fit_transform(X)

        rows = []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            rows.append((k, km.inertia_, sil))
        md.append("| k | inertia | silhouette |")
        md.append("|---|---|---|")
        best_k, best_sil = None, -1
        for k, inertia, sil in rows:
            md.append(f"| {k} | {inertia:,.0f} | {sil:.3f} |")
            if sil > best_sil:
                best_k, best_sil = k, sil
        md.append(f"\n- **Best silhouette: k = {best_k}** (score {best_sil:.3f}).")
        md.append("- Inertia keeps falling with k (expected); silhouette peaks "
                  "at the natural cluster count. Confirm with business "
                  "interpretability of the resulting segments.")
    except ImportError:
        md.append("- sklearn not installed — skipped elbow/silhouette. "
                  "Install scikit-learn to run k-selection.")

    md.append("\n## 5. Next steps\n")
    md.append("- Cluster on the log-transformed, scaled feature table "
              "(`reports/customer_features.csv`).")
    md.append("- Profile segments by mean R/F/M + cancellation_rate.")
    md.append("- `cancellation_rate` covers 1,589 of 4,371 customers (36%) — "
              "expect it to split high-value churn-risk customers from clean ones.")

    REPORT_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote report -> {REPORT_PATH}")
    print("\n--- Feature EDA summary ---")
    print(stats.round(2).to_string())
    print(f"\nSkewed features: {skewed}")
    print(f"Top correlation pairs: {[(a, b, round(r, 2)) for a, b, r in pairs[:5]]}")


if __name__ == "__main__":
    main()
