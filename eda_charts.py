"""Chart generation for the customer feature table.

Saves PNG charts to reports/charts/ and prints histogram bins
(for inline rendering) to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CHART_DIR = Path("reports/charts")


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    feat = pd.read_csv("reports/customer_features.csv", index_col="CustomerID")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    # 1. Log-monetary histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    log_mon = np.log1p(feat["monetary"])
    ax.hist(log_mon, bins=40, color="#4C78A8", edgecolor="white")
    ax.set_title("Monetary (log scale) — 4,338 customers")
    ax.set_xlabel("log(1 + monetary)")
    ax.set_ylabel("customers")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "monetary_hist.png", dpi=110)
    plt.close(fig)

    # 2. Log-frequency histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log1p(feat["frequency"]), bins=30, color="#F58518", edgecolor="white")
    ax.set_title("Frequency (log scale)")
    ax.set_xlabel("log(1 + invoices)")
    ax.set_ylabel("customers")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "frequency_hist.png", dpi=110)
    plt.close(fig)

    # 3. Correlation heatmap
    features = ["recency", "frequency", "monetary", "avg_order_value",
                "distinct_products", "cancellation_rate", "tenure"]
    corr = feat[features].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), features, rotation=45, ha="right")
    ax.set_yticks(range(len(features)), features)
    for i in range(len(features)):
        for j in range(len(features)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "correlation_heatmap.png", dpi=110)
    plt.close(fig)

    # 4. Elbow + silhouette
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    log_features = ["recency", "frequency", "monetary", "avg_order_value",
                    "distinct_products", "cancellation_rate"]
    X = feat[["tenure"]].copy()
    for f in log_features:
        X[f] = np.log1p(feat[f].clip(lower=0))
    X = StandardScaler().fit_transform(X)

    ks, inertias, sils = [], [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        ks.append(k)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, labels))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(ks, inertias, "o-", color="#4C78A8", label="inertia")
    ax2.plot(ks, sils, "s-", color="#E45756", label="silhouette")
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia", color="#4C78A8")
    ax2.set_ylabel("silhouette", color="#E45756")
    ax1.set_title("k-selection: elbow + silhouette")
    ax1.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "elbow_silhouette.png", dpi=110)
    plt.close(fig)

    # 5. Skew bar chart
    skew = feat[features].skew().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(skew.index, skew.values, color="#72B7B2")
    ax.set_title("Feature skew (log scale)")
    ax.set_xscale("symlog")
    ax.axvline(1, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("skew (red line = threshold 1)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "skew.png", dpi=110)
    plt.close(fig)

    # 6. Cancellation-rate distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(feat["cancellation_rate"], bins=30, color="#B279A2", edgecolor="white")
    ax.set_title("Cancellation rate — share of invoices that are credit notes")
    ax.set_xlabel("cancellation_rate")
    ax.set_ylabel("customers")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "cancellation_rate_hist.png", dpi=110)
    plt.close(fig)

    print(f"Wrote {len(list(CHART_DIR.glob('*.png')))} charts to {CHART_DIR}")

    # Histogram bins for inline rendering
    for name, values, bins in [
        ("monetary", log_mon, 20),
        ("frequency", np.log1p(feat["frequency"]), 15),
    ]:
        counts, edges = np.histogram(values, bins=bins)
        labels = [f"{edges[i]:.1f}" for i in range(bins)]
        print(json.dumps({
            "name": name,
            "labels": labels,
            "counts": counts.tolist(),
        }))


if __name__ == "__main__":
    main()
