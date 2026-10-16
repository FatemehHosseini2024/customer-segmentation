"""Final evaluation for chosen K-Means k=3 (alternative k=4)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "final_evaluation.md"
FEATURES = ["recency", "frequency", "monetary", "distinct_products", "tenure"]
LOG_FEATURES = ["recency", "frequency", "monetary", "distinct_products"]
EXTRA = ["has_cancellation", "avg_order_value"]
B = 30
FRAC = 0.8
RANDOM_STATE = 42

def transform(df, cols):
    X = df[cols].copy()
    for f in LOG_FEATURES:
        if f in cols:
            X[f] = np.log1p(X[f].clip(lower=0))
    sc = StandardScaler()
    return sc.fit_transform(X[cols]), sc

def main():
    feat = pd.read_csv(REPORT_DIR / "customer_features.csv", index_col="CustomerID")
    X, sc = transform(feat, FEATURES)
    lines=[]
    lines.append("# Final evaluation: K-Means (k=3 primary, k=4 alternative)")
    lines.append("")
    for k in (3,4):
        lab = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
        sil = silhouette_score(X, lab)
        sils = silhouette_samples(X, lab)
        lines.append(f"## K-Means k={k}")
        lines.append(f"| metric | value |")
        lines.append(f"|---|---|")
        lines.append(f"| silhouette (mean) | {sil:.3f} |")
        lines.append(f"| share silhouette < 0.05 | {(sils<0.05).mean():.3f} |")
        lines.append(f"| share silhouette < 0 | {(sils<0).mean():.3f} |")
        lines.append("")
        lines.append("### Per-cluster silhouette")
        lines.append("| cluster | n | mean silhouette |")
        lines.append("|---|---|---|")
        for c in sorted(set(lab)):
            m = sils[lab==c].mean()
            n = int((lab==c).sum())
            lines.append(f"| {c} | {n} | {m:.3f} |")
        lines.append("")
        lines.append("### Profiles (original scale) - selected features")
        profile_cols = FEATURES + EXTRA
        dfp = feat[profile_cols].copy()
        dfp["cluster"] = lab
        g = dfp.groupby("cluster")[profile_cols].agg(["mean","median"]).round(2)
        lines.append(g.to_string())
        lines.append("")
        lines.append("### Standardized deviation from global mean (selected features)")
        glob = feat[FEATURES].mean()
        std = feat[FEATURES].std()
        means = dfp.groupby("cluster")[FEATURES].mean()
        z = ((means - glob)/std).round(2)
        lines.append(z.to_string())
        lines.append("")
        # profile stability bootstrap
        rng = np.random.RandomState(0)
        n = len(X)
        m = int(FRAC*n)
        stds=[]
        for _ in range(B):
            idx = rng.choice(n, size=m, replace=False)
            labb = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X[idx])
            # map by nearest centroid? simpler: record sorted cluster means of monetary
            means_b = pd.Series(X[idx,2]).groupby(labb).mean()
            stds.append(means_b.std())
        lines.append(f"Bootstrap (B={B}, frac={FRAC}): monetary cluster-mean spread std across resamples = {np.mean(stds):.3f}")
        lines.append("")
    # ARI k3 vs k4
    lab3 = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
    lab4 = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
    lines.append(f"ARI between k=3 and k=4 partitions: {adjusted_rand_score(lab3, lab4):.3f}")
    Path(REPORT_PATH).write_text("\n".join(lines), encoding="utf-8")
    print("wrote", REPORT_PATH)

if __name__=="__main__":
    main()
