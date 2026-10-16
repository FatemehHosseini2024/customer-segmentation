from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('reports')
C = R / 'charts'
C.mkdir(exist_ok=True)
F = ['recency', 'frequency', 'monetary', 'distinct_products', 'tenure']
LOG = ['recency', 'frequency', 'monetary', 'distinct_products']
feat = pd.read_csv(R / 'customer_features.csv', index_col='CustomerID')
X = feat[F].copy()
for f in LOG:
    X[f] = np.log1p(X[f].clip(lower=0))
X = StandardScaler().fit_transform(X)
ks = list(range(2, 11))

km_sil = []
for k in ks:
    lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    km_sil.append(silhouette_score(X, lab))

agg_sil = {'ward': [], 'average': [], 'complete': []}
for k in ks:
    for l in agg_sil:
        lab = AgglomerativeClustering(n_clusters=k, linkage=l).fit_predict(X)
        agg_sil[l].append(silhouette_score(X, lab))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ks, km_sil, 'o-', label='K-Means')
for l, v in agg_sil.items():
    ax.plot(ks, v, '--', label=l)
ax.set_title('Silhouette vs k: K-Means vs Agglomerative')
ax.set_xlabel('k'); ax.set_ylabel('silhouette'); ax.legend()
fig.tight_layout()
fig.savefig(C / 'algorithm_comparison.png', dpi=110)
plt.close(fig)
print('ok')
