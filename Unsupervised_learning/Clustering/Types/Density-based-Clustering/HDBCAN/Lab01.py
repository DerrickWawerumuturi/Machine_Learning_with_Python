# step 1: Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import hdbscan

# step 2: Generate synthetic data
X, _ = make_moons(n_samples=500, noise=0.05, random_state=42)

# step 3: Visualize the raw data
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.title("Raw Data")
plt.show()

# step 4: Apply dbscan
dbscan = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
labels_dbscan = dbscan.fit_predict(X)

# step 5: apply hdbscan
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
labels_hdbscan = dbscan.fit_predict(X)

# step 6: Visualize the cluster
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# DBSCAN result
ax[0].scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', s=10)
ax[0].set_title("DBSCAN clustering")

# HDBSCAN result
ax[1].scatter(X[:, 0], X[:, 1], c=labels_hdbscan, cmap='plasma' ,s=10)
ax[1].set_title("HDBCAN clustering")

plt.show()