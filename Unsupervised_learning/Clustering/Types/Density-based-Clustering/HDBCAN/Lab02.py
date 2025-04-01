# step 1: Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import hdbscan


# step 2: Load the dataset
X, _ = make_moons(n_samples=1000, noise=0.09, random_state=42)

# step 3: Visualize raw data
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.show()

# step 4: create the model
hdbscan_clusterer =  hdbscan.HDBSCAN(min_cluster_size=5)
labels_hdbscan = hdbscan_clusterer.fit_predict(X)

# step 5: visualize the data
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(X[:, 0], X[:, 1], c=labels_hdbscan, cmap="viridis", s=10)
ax.set_title("HDBSCAN clustering")

plt.show()
