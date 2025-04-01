# step 1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# step 2: load dataset
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)


# step 3: defining our model
K_means = KMeans(init="k-means++",n_clusters=4, n_init=12)
K_means.fit(X)

# step 4: our labels and centroids
k_means_labels = K_means.labels_
k_means_cluster_centers = K_means.cluster_centers_

# step 5: initialise the visualization

fig = plt.figure(figsize=(6, 4))
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

ax = fig.add_subplot(1, 1, 1)


# step 6: plotting the visualization
for k, col in zip(range(len(set(k_means_labels))), colors):
    my_members = (k_means_labels == k)

    cluster_center = k_means_cluster_centers[k]

    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".", ms=10)

    ax.plot(cluster_center[0], cluster_center[1], markerfacecolor=col, marker="o", markeredgecolor="k", markersize=6)


#  step 7: visualization
ax.set_title("KMeans")

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

plt.show()