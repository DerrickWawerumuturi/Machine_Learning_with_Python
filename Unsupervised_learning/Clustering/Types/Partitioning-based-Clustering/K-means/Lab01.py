import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# step 2: load the data
#  creating our dataset

np.random.seed(0)

#  making random cluster points using the make_blobs
# The make_blobs class can take in many inputs, but we will be using these specific ones.
# Input
#
# n_samples: The total number of points equally divided among clusters.
# Value will be: 5000
# centres : The number of centres to generate, or the fixed centre locations.
# Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
# cluster_std: The standard deviation of the clusters.
# Value will be: 0.9
#
# Output
# X: Array of shape [n_samples, n_features]. (Feature Matrix)
# The generated samples.
# y: Array of shape [n_samples]. (Response Vector)
# The integer labels for cluster membership of each sample.

X,y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
# plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.3,ec='k',s=80)

# step 3: setting up k-means
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)
k_means.fit(X)

k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

# step 4: creating the visual plot
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.

colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
    # create a list of all data points, where the data points are inr eh cluser are labeled true else false
    my_members = (k_means_labels == k)

    # defined the centroid, ro  cluster center
    cluster_center = k_means_cluster_centers[k]

    # plot datapoints with color col

    ax.plot(X[my_members,0], X[my_members, 1], 'w', markerfacecolor=col, marker=".", ms=10)

    # plot the centroids wot specified color but with a marker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
print(plt.show())