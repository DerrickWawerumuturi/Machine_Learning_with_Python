# Part I: Using PCA to project 2-D data onto its principal axes

# Here, you will illustrate how you can use PCA to transform your 2-D data to represent it in terms of its principal axes (the directions of maximum variance in the data).
#       - the projection of your data onto the two orthogonal directions that explain most of the variance in your data.
# Let's see what all of this means.


# 1. Import required Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import  datasets
from sklearn.preprocessing import StandardScaler


# 2. Create Dataset
np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]
X = np.random.multivariate_normal(mean=mean, cov=cov, size=200)
#
# # 3. Visualize Dataset
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], edgecolor='k', alpha=0.7)
# plt.title('Scatter Plot of Bivariate Normal Distribution')
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.axis('equal')
# plt.grid(True)
# plt.show()


# 4. Perform PCA on the dataset
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

components = pca.components_
variance_ratio = pca.explained_variance_ratio_

# 5. Visualize the dat
projection_pc1 = np.dot(X, components[0])
projection_pc2 = np.dot(X, components[1])

x_pc1 = projection_pc1 * components[0][0]
y_pc1 = projection_pc1 * components[0][1]
x_pc2 = projection_pc2 * components[1][0]
y_pc2 = projection_pc2 * components[1][1]

plt.figure()
plt.scatter(X[:, 0], X[:, 1], label='Original Data', ec='k', s=50, alpha=0.6)

# Plot the projections along PC1 and PC2
plt.scatter(x_pc1, y_pc1, c='r', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 1')
plt.scatter(x_pc2, y_pc2, c='b', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 2')
plt.title('Linearly Correlated Data Projected onto Principal Components', )
plt.xlabel('Feature 1',)
plt.ylabel('Feature 2',)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
