# 1. Import required libraries

import pandas as pd
import plotly.express as px
from sklearn.datasets import make_blobs

# 2. Generate synthetic data

# Cluster centers:
centers = [
    [2, -6, -6],
    [-1, 9, 4],
    [-8, 7, 2],
    [4, 7, 9]
]

# cluster standard deviations:
cluster_std = [1, 1, 2, 3.5]

# Make the blobs and return the blob labels and data
X, labels_ = make_blobs(n_samples=500, centers=centers, n_features=3, cluster_std=cluster_std, random_state=42)

# 3. Display the data
df = pd.DataFrame(X, columns=['X', 'Y', 'Z'])

# create interactive plot
fig  = px.scatter_3d(df, x='X', y='Y', z='Z', color=labels_.astype(str), opacity=0.7, color_discrete_sequence=px.colors.qualitative.G10, title='3D Scatter Plot of Four Blobs')

fig.update_traces(marker=dict(size=5, line=dict(width=1, color='black')), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800)

fig.show()