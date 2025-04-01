# step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN




# step 2: Load the dataset
raw_data = pd.read_csv("Mall_Customers.csv")

# step 3: preprocessing the data

# convert the gender column to binary
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(raw_data[['Gender']])

encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Gender']))

# scale the other features
scaler = StandardScaler()
data_to_scale = raw_data.select_dtypes(include=["int"]).columns.tolist()
scaled_data = scaler.fit_transform(raw_data[data_to_scale])

scaled_df = pd.DataFrame(scaled_data, columns=scaler.get_feature_names_out(data_to_scale))

#  final form
prepped_data = pd.concat([raw_data.drop(columns=['Gender'] + data_to_scale), encoded_df, scaled_df], axis=1)

# step 6: Apply the models
X = prepped_data.values

eps=0.5
min_samples = 3
metric = 'euclidean'
min_cluster_size = 5


# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=min_samples, metric=metric)
dbscan_labels = dbscan.fit_predict(X)

# HDBSCAN


hdbscan_clusterer = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric=metric)
hdbscan_labels = hdbscan_clusterer.fit_predict(X)

# step 7: visualizing the end
plt.figure(figsize=(10, 7))
fig, ax = plt.subplots(1, 2 )

ax[0].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=10)
ax[0].set_title('DBSCAN Clustering')

ax[1].scatter(X[:, 0], X[:, 1], c=hdbscan_labels, cmap='plasma', s=10)
ax[1].set_title('HDBCAN Clustering')

plt.show()

