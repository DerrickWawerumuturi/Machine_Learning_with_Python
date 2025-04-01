# step 1: import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler

# geographical tools
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

# Write a function that plots clustered locations and overlays them on a basemap.

def plot_clustered_locations(df,  title='Museums Clustered by Proximity'):
    """
    Plots clustered locations and overlays on a basemap.

    Parameters:
    - df: DataFrame containing 'Latitude', 'Longitude', and 'Cluster' columns
    - title: str, title of the plot
    """

    # Load the coordinates into a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")

    # Reproject to Web Mercator to align with basemap
    gdf = gdf.to_crs(epsg=3857)

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Separate non-noise, or clustered points from noise, or unclustered points
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]

    # Plot noise points
    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')

    # Plot clustered points, colured by 'Cluster' number
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)

    # Add basemap of  Canada
    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)

    # Format plot
    plt.title(title, )
    plt.xlabel('Longitude', )
    plt.ylabel('Latitude', )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # Show the plot
    print(plt.show())

# step 2: load the data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df = pd.read_csv(url, encoding = "ISO-8859-1")


# Step 3: preprocessing
# In this case we know how to scale the coordinates. Using standardization would be an error because we aren't using the full range of the lat/lng coordinates.
# Since latitude has a range of +/- 90 degrees and longitude ranges from 0 to 360 degrees, the correct scaling is to double the latitude coordinates (or half the Latitudes)
coords_scale = df.copy()
coords_scale['Latitude'] = 2*coords_scale['Latitude']


# step 4: Build dbscan

# minimum number of samples to form a neighbourhood
min_samples = 3

# neighbourhood search radius
eps = 1.0

# distance measure
metric ='euclidean'

dbscan  = DBSCAN(eps, min_samples=min_samples, metric=metric).fit(coords_scale)

# step 5: Add cluster labels to the dataframe

# assign the cluster labels
df['Cluster'] =  dbscan.fit_predict(coords_scale)

# display the size of each cluster
print(df['Cluster'].value_counts)

# step 6: visualize
plot_clustered_locations(df, title='Museums Clustered by Proximity')



# HDBSCAN
min_samples = None
min_cluster_size = 3
hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric='euclidean').fit(coords_scale)

df['Cluster']  = hdb.fit_predict(coords_scale)
print(df['Cluster'].value_counts())

plot_clustered_locations(df, title='Museums Hierarchically Clustered by Proximity')