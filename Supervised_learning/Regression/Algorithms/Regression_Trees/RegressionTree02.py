# Problem: Predicting House Prices
# You are working with a dataset that contains information about various houses, and you need to predict the price of a house based on its features. The dataset includes the following features:
#
# Area (in square feet): The size of the house.
# Number of Bedrooms: The number of bedrooms in the house.
# Number of Bathrooms: The number of bathrooms in the house.
# Year Built: The year the house was built.
# Distance to City Center (in miles): The distance from the house to the city center.
# Parking Spaces: Number of parking spaces available.
# The target variable is Price: The price of the house (in dollars).

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing

# load the data
cali_housing = fetch_california_housing()

data = pd.DataFrame(cali_housing.data, columns=cali_housing.feature_names)
data['Price'] = cali_housing.target

print(data.head())