# Step 1: Import required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 2: Create a simple dataset

data = {
    "Size": [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    "Bedrooms": [3, 4, 3, 5, 2, 3, 4, 4, 3, 3],
    "Distance": [10, 15, 10, 12, 5, 8, 20, 25, 10, 15],
    "Price": [245000, 312000, 279000, 308000, 199000, 219000, 405000, 442000, 250000, 299000]
}

# convert dictionary to DataFrame
df = pd.DataFrame(data)


# Step 3: Define the independent(X) and dependent(y) variables
X = df[["Size", "Bedrooms", "Distance"]] # features
y = df["Price"] # target variable

# Step 4: Split data into training and testing sets (80% training, 20% testing)
# train_test_split(features, target, test_size, random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 5: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)