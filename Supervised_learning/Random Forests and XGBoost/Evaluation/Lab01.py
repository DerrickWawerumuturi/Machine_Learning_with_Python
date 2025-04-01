# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from  sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from scipy.stats import skew

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# 2. Load data
data = fetch_california_housing()
X, y = data.data, data.target

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# plt.hist(1e5*y_train, bins=30, color='lightblue', edgecolor='black')
# plt.title(f'Median House Value Distribution\nSkewness: {skew(y_train):.2f}')
# plt.xlabel('Median House Value')
# plt.ylabel('Frequency')

# 4. Model fitting and Prediction
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

y_pred_test = rf_regressor.predict(X_test)


# 5. Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)


# print(f"Mean Absolute Error (MAE): ${mae*1e5:.0f}")
# print(f"Mean Squared Error (MSE): ${mse*1e5:.0f}")
# print(f"Root Mean Squared Error (RMSE): ${rmse*1e5:.0f}")
# print(f"R² Score: {r2:.4f}")


# What do the statistics mean??
# We got a mean absolute error of $32754: which means the predicted median house prices
# are off by around $33k

# Mean squared error is less intuitive to interpret,
# but is usually what is being minimized by the model fit.

# we got the rsme of $50534

# AN r2 OF 0.80 is not considered very high. It means the model explains about 80% of the variance in median house prices,
# although this can be misleading since r2 works with linear data

# 6. Visualise
plt.scatter(y_test, y_pred_test, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regression - Actual vs Predicted")
plt.show()