import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

#  step 2: Load the data

data = pd.read_csv("housing.csv")
X_org = data.drop(["median_house_value", "ocean_proximity"], axis=1)
y = data["median_house_value"]

X = StandardScaler().fit_transform(X_org)

# step 3: split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# step 4: initialize models
#  n_estimators is the number of decision trees you use
n_estimators=100
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)

# step 5: training the models with time

# for random forests
start_time_rf = time.time()
rf.fit(X_train, y_train)
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf

# print("start_time_rf", start_time_rf)
# print("end_time_rf", end_time_rf)
# print("rf_train_time", rf_train_time)


# for xgboost
start_time_xgb = time.time()
xgb.fit(X_train, y_train)
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb

# print("start_time_xgb", start_time_xgb)
# print("end_time_xgb", end_time_xgb)
# print("xgb_train_time", xgb_train_time)

# STEP 6: Predictions
#  FOR Random forests
pred_start_time_rf = time.time()
rf_pred = rf.predict(X_test)
pred_end_time_rf = time.time()
rf_pred_time = pred_end_time_rf - pred_start_time_rf
# print("pred_rf_start_time", pred_start_time_rf)
# print("pred_rf_end_time", pred_end_time_rf)
# print("rf_pred_time", rf_pred_time)
# print("rf_pred", rf_pred)

#  for xgb
pred_start_time_xgb = time.time()
xgb_pred = xgb.predict(X_test)
pred_end_time_xgb = time.time()
xgb_pred_time = pred_end_time_xgb - pred_start_time_xgb
# print("xgb_pred_time", xgb_pred_time)
# print("pred_start_time_xgb", pred_start_time_xgb)
# print("pred_end_time_xgb", pred_end_time_xgb)
# print("xgb_pred", xgb_pred)

# step 7: calculate the mse and r2
mse_rf = mean_squared_error(y_test, rf_pred)
mse_xgb = mean_squared_error(y_test, xgb_pred)

print("mse_rf", mse_rf)
print("mse_xgb", mse_xgb)

r2_score_rf = r2_score(y_test, rf_pred)
r2_score_xgb = r2_score(y_test, xgb_pred)


print(f'Random Forest:  MSE = {mse_rf:.4f}, R^2 = {r2_score_rf:.4f}')
print(f'      XGBoost:  MSE = {mse_xgb:.4f}, R^2 = {r2_score_xgb:.4f}')

std_y = np.std(y_test)

# step 8: visualize the results
plt.figure(figsize=(14, 6))

# Random Forest plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, rf_pred, alpha=0.5, color="blue",ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(0,6)
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()

# XGBoost plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, xgb_pred, alpha=0.5, color="orange",ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(0,6)
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Actual Values")
plt.legend()
plt.tight_layout()
plt.show()