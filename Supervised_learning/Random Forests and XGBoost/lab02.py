import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time

# step 2: load the dataset
data = pd.read_csv("housing.csv")
X = data.drop(["median_house_value", "ocean_proximity"])
y = data['median_house_value']


# step 3: preprocessing
X_scaled = StandardScaler().fit_transform(X)

# step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# step 5: models

#  random tree
n_estimators = 100
rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
rf.fit(X_train, y_train)

# xgb
xgb = XGBClassifier(n_estimators=n_estimators, random_state=42)
xgb.fit(X_train, y_train)

# step 6: Predictions

# random tree prediction
start_time_rf = time.time()
rf_pred = rf.predict(X_test)
end_time = time.time()


# xgb
start_time_xgb = time.time()
xgb_pred = xgb.predict(X_test)
end_time_xgb = time.time()

#

