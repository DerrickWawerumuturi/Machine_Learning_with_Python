#
# Markdown
#
#
#
# Regression Trees
# Estimated time needed: 30 minutes
#
# In this exercise session you will use a real dataset to train a regression tree model. The dataset includes information about taxi tip and was collected and provided to the NYC Taxi and Limousine Commission (TLC) by technology providers authorized under the Taxicab & Livery Passenger Enhancement Programs (TPEP/LPEP).
# You will use the trained model to predict the amount of tip paid.

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

import warnings

from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings('ignore')

# load data
raw_data = pd.read_csv("Regression_01_yellow_tripdata.csv")
raw_data = raw_data.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis=1)

# understand the Dataset

correlation_values = raw_data.corr()['tip_amount'].drop("tip_amount")
correlation_values.plot(kind="barh", figsize=(10, 6))
# print(abs(correlation_values).sort_values(ascending=False)[:3])

# Preprocessing
y = raw_data[['tip_amount']].values.astype("float32")

proc_data = raw_data.drop(['tip_amount'], axis=1)

X = proc_data.values
X = normalize(X, axis=1, norm='l1', copy=False)


# train the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# BUILD THE TREE MODEL
dt_reg = DecisionTreeRegressor(criterion='squared_error', max_depth=4, random_state=35)

dt_reg.fit(X_train, y_train)

# prediction
y_pred = dt_reg.predict(X_test)

#evaluate the mean square error on the test
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score: {0:.3f}'.format(mse_score))

r2_score = dt_reg.score(X_test, y_test)
print('R^2 score : {0:.3f}'.format(r2_score))
