# Problem: Classify Wine Quality
# Dataset: UCI Wine Quality Dataset
# This dataset contains information about red and white wines, and you need to predict the quality of the wine (a score from 0 to 10) based on physicochemical tests.
#
# The dataset contains the following features:
#
# Fixed acidity
# Volatile acidity
# Citric acid
# Residual sugar
# Chlorides
# Free sulfur dioxide
# Total sulfur dioxide
# Density
# pH
# Sulphates
# Alcohol
# Quality (Target variable)
# Your task is to predict whether the wine quality is "good" (quality >= 6) or "bad" (quality < 6) using a Decision Tree Classifier.

# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# load the data
red_wine = pd.read_csv("Decision_03_winequality-red.csv", sep=';')
white_wine = pd.read_csv("Decision_03_winequality-white.csv", sep=';')



# preprocess the data
feature_columns_red = red_wine.drop('quality', axis=1).columns.tolist()
feature_columns_white = white_wine.drop('quality', axis=1).columns.tolist()

scaler = StandardScaler()
scaled_features_red = scaler.fit_transform(red_wine[feature_columns_red])
scaled_features_white = scaler.fit_transform(white_wine[feature_columns_red])

scaled_features_red_df = pd.DataFrame(scaled_features_red, columns=scaler.get_feature_names_out(feature_columns_red))
scaled_features_white_df = pd.DataFrame(scaled_features_white, columns=scaler.get_feature_names_out(feature_columns_white))

scaled_features_red_data = pd.concat([red_wine.drop(columns=feature_columns_red), scaled_features_red_df], axis=1)
scaled_features_white_data = pd.concat([white_wine.drop(columns=feature_columns_white), scaled_features_white_df], axis=1)

# encoding the target variable
scaled_features_red_data['quality'] = scaled_features_red_data['quality'].apply(lambda x: 1 if x >= 6 else 0)
scaled_features_white_data['quality'] = scaled_features_white_data['quality'].apply(lambda x: 1 if x >= 6 else 0)


# split the data
X_red = scaled_features_red_data.drop("quality", axis=1)
y_red = scaled_features_red_data["quality"]

X_white = scaled_features_white_data.drop("quality", axis=1)
y_white = scaled_features_white_data["quality"]


X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(X_white, y_white, test_size=0.2, random_state=42)
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red, y_red, test_size=0.2, random_state=42)


# train the data
Tree_white = DecisionTreeClassifier(criterion="entropy", max_depth=4)
Tree_red = DecisionTreeClassifier(criterion="entropy", max_depth=4)

Tree_red.fit(X_train_red, y_train_red)
Tree_white.fit(X_train_white, y_train_white)

# predict
red_pred = Tree_red.predict(X_test_red)
white_pred = Tree_white.predict(X_test_white)


print("Decision Tree Accuracy")
print(f'Red wine: {np.round(100*accuracy_score(y_test_red, red_pred), decimals=2)}%')
print(f'White wine: {np.round(100*accuracy_score(y_test_white, white_pred), decimals=2)}%')

