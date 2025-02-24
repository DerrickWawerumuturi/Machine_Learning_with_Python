# Step 1: Import necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score


# Step 2:  Load the data set

# read_csv => reads data stored oin table form and returns a pandas  dataframe
# head() => gets the first 5 rows
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
data.head()

# print(data.isnull().sum())
# print(data.info())
# print(data.describe())


# Step 3: Explore Data analysis
sns.countplot(y="NObeyesdad", data=data)
plt.title("Distribution of Obesity levels")
plt.show()


# Step 4: Preprocessing the data

# A] : Identify numerical columns that need scaling

continuous_columns = data.select_dtypes(include=["float64"]).columns.tolist()
print(continuous_columns)
# this finds all continuous numerical columns

# B] : Initialise the StandardScaler
scaler = StandardScaler()

# C] : Apply the scaler to the numerical columns
scaled_features = scaler.fit_transform(data[continuous_columns])
print(scaled_features)

# D] : Convert the scaled data back into a dataframe
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
print(scaled_df)

# E] : Combine with original dataset
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)
print(scaled_data)


# Step 4: One-hot encoding

# [i]. Encoding the features
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove("NObeyesdad")

# Applying one-hot encoding (converting the features into binary)
encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# [ii]. Encoding the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype("category").cat.codes
print(prepped_data.head())


# Step 5: Separate the input and target data
X = prepped_data.drop("NObeyesdad", axis=1)
y = prepped_data["NObeyesdad"]


#  Model Training and Evaluation

#  Step 6:  Splitting the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

#  Step 7: Logistic Regression with One-vs-All
model_ova = LogisticRegression(multi_class="ovr", max_iter=1000)
model_ova.fit(X_train, y_train)

# Step 8: Predictions
y_pred_ova = model_ova.predict(X_test)
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")


## using the One vs One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)
# Predictions
y_pred_ovo = model_ovo.predict(X_test)

# Evaluation metrics for OvO
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")