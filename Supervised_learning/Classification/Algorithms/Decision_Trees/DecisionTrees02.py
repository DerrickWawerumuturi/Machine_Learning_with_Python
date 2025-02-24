# You will use health-related features (e.g., blood pressure, glucose level) to classify whether a patient has diabetes (1) or not (0) using a Decision Tree Classifier.
#
# Dataset: Pima Indians Diabetes Dataset
# This dataset contains medical information from 768 patients, including:
#
# Pregnancies (Number of times pregnant)
# Glucose (Blood glucose level)
# BloodPressure (Diastolic blood pressure)
# SkinThickness (Triceps skin fold thickness)
# Insulin (Blood insulin level)
# BMI (Body Mass Index)
# DiabetesPedigreeFunction (Diabetes hereditary likelihood)
# Age (Age in years)
# Outcome (0 = No Diabetes, 1 = Diabetes)
#  import
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# load the data set
my_data = pd.read_csv("Decision_02_diabetes.csv")


# preprocessing the data
scaler = StandardScaler()

continuous_columns = my_data.drop("Outcome", axis=1).columns.tolist()
scaled_features = scaler.fit_transform(my_data[continuous_columns])

scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
scaled_data = pd.concat([my_data.drop(columns=continuous_columns), scaled_df], axis=1)


y = scaled_data['Outcome']
X = scaled_data.drop("Outcome", axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dataTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

# train
dataTree.fit(X_train, y_train)

# test
tree_pred = dataTree.predict(X_test)
print("Decision tree accuracy", np.round(100*accuracy_score(y_test, tree_pred), 2),"%")