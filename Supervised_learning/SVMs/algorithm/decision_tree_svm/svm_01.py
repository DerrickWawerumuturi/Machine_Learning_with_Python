# Introduction
#
# Imagine that you work for a financial institution and part of your job is to build a model that predicts if a credit card transaction is fraudulent or not. You can model the problem as a binary classification problem. A transaction belongs to the positive class (1) if it is a fraud, otherwise it belongs to the negative class (0).
#
# You have access to transactions that occurred over a certain period of time. The majority of the transactions are normally legitimate and only a small fraction are non-legitimate. Thus, typically you have access to a dataset that is highly unbalanced. This is also the case of the current dataset: only 492 transactions out of 284,807 are fraudulent (the positive class - the frauds - accounts for 0.172% of all transactions).
#
# This is a Kaggle dataset. You can find this "Credit Card Fraud Detection" dataset from the following link: Credit Card Fraud Detection.
#
# To train the model, you can use part of the input dataset, while the remaining data can be utilized to assess the quality of the trained model. First, let's import the necessary libraries and download the dataset.

# step 1: Import libraries

from __future__ import  print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

# step 2: Load the dataset
raw_data = pd.read_csv("svm_01_creditcard.csv")

labels = raw_data.Class.unique()

sizes = raw_data.Class.value_counts().values
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%',)
ax.set_title("Target variable Value Counts")

correlation_values = raw_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10, 6))

# step 3: Preprocessing
raw_data.iloc[:, 1: 30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values

X = data_matrix[:, 1:30]
y = data_matrix[:, 30]

# data normalization

X = normalize(X, norm='l1')

# step 4: Train, split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# step 5: Build a decision tree classifier model
w_train = compute_sample_weight("balanced", y_train)

dt = DecisionTreeClassifier(max_depth=4, random_state=35)
dt.fit(X_train, y_train, sample_weight=w_train)

# step 6: build a support vector machine model
svm = LinearSVC(class_weight='balanced', random_state=31, loss='hinge', fit_intercept=False)
svm.fit(X_train, y_train)

# step 7: make predictions
y_pred_dt = dt.predict_proba(X_test)[:,1]
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)

print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

# evaluate for svm model
y_pred_svm = svm.decision_function(X_test)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

