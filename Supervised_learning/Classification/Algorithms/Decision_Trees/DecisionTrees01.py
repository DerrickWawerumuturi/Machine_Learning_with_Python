# Step 1: Import the required Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# About the dataset
# Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug C, Drug X and Drug Y.
#
# Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness. The features of this dataset are the Age, Sex, Blood Pressure, and Cholesterol of the patients, and the target is the drug that each patient responded to.
#
# It is a sample of a multiclass classifier, and you can use the training part of the dataset to build a decision tree, and then use it to predict the class of an unknown patient or to prescribe a drug to a new patient.

#  Step 2: Download the data
my_data = pd.read_csv("Decision_01_drug200.csv")

# Step 3: Data analysis and pre-processing

label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])

# to map the correlation between the features and target variables
custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)

# Step 4: Modeling
y = my_data['Drug']
X = my_data.drop(['Drug', 'Drug_num'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_train, y_train)

# Step 5: Evaluation
tree_predictions = drugTree.predict(X_test)
print("Decision_Trees's Accuracy", metrics.accuracy_score(y_test, tree_predictions))

plot_tree(drugTree)
print(plt.show())
