# Problem: Spam Detection with SVM
# You are working for an email service provider, and they want to build a system that automatically classifies incoming emails as spam or not spam.
#
# Your goal is to train a Support Vector Machine (SVM) model to classify emails based on the content of the messages.
#

# step 1: import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

# step 2: load the data
raw_data = pd.read_csv("svm_02_spam.csv", encoding='ISO-8859-1')

# step 3: preprocessing

