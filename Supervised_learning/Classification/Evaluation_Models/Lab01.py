# 1. Import required libraries

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 2. Load the data
data = load_breast_cancer()
X, y = data.data, data.target
labels = data.target_names
feature_names = data.feature_names

# 3. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add some noise
np.random.seed(42)
noise_factor = 0.5
X_noisy = X_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

# Load the original and noisy data sets into a DataFrame for comparison and visualization
df = pd.DataFrame(X_scaled, columns=feature_names)
df_noisy = pd.DataFrame(X_noisy, columns=feature_names)

plt.figure(figsize=(12, 6))

# Original Feature Distribution (Noise-Free)
# plt.subplot(1, 2, 1)
# plt.hist(df[feature_names[5]], bins=20, alpha=0.7, color='blue', label='Original')
# plt.title('Original Feature Distribution')
# plt.xlabel(feature_names[5])
# plt.ylabel('Frequency')
#
# # Noisy Feature Distribution
# plt.subplot(1, 2, 2)
# plt.hist(df_noisy[feature_names[5]], bins=20, alpha=0.7, color='red', label='Noisy')
# plt.title('Noisy Feature Distribution')
# plt.xlabel(feature_names[5])
# plt.ylabel('Frequency')
#
# plt.tight_layout()  # Ensures proper spacing between subplots
# plt.show()

# 4. Evaluate the models
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(kernel='linear', C=1, random_state=42)

knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)

# 5. print the accuracy scores
print(f"KNN Testing Accuracy: {accuracy_score(y_test, y_pred_knn):.3f}")
print(f"SVM Testing Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}")

print("\nKNN Testing Data Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nSVM Testing Data Classification Report:")
print(classification_report(y_test, y_pred_svm))