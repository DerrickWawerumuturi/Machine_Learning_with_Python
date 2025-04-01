# step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# step 2: Load the data
data = pd.read_csv("knn_01_teleCust.csv")

# step 3: Data visualization and analysis
# correlation_matrix = data.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# correlation_values = abs(data.corr()['custcat'].drop('custcat')).sort_values(ascending=False)

# step 3: Separate data
X = data.drop('custcat', axis=1)
y = data['custcat']

# step 4: Normalize the data
X_norm = StandardScaler().fit_transform(X)

# step 5: train, test, split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# step 6: KNN classification
k=6
knn_classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn_models = knn_classifier.fit(X_train, y_train)

# step 7: predicting
yhat = knn_models.predict(X_test)

# step 8: accuracy eval
print("Test set Accuracy", 100*accuracy_score(y_test, yhat), "%")

Ks = 100
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
print(plt.show())

