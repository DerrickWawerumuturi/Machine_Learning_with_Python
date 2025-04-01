# 1. Imports
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix


# 2. Train the model using a pipeline


# i]. Load iris data set
data = load_iris()
X, y = data.data, data.target
labels = data.target_names

# ii]. Instantiate the pipeline
pipeline = Pipeline([
    # step 1: standardize the features
    ('scaler', StandardScaler()),
    # step 2: Reduce dimension to 2 using PCA
    ('pca', PCA(n_components=2),),
    # step 3: train a KNN classifier
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# 3. Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Fit the pipeline on the training set
pipeline.fit(X_train, y_train)

# 5. Make Predictions
y_pred = pipeline.predict(X_test)

# 6. Plot a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)

plt.title("Classification Pipeline Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

