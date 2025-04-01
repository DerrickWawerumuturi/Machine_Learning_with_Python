# 1. Imports
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])

# 3. Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# iii]. Define a model parameter grid to search over
param_grid = {
    'pca__n_components': [2, 3],
    'knn__n_neighbors': [3, 5, 7]
}

# iv]. Choose a cross validation method
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# v]. Instantiate a GridSearchCV object
best_model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)

best_model.fit(X_train, y_train)
test_score =  best_model.score(X_test, y_test)

# make predictions
y_pred = best_model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

# Create a single plot for the confusion matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)

# Set the title and labels
plt.title('KNN Classification Testing Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()


#  so you can also preprocess the data
preprocessor = ColumnTransformer([
    'num', StandardScaler(), ['age', 'salary'], # applies scaling on numerical data
    'cat', OneHotEncoder(), ['city'] # Applies one hot encoding to categorical columns (columns not containing integers/floats)
])

pipeline2 = Pipeline([
    ('preprocess', preprocessor),

])