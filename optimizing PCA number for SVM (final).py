# As part of finding the optimal PCA number for the SVM machine learning model, I began by loading the dataset from a local CSV file.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# I loaded the dataset from a local CSV file named "new_train.csv".
file_path = "new_train.csv"
df = pd.read_csv(file_path)

# For the machine learning task, I defined input features (X) and the target variable (y).
X = df.drop('y', axis=1)
y = df['y']

# To work with categorical features, I applied one-hot encoding to convert them into numeric format.
X = pd.get_dummies(X, drop_first=True)

# In preparation for model training, I divided the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# To ensure consistent scaling of features, I standardized them using StandardScaler, a recommended step for SVM.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# My exploration included dimensionality reduction with PCA (Principal Component Analysis).
component_range = range(1, X_train.shape[1] + 1)
accuracy_scores = []

# I iterated through different numbers of principal components to find the optimal value.
for n_components in component_range:
    # I reduced the dataset's dimensionality with PCA.
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # For classification, I created and trained a Support Vector Machine (SVM) model with a linear kernel.
    svm_classifier = SVC(kernel='linear', C=1)
    svm_classifier.fit(X_train_pca, y_train)

    # Using the trained model, I made predictions on the test set.
    y_pred = svm_classifier.predict(X_test_pca)

    # To evaluate model performance, I calculated the accuracy and stored it for analysis.
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# I visualized the results by plotting accuracy against the number of principal components.
plt.figure(figsize=(10, 6))
plt.plot(component_range, accuracy_scores, marker='o', linestyle='-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Accuracy')
plt.title('PCA Component Analysis for SVM')
plt.grid(True)
plt.show()

# Finally, I determined the number of components that yielded the highest accuracy.
best_num_components = component_range[np.argmax(accuracy_scores)]
print(f'Best number of components: {best_num_components}')
