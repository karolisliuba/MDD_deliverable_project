# First, I import the necessary libraries for working with data and machine learning.

import pandas as pd  # Pandas allows us to work with data in tabular form.
from sklearn.svm import SVC  # SVC implements the Support Vector Machine classifier.
from sklearn.preprocessing import StandardScaler  # StandardScaler scales our data to improve model performance.
from sklearn.metrics import accuracy_score  # Accuracy_score measures the accuracy of our model's predictions.
from sklearn.decomposition import PCA  # PCA performs Principal Component Analysis for dimensionality reduction.
from tqdm import tqdm  # TQDM provides a progress bar to track training progress.

# Load the dataset from my github account
url = "https://raw.githubusercontent.com/jhuisman3/project/main/new_train.csv"
df = pd.read_csv(url)

# Define our features (X) and target (y)
X = df.drop('y', axis=1)  # 'X' contains the input features, excluding the target variable 'y'.
y = df['y']  # 'y' is the target variable we want to predict.

# Perform one-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)  # Convert categorical data into numerical format.
# 'drop_first=True' is used to avoid multicollinearity, which helps in modeling.

# Standardize (scale) the features to have a consistent scale
scaler = StandardScaler()  # This ensures that each feature has a similar influence on the model.
X = scaler.fit_transform(X)  # Scale the data.

# Reduce dimensionality using Principal Component Analysis (PCA) with a specific number of components
n_components = 29  # Enter the desired number of components.
pca = PCA(n_components=n_components)  # Reduce dimensions to 'n_components' principal components.
X_pca = pca.fit_transform(X)  # Apply PCA to the data.

# Create and train a Support Vector Machine (SVM) model
svm_classifier = SVC(kernel='linear', C=1)
# 'kernel' defines the type of mathematical function used to separate data.
# 'C' is a regularization parameter that controls the trade-off between fitting the data and preventing overfitting.

# Track the training progress with a progress bar (tqdm)
with tqdm(total=100, desc="Training SVM") as pbar:
    svm_classifier.fit(X_pca, y)  # Train the SVM model.
    pbar.update(100)  # Update the progress bar to indicate 50% completion of training.

# Make predictions using the trained model (no test set in this code)
y_pred = svm_classifier.predict(X_pca)

# Evaluate the model's accuracy by comparing its predictions to the actual target values
accuracy = accuracy_score(y, y_pred)

# Print the accuracy of the SVM model's predictions
print(f'Accuracy: {accuracy * 100:.2f}%')
# This line displays the accuracy as a percentage.



