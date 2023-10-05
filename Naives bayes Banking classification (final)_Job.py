# As part of the machine learning project, I'll begin by importing the necessary libraries.

import pandas as pd  # First, I import Pandas, which allows me to work with data in a tabular form.
from sklearn.naive_bayes import GaussianNB  # For this project, I've chosen to use the Gaussian Naive Bayes classifier.
from sklearn.metrics import accuracy_score  # To evaluate the model, I'll use accuracy_score to measure its accuracy.

# Now, I'll load the dataset. In this case, it's from my github account.

url = "https://raw.githubusercontent.com/jhuisman3/project/main/new_train.csv"
df = pd.read_csv(url)

# With the data loaded, I can define my features (X) and my target variable (y).

X = df.drop('y', axis=1)  # 'X' will contain the input features, excluding the target variable 'y'.
y = df['y']  # 'y' is the target variable I want to predict.

# Since some of the features are categorical, I'll perform one-hot encoding to convert them into a numerical format.
# This step is crucial as machine learning models require numerical input.

X = pd.get_dummies(X, drop_first=True)
# I include the 'drop_first=True' parameter to prevent multicollinearity, which can affect the model's performance.

# With the data prepared, I'm ready to create and train the Gaussian Naive Bayes model.

nb_classifier = GaussianNB()
nb_classifier.fit(X, y)
# I fit the model using the entire dataset, allowing it to learn the patterns in the features.

# Now, it's time to make predictions using the trained model.

y_pred = nb_classifier.predict(X)

# To understand how well the model performs, I'll calculate its accuracy by comparing its predictions to the actual target values.

accuracy = accuracy_score(y, y_pred)
# Accuracy is a common metric for classification tasks, measuring how many predictions were correct.

# Finally, I'll display the accuracy of the Naive Bayes model's predictions.

print(f'Accuracy: {accuracy * 100:.2f}%')
# This line displays the accuracy as a percentage, providing an assessment of the model's performance.

# This completes the process of loading data, preparing it, training a machine learning model, and evaluating its accuracy.
