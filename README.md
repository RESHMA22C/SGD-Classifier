# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import Libraries
Import pandas, numpy, load_iris, SGDClassifier, train_test_split, and evaluation metrics from sklearn.

2.Load Dataset
Load the Iris dataset using load_iris().

3.Create DataFrame
Convert the data into a pandas DataFrame and add the target labels as a new column.

4.Inspect Data
Use df.head() to preview the first few rows.

5.Define Features and Target
Set x as all feature columns and y as the target column.

6.Split the Data
Use train_test_split() to split into training and testing sets (80/20 split).

7.Initialize Classifier
Create an SGDClassifier with max_iter=1000 and tol=1e-3.

8.Train the Model
Fit the classifier on the training data.

9.Make Predictions
Predict target values for the test set.

10.Evaluate the Model
Use accuracy_score, confusion_matrix, and classification_report to assess performance.



## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: RESHMA C
RegisterNumber:  212223040168
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
![image](https://github.com/user-attachments/assets/d167fcdc-3746-4d2f-99b7-72ca1ca4013e)

![image](https://github.com/user-attachments/assets/54f0a896-d828-49b4-8d3c-8732cfa09d22)

![image](https://github.com/user-attachments/assets/891f0ccf-846a-4a2e-b376-681659bf7bb9)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
