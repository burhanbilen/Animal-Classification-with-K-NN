import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('zoo.data') # Read the dataset

X = df.iloc[:,1:17].values # Get the features as input values
y = np.array(df.iloc[:,17:]).reshape(len(X),) # Shape the output to fit the model properly

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # Split the entire data to train and test packages.


knn = KNeighborsClassifier(n_neighbors=3) # Model setup
knn.fit(X_train, y_train) # Fit the train input and output to the model

#scr = knn.score(X_test, y_test) # alternative accuracy score

y_pred = knn.predict(X_test[:]) # Predict whole test data
print(y_pred)

accuracy = accuracy_score(y_test, y_pred) # Get the accuracy score
print(accuracy)

print(confusion_matrix(y_test,y_pred)) # Print the confusion matrix
print(classification_report(y_test,y_pred)) # Print the model report to examine precision, recall and f-measure values
