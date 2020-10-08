import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('zoo.data')

X = df.iloc[:,1:17].values
y = np.array(df.iloc[:,17:]).reshape(len(X),)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
#scr = knn.score(X_test, y_test) # alternative accuracy score

y_pred = knn.predict(X_test[:])
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
