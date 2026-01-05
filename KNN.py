import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load the dataset
dataset = pd.read_csv(r"C:\Users\ANITHA\AppData\Local\Temp\de5193ac-18cb-47c7-b820-20a58315ac3f_2.LOGISTIC REGRESSION CODE.rar.c3f\2.LOGISTIC REGRESSION CODE\logit classification.csv")
dataset.head()

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


