import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv(r"C:\Users\ANITHA\AppData\Local\Temp\de5193ac-18cb-47c7-b820-20a58315ac3f_2.LOGISTIC REGRESSION CODE.rar.c3f\2.LOGISTIC REGRESSION CODE\logit classification.csv")

d2 = dataset1.copy()

# ---------------- FIXED ERROR HERE (dataset → dataset1) ----------------
X = dataset1.iloc[:, [3, 4]].values
y = dataset1.iloc[:, -1].values
# ----------------------------------------------------------------------

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
ac

# Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr

# Bias & Variance
bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test, y_test)
variance

# ---------------- FUTURE PREDICTION ----------------
dataset1 = pd.read_csv(
    r"C:\Users\ANITHA\AppData\Local\Temp\de5193ac-18cb-47c7-b820-20a58315ac3f_2.LOGISTIC REGRESSION CODE.rar.c3f\2.LOGISTIC REGRESSION CODE\logit classification.csv")
d2 = dataset1.copy()


dataset1 = dataset1.iloc[:, [3, 4]]   # Age, Estimated Salary only

# ---------------- FIXED ERROR HERE (DO NOT refit scaler) ----------------
M = sc.transform(dataset1)
# ----------------------------------------------------------------------

d2['y_pred1'] = classifier.predict(M)
d2.to_csv('final1.csv')

# ---------------- ROC - AUC ----------------
from sklearn.metrics import roc_auc_score, roc_curve

y_pred_prob = classifier.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_pred_prob)
auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# ---------------- TRAINING SET VISUALIZATION ----------------
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green'))(i),
        label=j
    )

plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# ---------------- TEST SET VISUALIZATION ----------------
X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green'))(i),
        label=j
    )

plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

