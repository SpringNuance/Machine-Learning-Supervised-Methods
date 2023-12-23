import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler

"""
More info about the attributes in the dataset:
https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset
"""

X, y = load_breast_cancer(return_X_y=True)
print("data shapes:", X.shape, y.shape, np.unique(y))

# ------------------------------------------------------------------------------------
# add here transformations on X

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# ------------------------------------------------------------------------------------
# linear classification

# divide into training and testing
np.random.seed(42)
order = np.random.permutation(len(y))
tr = np.sort(order[:250])
tst = np.sort(order[250:])

svm = LinearSVC(fit_intercept=False, random_state=2)
svm.fit(X[tr, :], y[tr])
preds = svm.predict(X[tst, :])
print("SVM accuracy:", np.round(100*accuracy_score(y[tst], preds), 1), "%")
