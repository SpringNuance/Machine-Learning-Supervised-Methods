import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# the data
from sklearn.datasets import make_blobs
# linear models
from sklearn.linear_model import Perceptron, LinearRegression
# multi-class models
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier


# Create the dataset
C = 4
n = 800
X, y = make_blobs(n, centers=C, random_state=0)

np.random.seed(0)
order = np.random.permutation(n)
tr = order[:int(n/2)]
tst = order[int(n/2):]

Xt = X[tst, :]
yt = y[tst]
X = X[tr, :]
y = y[tr]

# use perceptron with default parameters as the base classifier for the multi-class methods
linear_classifier = Perceptron()

# Question 5
ovr = OneVsRestClassifier(linear_classifier).fit(X, y)
ovr_pred = ovr.predict(Xt)
ovo = OneVsOneClassifier(linear_classifier).fit(X, y)
ovo_pred = ovo.predict(Xt)
ecoc = OutputCodeClassifier(linear_classifier, random_state=42).fit(X, y)
ecoc_pred = ecoc.predict(Xt)

print("OVR accuracy:", np.round(100*accuracy_score(yt, ovr_pred), 1), "%")
print("OVO accuracy:", np.round(100*accuracy_score(yt, ovo_pred), 1), "%")
print("ECOC accuracy:", np.round(100*accuracy_score(yt, ecoc_pred), 1), "%")


# Question 6
len_params = np.arange(0.3, 4.1, 0.1)
X_new = np.ndarray(0)
y_new = np.ndarray(0)

for length in len_params:
    model = OutputCodeClassifier(linear_classifier,code_size=length, random_state=42).fit(X, y)
    model_pred = model.predict(Xt)
    X_new = np.append(X_new, model.code_book_[1].size)
    acc = 100*accuracy_score(yt, model_pred)
    y_new = np.append(y_new, acc)

X_new = X_new.reshape(-1, 1)
lin_model = LinearRegression().fit(X_new, y_new)

print("w = ", lin_model.coef_)
