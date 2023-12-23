## ####################################################
import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler
## ###################################################

# load the data
X, y = load_breast_cancer(return_X_y=True)  ## X input, y output
## to convert the {0,1} output into {-1,+1}
y = 2*y - 1

transformer = MaxAbsScaler().fit(X)
X = transformer.transform(X)

print(X.shape, y.shape)
mdata,ndim = X.shape


nfold = 5         ## number of folds
cselection = KFold(n_splits=nfold, random_state=None, shuffle=False)
## initialize the learning parameters for all folds
f1 = np.zeros(nfold)
maxmargin_train = np.zeros(nfold)

"""
To do ....

"""
pos = 0
for train_index, test_index in cselection.split(X):
    w = np.zeros(ndim)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    iteration = 0
    max_margin = -np.inf

    #Train
    while iteration < 8:
        for i in range(X_train.shape[0]):
            score = np.dot(y_train[i]*np.transpose(w), X_train[i])
            margin = abs(y_train[i] * np.dot(w, X_train[i]))
            if score <= 0:
                w += y_train[i]*X_train[i]
            if(margin > max_margin):
                max_margin = margin
        iteration += 1

    maxmargin_train[pos] = max_margin

    #Apply the trained weight
    y_pred = np.zeros(y_test.shape[0])
    for i in range(y_test.shape[0]):
        y_hat = np.dot(w, X_test[i])
        if y_hat <= 0:
            y_pred[i] = -1
        else:
            y_pred[i] = 1

    #computing the score
    f1_loc = f1_score(y_test, y_pred)
    f1[pos] = f1_loc
    pos += 1


print('The average F1:',np.mean(f1))
print('The average maximum margin achieved in the training:',np.mean(maxmargin_train))


 
