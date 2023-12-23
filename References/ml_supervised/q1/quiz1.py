import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# load the data
X, y = load_diabetes(return_X_y=True)
print(X.shape, y.shape)

# division into training and testing
np.random.seed(0)
order = np.random.permutation(len(y))
tst = np.sort(order[:200])
tr = np.sort(order[200:])

Xtr = X[tr, :]
Xtst = X[tst, :]
Ytr = y[tr]
Ytst = y[tst]

lnr_false = LinearRegression(fit_intercept=False)
reg = lnr_false.fit(Xtr, Ytr)

Ypred = reg.predict(Xtst)

print(mean_squared_error(Ytst, Ypred))