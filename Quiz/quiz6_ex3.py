import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import Lasso

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from sklearn.datasets import make_moons


n = 300
n_tr = 200
X, y = make_moons(n, shuffle=True, noise=0.2, random_state=112)
y[y==0] = -1

# standardise the data
trmean = np.mean(X[:n_tr, :], axis=0)
trvar = np.var(X[:n_tr, :], axis=0)
X = (X - trmean[np.newaxis, :]) / np.sqrt(trvar)[np.newaxis, :]

# take first n_tr as training, others as test.
Xtr = X[:n_tr, :]
ytr = y[:n_tr]
Xt = X[n_tr:, :]
yt = y[n_tr:]

# inspect the dataset visually
plt.figure()
plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, marker='x')
plt.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o')
plt.show()

# the parameters to consider in cross-validation
# order lasso params from largest to smallest so that if ties occur CV will select the one with more sparsity
params_lasso = [1, 1e-1, 1e-2, 1e-3, 1e-4]
params_rbf = [1e-3, 1e-2, 1e-1, 1, 10, 100]
params_svm = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]  # smaller C: more regularisation

# the noisy features to be considered; for n noisy features, add the first n columns
n_tot_noisy_feats = 50
np.random.seed(92)
X_noise = 2*np.random.randn(X.shape[0], n_tot_noisy_feats)
Xtr_noise = X_noise[:n_tr, :]
Xt_noise = X_noise[n_tr:, :]

