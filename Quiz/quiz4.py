## ####################################################
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score 

# load the data
X, y = load_breast_cancer(return_X_y=True)  ## X input, y output
print(X.shape, y.shape)

## to convert the {0,1} output into {-1,+1}
y = 2 * y -1

## X is the input matrix
mdata,ndim = X.shape
## normalization by L infinity norm
X/= np.outer(np.ones(mdata),np.max(np.abs(X),0))

## number of iteration 
niter = 10  

## penalty constant for the of the Stochastic Dual Coordinate Ascent algorithm
C = 1000



