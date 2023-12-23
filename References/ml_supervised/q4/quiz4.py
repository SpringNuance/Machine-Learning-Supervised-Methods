## ####################################################
import numpy as np
from sklearn.datasets import load_breast_cancer

# load the data
X, y = load_breast_cancer(return_X_y=True)  ## X input, y output
print(X.shape, y.shape)
## to convert the {0,1} output into {-1,+1}
y = 2 * y -1

mdata,ndim=X.shape                                   ## size of the data 

iscale = 1   ## =0 no scaling, =1 scaling the by the maximum absolute value
if iscale == 1:
  X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))

niter = 10 ## number of iteration 

## initialize eta, lambda for the primal algorithm
eta=0.1              ##  step size
xlambda=0.01          ## balancing constant between loss and regularization
## set the penalty constant for the dual algorithm
C = 1000

# the primal function
w_primal = np.zeros(ndim)
for iteration in range (10):
  for i in range(mdata):
    if(np.dot(y[i], np.dot(w_primal, X[i])) < 1):
      gradient = -np.dot(y[i], X[i]) + xlambda*w_primal
    else:
      gradient = xlambda*w_primal
    w_primal = w_primal-eta*gradient

# the dual function
alpha = np.zeros(mdata)
for iteration in range (10):
  for i in range(mdata):
    kernel_sum = 0
    for j in range(mdata):
      if(i != j):
        kernel = np.dot(X[i], X[j])
        kernel_sum += alpha[j]*np.dot(y[j], kernel)
      alpha_i = (1 - y[i]*kernel_sum) / (np.dot(X[i], X[i]))
      alpha[i] = min(C/mdata, max(0, alpha_i))

w_dual = np.zeros(ndim)
for i in range(mdata):
  w_dual += alpha[i]*np.dot(y[i], X[i])

# the correlation
print("Pearson correlation coefficient: ", np.corrcoef(w_primal, w_dual))




