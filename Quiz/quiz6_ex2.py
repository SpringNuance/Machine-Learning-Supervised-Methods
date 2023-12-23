import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score

"""
More info about the attributes in the dataset:
https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset
"""


X_original, y_original = load_breast_cancer(return_X_y=True)
y_original[y_original==0] = -1
print("data shapes:", X_original.shape, y_original.shape)

# return_X_y=True is an easy option here if you just want to quickly apply ML algorithms
# with return_X_y=False, a pandas dataframe is returned instead
# in this dataframe there is more information about the data, for example the feature names:
bc_pandas_frame = load_breast_cancer(return_X_y=False)
print("\nfeature names:")
for ii in range(X_original.shape[1]):
    print(ii, bc_pandas_frame.feature_names[ii])

# divide into training and testing
np.random.seed(7)
order = np.random.permutation(len(y_original))
tr = np.sort(order[:250])
tst = np.sort(order[250:])

from collections import Counter
print("\nClasses in training:", Counter(y_original[tr]))
print("Classes in testing:", Counter(y_original[tst]))
print("Majority vote accuracy:", np.round(100*accuracy_score(y_original[tst],
                                                             np.sign(np.sum(y_original[tr]))*np.ones(len(tst))), 2))
