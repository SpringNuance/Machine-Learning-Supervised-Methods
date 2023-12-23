"""
This code is based on the Sklearn example:
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html?highlight=classifier%20comparison
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
## data set
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
## learners
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


## the data sets
datasets = [
    load_breast_cancer(return_X_y=True),  ## X input, y output
    make_moons(n_samples = 100, noise=0.3, random_state=1),
    make_circles(n_samples = 100, noise=0.2, factor=0.5, random_state=1)
]

## setting the learners parameters
classifiers = [
    MLPClassifier(hidden_layer_sizes = (100,), alpha=0.0001, max_iter=200, \
                    random_state=1),
    AdaBoostClassifier(n_estimators = 100, random_state=1),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1, \
                                 max_depth=1, random_state=1), \
    RandomForestClassifier(max_depth=4, random_state=1)
]




