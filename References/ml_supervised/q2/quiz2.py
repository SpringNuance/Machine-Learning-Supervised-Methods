import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# ========================================================================
# dataset

n_tot = 200
# two blobs, not completely separated
X, y = make_blobs(n_tot, centers=2, cluster_std=3.0, random_state=2)

plt.figure()
colors = ["g", "b"]
for ii in range(2):
    class_indices = np.where(y==ii)[0]
    plt.scatter(X[class_indices, 0], X[class_indices, 1], c=colors[ii])
plt.title("full dataset")
plt.show()

# divide data into training and testing
# NOTE! Test data is not needed in solving the exercise
# But it can be interesting to investigating how that behaves w.r.t. training set
# performance and the bounds :)
np.random.seed(42)
order = np.random.permutation(n_tot)
train = order[:100]
# test = order[100:]

Xtr = X[train, :]
ytr = y[train]
# Xtst = X[test, :]
# ytst = y[test]

# ========================================================================
# classifier

# The perceptron algorithm will be encountered later in the course
# How exactly it works is not relevant yet, it's enough to just know it's a binary classifier
from sklearn.linear_model import Perceptron as binary_classifier

# # It can be used like this:
#bc = binary_classifier()
#bc.fit(Xtr, ytr)  # train the classifier on training data
#preds = bc.predict(Xtst)  # predict with test data

# ========================================================================
# setup for analysing the Rademacher complexity

# consider these sample sizes
print_at_n = [20, 50, 100]
# when analysing Rademacher complexity, take always n first samples from training set, n as in this array

delta = 0.05

# todo solution
from sklearn.metrics import zero_one_loss

# Calculating R(h), i.e., the empirical loss
bc = binary_classifier()

Xtr1 = Xtr[:20]
Xtr2 = Xtr[:50]
Xtr3 = Xtr[:100]

ytr1 = ytr[:20]
ytr2 = ytr[:50]
ytr3 = ytr[:100]

bc.fit(Xtr1, ytr1)
pred1 = bc.predict(Xtr1)
l20 = (zero_one_loss(pred1, ytr1))

bc.fit(Xtr2, ytr2)
pred2 = bc.predict(Xtr2)
l50 = (zero_one_loss(pred2, ytr2))

bc.fit(Xtr3, ytr3)
pred3 = bc.predict(Xtr3)
l100 = (zero_one_loss(pred3, ytr3))


# Calculating R(H), i.e., the Rademacher value
order1 = np.random.permutation(n_tot)
shuff = order1[:100]
yrand = y[shuff]

bc = binary_classifier()

Xtr1 = Xtr[:20]
Xtr2 = Xtr[:50]
Xtr3 = Xtr[:100]

ytr1 = yrand[:20]
ytr2 = yrand[:50]
ytr3 = yrand[:100]

bc.fit(Xtr1, ytr1)
pred1 = bc.predict(Xtr1)
r20 = (0.5 - zero_one_loss(pred1, ytr1))

bc.fit(Xtr2, ytr2)
pred2 = bc.predict(Xtr2)
r50 = (0.5 - zero_one_loss(pred2, ytr2))

bc.fit(Xtr3, ytr3)
pred3 = bc.predict(Xtr3)
r100 = (0.5 - zero_one_loss(pred3, ytr3))

# Calculating Rademacher generalisation
intermediate = 2 / delta
intermediate = math.log(intermediate)
print("20: ", r20 + l20 + 3 * math.sqrt(intermediate / (2 * 20)))
print("50: ", r50 + l50 + 3 * math.sqrt(intermediate / (2 * 50)))
print("100: ", r100 + l100 + 3 * math.sqrt(intermediate / (2 * 100)))

# Calculating VC dimension generalisation
vc20 = l20 + math.sqrt(2 * math.log(math.e*20/3) / (20/3)) + math.sqrt(math.log(1/delta) / (2*20))
vc50 = l50 + math.sqrt(2 * math.log(math.e*50/3) / (50/3)) + math.sqrt(math.log(1/delta) / (2*50))
vc100 = l100 + math.sqrt(2 * math.log(math.e*100/3) / (100/3)) + math.sqrt(math.log(1/delta) / (2*100))
print("vc20: ", vc20)
print("vc50: ", vc50)
print("vc100: ", vc100)