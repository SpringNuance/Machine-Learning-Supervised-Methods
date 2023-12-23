from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Preparing the data
X,y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# The model
mlp = MLPClassifier(alpha=1, max_iter=100, random_state=1)
ada = AdaBoostClassifier(random_state=1)
gdc = GradientBoostingClassifier(n_estimators=200, learning_rate=1, max_depth=1, random_state=1)

# Fit model
mlp.fit(X_train, y_train)
ada.fit(X_train, y_train)
gdc.fit(X_train, y_train)

# Use the model to predict the test set
mlp_pred = mlp.predict(X_test)
ada_pred = ada.predict(X_test)
gdc_pred = gdc.predict(X_test)

# Calculate the F1 scores
f1_mlp = f1_score(y_test, mlp_pred)
f1_ada = f1_score(y_test, ada_pred)
f1_gdc = f1_score(y_test, gdc_pred)

# Print the results
print("MLP F1: ", f1_mlp)
print("ADA F1: ", f1_ada)
print("GDC F1: ", f1_gdc)