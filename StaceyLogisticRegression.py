import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV

#load in dataset
iris = load_iris()
x_iris = iris.data
y_iris = iris.target

#create a separate array for each feature
a = x_iris[:,0]
b = x_iris[:,1]
c = x_iris[:,2]
d = x_iris[:,3]

#scale the data
scaler = StandardScaler()
scaler.fit_transform(x_iris, y_iris)
x_scaled = scaler.transform(x_iris)

#make modified features sets
x_squares =- np.vstack(([a**2], [b**2], [c**2], [d**2])).T
x_multi = np.vstack((a*b, a*c, a*d, b*c, b*d, c*d)).T

#make split for datasets
(x_tr_o, x_ts_o, y_tr_o, y_ts_o) = train_test_split(x_iris, y_iris, stratify=y_iris, test_size= 0.3)
(x_tr_sc, x_ts_sc, y_tr_sc, y_ts_sc) = train_test_split(x_scaled, y_iris, stratify=y_iris, test_size= 0.3)
(x_tr_sq, x_ts_sq, y_tr_sq, y_ts_sq) = train_test_split(x_squares, y_iris, stratify=y_iris, test_size= 0.3)
(x_tr_m, x_ts_m, y_tr_m, y_ts_m) = train_test_split(x_multi, y_iris, stratify=y_iris, test_size= 0.3)

#create estimator class
estimator = LogisticRegression()
paramgrid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'penalty': ['l1', 'l2']}
optimizer = GridSearchCV(estimator, paramgrid, cv = 10)

#original data fit
optimizer.fit(x_tr_o, y_tr_o)
predict = optimizer.best_estimator_.predict(x_ts_o)
z_o = accuracy_score(y_ts_o, predict)

#scaled data fit
optimizer.fit(x_tr_sc, y_tr_sc)
predict = optimizer.best_estimator_.predict(x_ts_sc)
z_sc = accuracy_score(y_ts_sc, predict)

#squares data fit
optimizer.fit(x_tr_sq, y_tr_sq)
predict = optimizer.best_estimator_.predict(x_ts_sq)
z_sq = accuracy_score(y_ts_sq, predict)

#multiplied pairs data fit
optimizer.fit(x_tr_m, y_tr_m)
predict = optimizer.best_estimator_.predict(x_ts_m)
z_m = accuracy_score(y_ts_m, predict)

print ('Accuracy score for original:', z_o)
print ('Accuracy score for scaled:', z_sc)
print ('Accuracy score for squares:', z_sq)
print ('Accuracy score for multi:', z_m)