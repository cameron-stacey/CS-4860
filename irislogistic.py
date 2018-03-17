import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import pandas as pd

#import dataset
dataFrame = pd.read_csv('iris.csv')
features = dataFrame.drop('flowerclass', axis = 1)

#create logistic regression model
logreg = linear_model.LogisticRegression()
logreg.fit(features, dataFrame.flowerclass)

#print results
print(logreg.intercept_)
featureList = pd.DataFrame(list(zip(features.columns, logreg.coef_)), columns = ['features','estCoefficients'])
print(featureList)