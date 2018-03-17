import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression

#import csv file
dataFrame = pd.read_csv('Folds5x2_pp.csv')

#drop our target variable from the data
features = dataFrame.drop('PE', axis = 1)

#perform a linear regression
lm = LinearRegression()
lm.fit(features, dataFrame.PE)

#Print coefficients
print ('Estimated intercept coefficient: ', lm.intercept_)
featureList = pd.DataFrame(list(zip(features.columns, lm.coef_)), columns = ['features','estCoefficients'])
print('\n', featureList)

#Mean Squared Error and R^2
mseFull = np.mean((dataFrame.PE - lm.predict(features)) ** 2)
print ("\nMean squared error:", mseFull)
rSquared = lm.score(features, dataFrame.PE)
print ("Coefficient of determination:", rSquared)

#Plot
plt.scatter(dataFrame.PE, lm.predict(features))
plt.xlabel("Actual Energy Output")
plt.ylabel("Predicted Energy Output")
plt.title("Energy Output vs Predicted Energy Output")
plt.show()