import numpy as np
import matplotlib.pyplot as plt
import csv

#sigmoid function
def sigmoid(scores):
	return 1 / (1 + np.exp(-scores))

#log likelihood function: the function we want to make sure is being maximized
def log_likelihood(features, target, weights):
	scores = np.dot(features, weights)
	ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
	return ll

#main logistic regression function
def logistic_regression(features, target, num_steps, learning_rate):
	#initialize weights list
	weights = np.zeros(features.shape[1])

	for step in range(num_steps):
		scores = np.dot(features, weights)
		predictions = sigmoid(scores)

		#update weights with gradient
		output_error_signal = target - predictions
		gradient = np.dot(features.T, output_error_signal)
		weights += learning_rate * gradient

		#print log-likelihood periodically to ensure it is being maximized
		if step % 10000 == 0:
			print(log_likelihood(features, target, weights))

	return weights

#import Iris data
my_file = open('irisMODIFIED.csv', "rt")
iris = csv.reader(my_file)
headers = next(iris)
rownum = 0
X1 = []
X2 = []

#X1 is the feature 2-d list. X2 is a list of the flower class.
for row in iris:
	X1.append (row)
	X2.append (X1[rownum].pop())
	rownum += 1

#put lists into numpy arrays
iris_features = np.array(X1).astype(np.float32)
iris_labels = np.array(X2).astype(np.float32)

#perform logistic regression and print the weights/coefficients
weights = logistic_regression(iris_features, iris_labels, num_steps = 500000, learning_rate = 5e-5)
print(weights)