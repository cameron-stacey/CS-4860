import numpy as np
import matplotlib.pyplot as plt
import csv

def sigmoid(scores):
	return 1 / (1 + np.exp(-scores))

def log_likelihood(features, target, weights):
	scores = np.dot(features, weights)
	l1 = np.sum(target * scores - np.log(1 + np.exp(scores)))
	return l1

def logistic_regression(features, target, num_steps, learning_rate):
	weights = np.zeros(features.shape[1])

	for step in range(num_steps):
		scores = np.dot(features, weights)
		predictions = sigmoid(scores)

		#Update weights with gradient
		output_error_signal = target - predictions
		gradient = np.dot(features.T, output_error_signal)
		weights += learning_rate * gradient

		#Print log-likelihood every so often
		if step % 10000 == 0:
			print(log_likelihood(features, target, weights))

	return weights

#import Wisconsin Breast Cancer data
my_file = open('wisconsinMODIFIED.csv', "rt")
wisconsin = csv.reader(my_file)
headers = next(wisconsin)
rownum = 0
X1 = []
X2 = []

for row in wisconsin:
	X1.append (row)
	X2.append (X1[rownum].pop())
	rownum += 1

wisconsin_features = np.array(X1).astype(np.float32)
wisconsin_labels = np.array(X2).astype(np.float32)

weights = logistic_regression(wisconsin_features, wisconsin_labels, num_steps = 100000, learning_rate = 5e-5)
print(weights)