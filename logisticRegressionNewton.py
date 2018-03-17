import numpy as np
from numpy import genfromtxt
import pandas as pd

#sigmoid function (hypothesis)
def sigmoid(x, Θ_1, Θ_2):
	z = (Θ_1*x + Θ_2).astype("float_")
	return 1.0 / (1.0 + np.exp(-z))

#log-likelihood function
def log_likelihood(x, y, Θ_1, Θ_2):
	sigmoid_probs = sigmoid(x, Θ_1, Θ_2)
	return np.sum(y * np.log(sigmoid_probs) + (1 - y) * np.log(1 - sigmoid_probs))

#gradient implementation
def gradient(x, y, Θ_1, Θ_2):
	sigmoid_probs = sigmoid(x, Θ_1, Θ_2)
	return np.array([[np.sum((y - sigmoid_probs) * x), np.sum(y - sigmoid_probs) * 1]])

#hessian implementation
def hessian(x, y, Θ_1, Θ_2):
	sigmoid_probs = sigmoid(x, Θ_1, Θ_2)
	d1 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x * x)
	d2 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x * 1)
	d3 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * 1 * 1)
	H = np.array([[d1, d2],[d2, d3]])
	return H

def newtons_method(x, y):                                                             
    """
    :param x (np.array(float)): feature/input vector
    :param y (np.array(boolean)): output/result vector
    :returns: np.array of logreg's parameters after convergence, [Θ_1, Θ_2]
    """

    # Initialize log_likelihood & parameters                                                                   
    Θ_1 = 15.1                                                                     
    Θ_2 = -.4 # The intercept term                                                                 
    Δl = np.Infinity                                                                
    l = log_likelihood(x, y, Θ_1, Θ_2)                                                                 
    # Convergence Conditions                                                        
    δ = .0000000001                                                                 
    max_iterations = 15                                                            
    i = 0                                                                           
    while abs(Δl) > δ and i < max_iterations:                                       
        i += 1                                                                      
        g = gradient(x, y, Θ_1, Θ_2)                                                      
        hess = hessian(x, y, Θ_1, Θ_2)                                                 
        H_inv = np.linalg.inv(hess)                                                 
        # @ is syntactic sugar for np.dot(H_inv, g.T)¹
        Δ = H_inv @ g.T                                                             
        ΔΘ_1 = Δ[0][0]                                                              
        ΔΘ_2 = Δ[1][0]                                                              
                                                                                    
        # Perform our update step                                                    
        Θ_1 += ΔΘ_1                                                                 
        Θ_2 += ΔΘ_2                                                                 
                                                                                    
        # Update the log-likelihood at each iteration                                     
        l_new = log_likelihood(x, y, Θ_1, Θ_2)                                                      
        Δl = l - l_new                                                           
        l = l_new                                                                
    return np.array([Θ_1, Θ_2])

#read in data, and replace output/result column with booleans
my_data = pd.read_csv('iris.csv')
my_data.replace({'flowerclass': {'Iris-setosa': True, 'Iris-virginica': False, 
	'Iris-versicolor': False}}, inplace = True)
features = my_data.drop('flowerclass', axis = 1)
testVar = newtons_method(features, my_data.flowerclass)