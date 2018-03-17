from sympy import symbols, var, diff, stats, Function, Derivative, Eq, Poly
from sympy.solvers import solve
import math
import numpy as np
import csv

my_file = open('Folds5x2_pp.csv', "rt")
read = csv.reader(my_file)
X1 = []
X2 = []
X3 = []
X4 = []
X5 = []

headers = next(read)
for row in read:
	X1.append(row[0])
	X2.append(row[1])
	X3.append(row[2])
	X4.append(row[3])
	X5.append(row[4])

data1 = np.array([float(i) for i in X1])
data2 = np.array([float(i) for i in X2])
data3 = np.array([float(i) for i in X3])
data4 = np.array([float(i) for i in X4])
data5 = np.array([float(i) for i in X5])
var('a:5') #declare a-subnumber variables
var('y')
var('x:5')

equation = np.sum((y - a0 - (a1*x1) - (a2*x2) - (a3*x3) - (a4*x4))**2)

#calculate partial derivative with respect to the second variable
partial1 = diff(equation, a0)
partial2 = diff(equation, a1)
partial3 = diff(equation, a2)
partial4 = diff(equation, a3)
partial5 = diff(equation, a4)




#print(partial3)
#solve(partial3, y)


newsol = np.array([])

for x in range(data2.size): #for each variable in our array (or data set) we are going to make it's corresponding x sub value and plug it into the array
	x1 = data1[x] #this is our data at the ith point for each coefficient
	x2 = data2[x]
	x3 = data3[x]
	x4 = data4[x]

	#this is where we multiply them to find the entire coefficient
	A = np.array([[2, 2*x1, 2*x2, 2*x3, 2*x4, -2], [2*x1, 2*(x1**2), 2*x1*x2, 2*x1*x3, 2*x1*x4, -2*2*x1], [2*x2, 2*x1*x2, 2*(x2**2), 2*x2*x3, 2*x2*x4, -2*x2], [2*x3, 2*x1*x3, 2*x2*x3, 2*(x3**2), 2*x3*x4, -2*x3], [2*x4, 2*x1*x4, 2*x2*x4, 2*x3*x4, 2*(x4**2), -2*x4]])
	B = np.array([-2, -2*x1, -2*x2, -2*x3, -2*x4])

	solution = np.linalg.lstsq(A, B)
print (solution)