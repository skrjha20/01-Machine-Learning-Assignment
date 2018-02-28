from __future__ import division
import numpy as np
from scipy import optimize

def sigmoid(z):
     g =  1/(1+np.exp(-z))
     return g
    
def cost_function(theta, X, y):
    
    m, n = X.shape
    J = (-1/m)*(y.T.dot(np.log(sigmoid(X.dot(theta.T)))) + (1 - y).T.dot(np.log(1 - sigmoid(X.dot(theta.T)))))
    grad = (1/m)*(sigmoid(X.dot(theta.T)) - y).T.dot(X)
    return J, grad
        
if __name__ == "__main__":
    
    data = np.loadtxt(open("ex2data1.txt", "r"), delimiter=",")
    X = data[:, 0:2]
    y = data[:, 2]
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    theta = np.zeros(n + 1)
    cost, grad = cost_function(theta, X, y)
    theta, nfeval, rc = optimize.fmin_tnc(func=cost_function, x0=theta, args=(X, y), messages=0)
    print(theta)
    
    