from __future__ import division
import numpy as np
from scipy import optimize
from sklearn.preprocessing import PolynomialFeatures

def sigmoid(z):
     g =  1/(1+np.exp(-z))
     return g

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, ddof=1, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm

def cost_function(theta, X, y, lamda):
    
    m, n = X.shape
    J = (-1/m)*(y.T.dot(np.log(sigmoid(X.dot(theta.T)))) + (1 - y).T.dot(np.log(1 - sigmoid(X.dot(theta.T))))) + \
        (lamda/(2*m))*np.sum(np.power(np.eye(len(theta)).dot(theta),2))
    mask = np.eye(len(theta))
    mask[0, 0] = 0
    grad = (1/m)*(sigmoid(X.dot(theta.T)) - y).T.dot(X) - (lamda/m)*mask.dot(theta)
    return J, grad
        
if __name__ == "__main__":
    
    data = np.loadtxt(open("ex2data2.txt", "r"), delimiter=",")
    X = data[:, 0:2]
    y = data[:, 2]
    
    poly = PolynomialFeatures(degree=2)
    X = poly.fit_transform(X)
    X = X[:,1:]
    X = feature_normalize(X)
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    theta = np.zeros(n + 1)
    lamda = 1
    cost, grad = cost_function(theta, X, y, lamda)
    theta, nfeval, rc = optimize.fmin_tnc(func=cost_function, x0=theta, args=(X, y, lamda), messages=0)
    print(theta)
    