from __future__ import division
import pandas as pd
import numpy as np

def gradient_descent(X,y,theta,alpha):
    
    J1 = 0
    m = len(y)     
    converged = False
    while not converged:
        J2 = np.sum(np.power((X.dot(theta.T)-y),2))/(2*m) 
        if abs(J1-J2) > 0.0000001:
            J1 = J2
            grad = (alpha/m)*(X.dot(theta.T) - y).T.dot(X)
            theta = theta - grad
        else:
            converged = True
    return theta, J2
        
if __name__ == "__main__":
    
    data = np.loadtxt(open("ex1data1.txt", "r"), delimiter=",")
    X = data[:, 0:1]
    y = data[:, 1]
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    theta = np.zeros(n + 1)
    alpha = 0.01
    theta, cost = gradient_descent(X,y,theta,alpha)
    print(theta)
    print(cost)

    #Normal Equation
    theta_neq = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    print(theta_neq)
    