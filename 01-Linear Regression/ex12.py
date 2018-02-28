from __future__ import division
import pandas as pd
import numpy as np

def gradient_descent(X,y,theta,alpha):
    
    J1 = 0
    m = len(y)     
    converged = False
    while not converged:
        J2 = np.sum(np.power((X.dot(theta.T)-y),2))/(2*m) 
        if abs(J1-J2) > 0.001:
            J1 = J2
            grad = (alpha/m)*(X.dot(theta.T) - y).T.dot(X)
            theta = theta - grad
        else:
            converged = True
    return theta, J2
        
if __name__ == "__main__":
    
    data = pd.read_csv('ex1data2.txt', sep=",", header=None)
    data.columns = ['X1', 'X2', 'y']
    
    data['X1'] = (data['X1']-np.mean(data['X1']))/np.std(data['X1'])
    data['X2'] = (data['X2']-np.mean(data['X2']))/np.std(data['X2'])
    data['y'] = (data['y']-np.mean(data['y']))/np.std(data['y'])

    data['X0'] = 1
    X = data[['X0','X1', 'X2']]
    y = data['y']
        
    X = np.array(X)
    y = np.array(y)
    iterations = 10
    alpha = 0.01
    theta = np.array([0, 0, 0])
    theta, cost = gradient_descent(X,y,theta,alpha)
    print(theta)
    print(cost)

    #Normal Equation
    theta_neq = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    print(theta_neq)
    
