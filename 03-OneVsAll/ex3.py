from __future__ import division
import numpy as np
from scipy import optimize, io
from sklearn.preprocessing import PolynomialFeatures

def sigmoid(z):
     g =  1/(1+np.exp(-z))
     return g
    
def cost_function(theta, X, y, lamda):
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    
    J = (-1/m)*(y.T.dot(np.log(sigmoid(X.dot(theta.T)))) + (1 - y).T.dot(np.log(1 - sigmoid(X.dot(theta.T))))) + \
        (lamda/(2*m))*np.sum(np.power(np.eye(len(theta)).dot(theta),2))
    grad = (1/m)*(sigmoid(X.dot(theta.T)) - y).T.dot(X) - (lamda/m)*np.eye(len(theta)).dot(theta)
    return J, grad
        
def one_vs_all(theta, X, y, num_labels, lamda):
    m, n = X.shape    
    all_theta = np.zeros((num_labels, n + 1))
    
    for i in range(0, 10):
        label = 10 if i == 0 else i
        result = optimize.minimize(fun=cost_function, x0=theta, args=(X, (y==label).astype(int), lamda),
                                   method='TNC', jac=True)
        print ('one_vs_all(): label =', label, ', success =', result.success)
        all_theta[i, :] = result.x
    return all_theta

def predict_one_vs_all(all_theta, X):
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    p = np.argmax(X.dot(all_theta.T), axis=1)
    p[p == 0] = 10
    return p

if __name__ == "__main__":
    
    data = io.loadmat('ex3data1.mat')
    X = data['X']
    y = data['y'].ravel()    
    
    m, n = X.shape
    num_labels = len(np.unique(y))    
    theta = np.zeros(n + 1)    
    lamda = 0.1
    
    J, grad = cost_function(theta, X, y, lamda)
    print (J)
    all_theta = one_vs_all(theta, X, y, num_labels, lamda)

    pred = predict_one_vs_all(all_theta, X)
    print ('Training Set Accuracy:', np.mean(pred == y) * 100)