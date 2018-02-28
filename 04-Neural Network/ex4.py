from __future__ import division
import numpy as np
import pandas as pd
from scipy import optimize, io

def sigmoid(z):
    g =  1/(1+np.exp(-z))
    return g

def sigmoidGradient(z):
    g = sigmoid(z)*(1 - sigmoid(z))
    return g

def cost_function(theta, X, y, input_layer_size, hidden_layer_size, lamda):
    num_labels = len(np.unique(y))
    y = pd.get_dummies(y.ravel()).as_matrix()
    theta1 = np.reshape(theta[0:(hidden_layer_size*(input_layer_size + 1)),], (hidden_layer_size, input_layer_size + 1))
    theta2 = np.reshape(theta[(hidden_layer_size * (input_layer_size + 1)):, ], (num_labels, hidden_layer_size + 1))

    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))

    a1 = X
    z2 = theta1.dot(a1.T)
    a2 = sigmoid(z2.T)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = theta2.dot(a2.T)
    hx = sigmoid(z3)

    J = (-1/m)*np.sum((np.log(hx.T)*y + np.log(1-hx).T*(1-y))) + \
        (lamda/(2*m))*(np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:])))

    d3 = hx.T - y
    d2 = theta2[:,1:].T.dot(d3.T)*sigmoidGradient(z2)

    delta1 = d2.dot(a1)
    delta2 = d3.T.dot(a2)

    theta1_ = np.c_[np.ones((theta1.shape[0], 1)), theta1[:, 1:]]
    theta2_ = np.c_[np.ones((theta2.shape[0], 1)), theta2[:, 1:]]

    theta1_grad = delta1/m + (theta1_*lamda)/m
    theta2_grad = delta2/m + (theta2_*lamda)/m
    grad = np.hstack((theta1_grad.ravel(), theta2_grad.ravel()))
    return J, grad

def predict(theta1, theta2, X):
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    a1 = X
    z2 = theta1.dot(a1.T)
    a2 = sigmoid(z2.T)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = theta2.dot(a2.T)
    hx = sigmoid(z3.T)
    p = np.argmax(hx, axis=1) + 1
    return p

if __name__ == "__main__":
    
    data = io.loadmat('ex4data1.mat')
    X = data['X']
    y = data['y']
    num_labels = len(np.unique(y))

    mat_param = io.loadmat('ex4weights.mat')
    theta1 = mat_param['Theta1']
    theta2 = mat_param['Theta2']

    input_layer_size = np.shape(theta1)[1] - 1
    hidden_layer_size = np.shape(theta2)[1] - 1
    theta = np.hstack((theta1.flatten(), theta2.flatten()))
    lamda = 0
    J, grad = cost_function(theta, X, y, input_layer_size, hidden_layer_size, lamda)

    initial_epsilon = 0.12
    initial_theta1 = np.random.rand(np.shape(theta1)[0], np.shape(theta1)[1])* 2 * initial_epsilon - initial_epsilon
    initial_theta2 = np.random.rand(np.shape(theta2)[0], np.shape(theta2)[1])* 2 * initial_epsilon - initial_epsilon
    initial_theta = np.hstack((initial_theta1.flatten(), initial_theta2.flatten()))
    result = optimize.minimize(fun=cost_function, x0=initial_theta,
                               args=(X, y, input_layer_size, hidden_layer_size, lamda),
                               method='TNC', jac=True, options={'maxiter': 150})

    final_theta = result.x
    final_theta1 = np.reshape(final_theta[0:(hidden_layer_size*(input_layer_size + 1)),],
                              (hidden_layer_size, input_layer_size + 1))
    final_theta2 = np.reshape(final_theta[(hidden_layer_size*(input_layer_size + 1)):,],
                              (num_labels, hidden_layer_size + 1))

    pred = predict(final_theta1, final_theta2, X)
    print ('Training Set Accuracy:', np.mean(pred == y.ravel())*100)