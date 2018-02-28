from __future__ import division
import numpy as np
from scipy import optimize, io
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

def cost_function(theta, X, y, lamda):
    m, n = X.shape
    J = np.sum(np.power((X.dot(theta.T)-y),2))/(2*m)  + (lamda/(2*m))*np.sum(np.square(theta[1:]))
    mask = np.eye(len(theta))
    mask[0, 0] = 0
    grad = (1/m)*(X.dot(theta.T) - y).T.dot(X) + (lamda/m)*mask.dot(theta)
    return J, grad

def theta_estimation(X, y, lamda):
    m, n = X.shape
    initial_theta = np.zeros((n, 1))
    result = optimize.minimize(fun=cost_function, x0=initial_theta, args=(X, y, lamda),
                               method='TNC', jac=True, options={'maxiter': 100})
    theta = result.x
    return theta

def learning_curve(X, y, X_val, y_val, lamda):
    m, n = X.shape
    m_val, n_val = X_val.shape
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(1, m + 1):
        theta = theta_estimation(X[:i,], y[:i,], lamda)
        error_train[i-1] = (1/(2*i))*np.sum(np.square(X[:i,].dot(theta) - y[:i,]))
        error_val[i-1] = (1/(2*m_val))*np.sum(np.square(X_val.dot(theta) - y_val))
    return error_train, error_val

def lamda_estimation(X, y, X_val, y_val):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    m, n = X.shape
    m_val, n_val = X_val.shape
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    for i in range(len(lambda_vec)):
        theta = theta_estimation(X, y, lambda_vec[i])
        error_train[i] = (1/(2*m))*np.sum(np.square(X.dot(theta) - y))
        error_val[i] = (1/(2*m_val))*np.sum(np.square(X_val.dot(theta) - y_val))
    return lambda_vec, error_train, error_val

def degree_estimation(data, lamda):
    degree_vec = np.array([2, 3, 4, 5, 6, 7, 8])
    y = data['y'].ravel()
    y_val = data['yval'].ravel()
    error_train = np.zeros(len(degree_vec))
    error_val = np.zeros(len(degree_vec))

    for i in range(len(degree_vec)):
        X_poly, X_val_poly, X_test_poly = poly_features(data, degree_vec[i])
        m, n = np.shape(X_poly)
        m_val, n_val = X_val_poly.shape
        theta = theta_estimation(X_poly, y, lamda)
        error_train[i] = (1/(2*m))*np.sum(np.square(X_poly.dot(theta) - y))
        error_val[i] = (1/(2*m_val))*np.sum(np.square(X_val_poly.dot(theta) - y_val))
    return degree_vec, error_train, error_val

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, ddof=1, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm

def poly_features(data,d):
    X = data['X']
    X_test = data['Xtest']
    X_val = data['Xval']

    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    X_poly = X_poly[:,1:]
    X_poly = feature_normalize(X_poly)
    m, n = X.shape
    X_poly = np.hstack((np.ones((m, 1)), X_poly))

    X_val = poly.fit_transform(X_val)
    X_val = X_val[:, 1:]
    X_val = feature_normalize(X_val)
    m_val, n_val = X_val.shape
    X_val = np.hstack((np.ones((m_val, 1)), X_val))

    X_test = poly.fit_transform(X_test)
    X_test = X_test[:, 1:]
    X_test = feature_normalize(X_test)
    m_test, n_test = X_test.shape
    X_test = np.hstack((np.ones((m_test, 1)), X_test))
    return X_poly, X_val, X_test

def plot_error_for_features(error_train, error_val):
    m = len(error_train)
    plt.plot(range(1, m + 1), error_train, color='b', label='Train')
    plt.plot(range(1, m + 1), error_val, color='r', label='Cross Validation')
    plt.legend(loc='upper right')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')

def plot_error(parameter, error_train, error_val):
    plt.plot(parameter, error_train, color='b', label='Train')
    plt.plot(parameter, error_val, color='r', label='Cross Validation')
    plt.legend(loc='upper right')
    plt.ylabel('Error')

if __name__ == "__main__":

    data = io.loadmat('ex5data1.mat')
    X = data['X']
    y = data['y'].ravel()
    X_test = data['Xtest']
    y_test = data['ytest'].ravel()
    X_val = data['Xval']
    y_val = data['yval'].ravel()

    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    inital_theta = np.ones(n + 1)
    lamda = 1

    J, grad = cost_function(inital_theta, X, y, lamda)
    theta = theta_estimation(X, y, lamda)
    pred = X.dot(theta.T)

    m_val, n_val = X_val.shape
    X_val = np.hstack((np.ones((m_val, 1)), X_val))
    error_train, error_val = learning_curve(X, y, X_val, y_val, lamda)

    plt.figure(1)
    plot_error_for_features(error_train, error_val)

    degree = 5
    X_poly, X_val_poly, X_test_poly = poly_features(data, degree)
    m, n = np.shape(X_poly)
    initial_theta = np.zeros((n, 1))
    theta_poly = theta_estimation(X_poly, y, lamda)
    pred_poly = X_poly.dot(theta_poly.T)

    error_train_poly, error_val_poly = learning_curve(X_poly, y, X_val_poly, y_val, lamda)
    plt.figure(2)
    plot_error_for_features(error_train_poly, error_val_poly)

    lambda_vec, error_train_lamda, error_val_lamda = lamda_estimation(X_poly, y, X_val_poly, y_val)
    plt.figure(3)
    plot_error(lambda_vec, error_train_lamda, error_val_lamda)
    plt.xlabel('lambda')

    degree_vec, error_train_degree, error_val_degree = degree_estimation(data, lamda)
    plt.figure(4)
    plot_error(degree_vec, error_train_degree, error_val_degree)
    plt.xlabel('Degree')

    plt.figure(5)
    plt.plot(X[:,1], y, linestyle='', marker='x', color='r', label='Original')
    plt.plot(X[:,1], pred, linestyle='--', marker='o', color='b', label='Linear Fit')
    plt.plot(X[:,1], pred_poly, linestyle='', marker='*', color='g', label='Poly Fit')
    plt.legend(loc='upper right')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()