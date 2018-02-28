from __future__ import division
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from sklearn import svm

def plot_data(X, y):
    pos = np.nonzero(y == 1)
    neg = np.nonzero(y == 0)
    plt.plot(X[pos, 0], X[pos, 1], linestyle='', marker='+', color='k')
    plt.plot(X[neg, 0], X[neg, 1], linestyle='', marker='o', color='y')

def visualize_boundary_linear(X, y, clf):
    plot_data(X, y)
    coef = clf.coef_.ravel()
    intercept = clf.intercept_.ravel()

    xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    yp = -1.0 * (coef[0] * xp + intercept[0]) / coef[1]
    plt.plot(xp, yp, linestyle='-', color='b')

def visualize_boundary(X, y, clf):
    plot_data(X, y)
    x1_plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    x2_plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1_plot, x2_plot)
    vals = np.zeros(X1.shape)

    for i in range(X1.shape[1]):
        X_tmp = np.hstack((X1[:, i:i + 1], X2[:, i:i + 1]))
        vals[:, i] = clf.predict(X_tmp)
    plt.contour(X1, X2, vals, levels=[0])

def dataset1():
    data = io.loadmat('ex6data1.mat')
    X = data['X']
    y = data['y'].ravel()

    clf = svm.LinearSVC(C=1)
    clf.fit(X, y)
    print('Accuracy score:', clf.score(X, y))
    plt.figure(1)
    visualize_boundary_linear(X, y, clf)
    plt.show()

def dataset2():
    data = io.loadmat('ex6data2.mat')
    X = data['X']
    y = data['y'].ravel()

    clf = svm.SVC(C=100, kernel='rbf', gamma=10)
    clf.fit(X, y)
    print('Accuracy score:', clf.score(X, y))
    plt.figure(2)
    visualize_boundary(X, y, clf)
    plt.show()

def data3_params(X, y, X_val, y_val):
    C_cands = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    gamma_cands = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    max_score = -1
    C_pick = -1
    gamma_pick = -1
    for C in C_cands:
        for gamma in gamma_cands:
            clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            clf.fit(X, y)
            score = clf.score(X_val, y_val)
            if score > max_score:
                max_score = score
                C_pick = C
                gamma_pick = gamma

    return C_pick, gamma_pick

def dataset3():
    data = io.loadmat('ex6data3.mat')
    X = data['X']
    y = data['y'].ravel()
    X_val = data['Xval']
    y_val = data['yval'].ravel()

    C_pick, gamma_pick = data3_params(X, y, X_val, y_val)

    clf = svm.SVC(C=C_pick, kernel='rbf', gamma=gamma_pick)
    clf.fit(X, y)
    print('Accuracy score:', clf.score(X, y))
    plt.figure(3)
    visualize_boundary(X, y, clf)
    plt.show()

if __name__ == "__main__":

    dataset1()
    dataset2()
    dataset3()