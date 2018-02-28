import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'bx')
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')

def visualize_fit(X, mu, sigma2):
    l = np.arange(0, 30.5, 0.5)
    X1, X2 = np.meshgrid(l, l)
    X_tmp = np.vstack((X1.ravel(), X2.ravel())).T
    Z = multivariate_gaussian(X_tmp, mu, sigma2)
    Z.resize(X1.shape)
    plt.plot(X[:, 0], X[:, 1], 'bx')
    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, 10.0 ** np.arange(-20, 0, 3))

def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    sigma2 = np.var(X, axis=0)
    return mu, sigma2

def multivariate_gaussian(X, mu, sigma2):
    m,n = np.shape(X)
    if len(sigma2.shape) == 1:
        sigma2 = np.diag(sigma2)
    p = np.power((2*np.pi),(-n/2))*np.power(np.linalg.det(sigma2),-0.5)*\
        np.exp(-0.5*np.sum(((X-mu).dot(np.linalg.pinv(sigma2))* (X-mu)),axis=1))
    return p

def select_threshold(y_val, p_val):
    step_size = (np.max(p_val) - np.min(p_val)) / 1000
    best_epsilon = 0.0
    best_F1 = 0.0
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = p_val < epsilon
        tp = np.sum(predictions[np.nonzero(y_val == True)])
        fp = np.sum(predictions[np.nonzero(y_val == False)])
        fn = np.sum(y_val[np.nonzero(predictions == False)] == True)
        if tp != 0:
            prec = 1.0 * tp / (tp + fp)
            rec = 1.0 * tp / (tp + fn)
            F1 = 2.0 * prec * rec / (prec + rec)
            if F1 > best_F1:
                best_F1 = F1
                best_epsilon = epsilon
    return best_epsilon, best_F1

if __name__ == "__main__":
    # ================== Part 1: Load Example Dataset  ===================
    print ('Visualizing example dataset for outlier detection.')
    data = io.loadmat('ex8data1.mat')
    X = data['X']
    X_val = data['Xval']
    y_val = data['yval'].ravel()
    plt.figure(1)
    plot_data(X)

    # ================== Part 2: Estimate the dataset statistics ===================
    print ('Visualizing Gaussian fit.')
    mu, sigma2 = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, sigma2)
    plt.figure(2)
    visualize_fit(X, mu, sigma2)

    # ================== Part 3: Find Outliers ===================
    p_val = multivariate_gaussian(X_val, mu, sigma2)
    epsilon, F1 = select_threshold(y_val, p_val)

    print ('Best epsilon found using cross-validation:', epsilon)
    print ('Best F1 on Cross Validation Set:', F1)

    outliers = np.nonzero(p < epsilon)
    plt.figure(2)
    plt.scatter(X[outliers, 0], X[outliers, 1], facecolors='none', edgecolors='r', s=100)
    plt.show()

    # ================== Part 4: Multidimensional Outliers ===================
    # Loads the second dataset.
    mat_data = io.loadmat('ex8data2.mat')
    X = mat_data['X']
    X_val = mat_data['Xval']
    y_val = mat_data['yval'].ravel()

    mu, sigma2 = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, sigma2)
    p_val = multivariate_gaussian(X_val, mu, sigma2)
    epsilon, F1 = select_threshold(y_val, p_val)
    print('Best epsilon found using cross-validation:', epsilon)
    print('Best F1 on Cross Validation Set:', F1)
    print ('# Outliers found:', np.sum(p < epsilon))