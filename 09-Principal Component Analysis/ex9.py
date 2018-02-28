import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import cv2

def plot_data(X):
    plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b')
    plt.xlim(0.5, 6.5)
    plt.ylim(2, 8)
    plt.gca().set_aspect('equal', adjustable='box')

def plot_projection(X_norm,X_rec):
    plt.scatter(X_rec[:, 0], X_rec[:, 1], facecolors='none', edgecolors='r')
    for i in range(X_norm.shape[0]):
        draw_line(X_norm[i, :], X_rec[i, :], dash=True)
    axes = plt.gca()
    axes.set_xlim([-4, 3])
    axes.set_ylim([-4, 3])
    axes.set_aspect('equal', adjustable='box')

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def draw_line(p1, p2, dash=False):
    if dash:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color='k')
    else:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k')

def pca(X):
    m, n = X.shape
    sigma = X.T.dot(X)/m
    U, S, V = np.linalg.svd(sigma)
    return U, S, V

def project_data(X, U, K):
    Z = X.dot(U[:,0:K])
    return Z

def recover_data(Z, U, K):
    X_rec = Z.dot(U[:, 0:K].T)
    return X_rec

if __name__ == "__main__":

    # ==================== Part 1: Load Example Dataset ====================
    print ('Visualizing example dataset for PCA...')
    data = io.loadmat('ex7data1.mat')
    X = data['X']
    plt.figure(1)
    plot_data(X)

    # =============== Part 2: Principal Component Analysis  ===============
    print ('Running PCA on example dataset.')
    X_norm, mu, sigma = feature_normalize(X)
    U, S, V = pca(X_norm)
    print ('Top eigenvector:')
    print ('U = ', U[:, 0])

    # =================== Part 3: Dimension Reduction ===================
    print ('Dimension reduction on example dataset.')
    # Project the data onto K = 1 dimension
    K = 1
    Z = project_data(X_norm, U, K)
    print ('Projection of the first example: ', Z[0])

    X_rec = recover_data(Z, U, K)
    print ('Approximation of the first example:', X_rec[0])

    plt.figure(2)
    plot_projection(X_norm,X_rec)
    plt.show()

    # =============== Part 4: PCA on Face Data ===========================
    print ('Loading face dataset.')
    data = io.loadmat('ex7faces.mat')
    X = data['X']
    X_norm, mu, sigma = feature_normalize(X)
    U, S, V = pca(X_norm)

    # ============= Part 5: Dimension Reduction for Faces =================
    print ('Dimension reduction for face dataset.')
    K = 100
    Z = project_data(X_norm, U, K)
    print ('The projected data Z has a size of:', Z.shape)
    X_rec = recover_data(Z, U, K)