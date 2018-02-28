import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import cv2

def k_means_init_centroids(X, K):
    rand_idx = np.random.permutation(X.shape[0])
    centroids = X[rand_idx[0:K], :]
    return centroids

def find_closest_centroids(X, centroids):
    m = X.shape[0]
    idx = np.zeros(m)
    for i in range(m):
        dist = np.sum(np.square(centroids - X[i, :]), axis=1)
        idx[i] = np.argmin(dist)
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        x = X[idx == k]
        centroids[k, :] = np.mean(x, axis=0)
    return centroids

def run_k_means(X, initial_centroids, max_iters):
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    history_centroids = np.zeros((max_iters, centroids.shape[0], centroids.shape[1]))
    idx = np.zeros(X.shape[0])

    for i in range(max_iters):
        print ('K-Means iteration {}/{}'.format(i + 1, max_iters))
        history_centroids[i, :] = centroids
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

if __name__ == "__main__":

    # ==================== Part 1: Find Closest Centroids ====================
    print ('Finding closest centroids...')
    data = io.loadmat('ex7data2.mat')
    X = data['X']
    K = 3
    initial_centroids = k_means_init_centroids(X,K)
    idx = find_closest_centroids(X, initial_centroids)

    # ===================== Part 2: Compute Means =========================
    print ('Computing centroids means...')
    centroids = compute_centroids(X, idx, K)
    print(centroids)

    # =================== Part 3: K-Means Clustering ======================
    print ('Running K-Means clustering on example dataset...')
    max_iters = 10
    centroids, idx = run_k_means(X, initial_centroids, max_iters)
    print ('K-Means Done.')

    # ============= Part 4: K-Means Clustering on Pixels ===============
    print ('Running K-Means clustering on pixels from an image.')
    A = cv2.imread('bird_small.png')
    A = A.astype(float)/255
    img_size = A.shape

    # Reshape the image into an Nx3 matrix where N = number of pixels.
    X = A.reshape([img_size[0] * img_size[1], img_size[2]])
    K = 16
    initial_centroids = k_means_init_centroids(X, K)
    centroids, idx = run_k_means(X, initial_centroids, max_iters)

    # ================= Part 5: Image Compression ======================
    print ('Applying K-Means to compress an image.')
    idx = find_closest_centroids(X, centroids)

    # Recover the image from the indices (idx) by mapping each pixel to the centroid value.
    X_recovered = centroids[idx.astype(int), :]

    # Reshape the recovered image into proper dimensions
    X_recovered = X_recovered.reshape(img_size)

    fig = plt.figure()
    # Display the original image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(A)
    ax1.set_title('Original')
    # Display compressed image side by side
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(X_recovered)
    ax2.set_title('Compressed, with {} colors.'.format(K))
    plt.show()