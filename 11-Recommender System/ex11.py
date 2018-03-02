import numpy as np
from scipy import io, optimize
import matplotlib.pyplot as plt

def plot_data(Y):
    plt.imshow(Y, aspect='auto')
    plt.ylabel('Movies')
    plt.xlabel('Users')

def collaborative_cost_function(params, Y, R, num_users, num_movies, num_features, lamda):
    X = params[0:num_movies * num_features].reshape((num_movies, num_features))
    theta = params[num_movies * num_features:].reshape((num_users, num_features))

    J = 0.5 * np.sum(np.sum(R * np.square(X.dot(theta.T) - Y)))
    X_grad = (R * (X.dot(theta.T) - Y)).dot(theta)
    theta_grad = (R * (X.dot(theta.T) - Y)).T.dot(X)

    J = J + 0.5*lamda*np.sum(np.square(theta)) + 0.5*lamda*np.sum(np.square(X))
    X_grad = X_grad + lamda*X
    theta_grad = theta_grad + lamda*theta
    grad = np.hstack((X_grad.ravel(), theta_grad.ravel()))
    return J, grad

def load_movie_list():
    movie_list = []
    with open("movie_ids.txt") as f:
        for line in f:
            movie_list.append(line[line.index(' ') + 1:].rstrip())
    return movie_list

def normalize_ratings(Y, R):
    m = Y.shape[0]
    Y_mean = np.zeros(m)
    Y_norm = np.zeros(Y.shape)
    for i in range(m):
        idx = np.nonzero(R[i,] == 1)
        Y_mean[i] = np.mean(Y[i, idx])
        Y_norm[i, idx] = Y[i, idx] - Y_mean[i]
    return Y_norm, Y_mean

if __name__ == "__main__":

    # =============== Part 1: Loading movie ratings dataset ================
    print ('Loading movie ratings dataset.')
    data = io.loadmat('ex8_movies.mat')

    # Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
    Y = data['Y']
    # R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
    R = data['R']

    print ('Average rating for movie 1 (Toy Story): {}/5'.format(np.mean(Y[0, np.nonzero(R[0,])])))
    plt.figure(1)
    plot_data(Y)

    # ============ Part 2: Collaborative Filtering Cost Function ===========
    # Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
    data = io.loadmat('ex8_movieParams.mat')
    X = data['X']
    theta = data['Theta']
    num_users = data['num_users'].ravel()[0]
    num_movies = data['num_movies'].ravel()[0]
    num_features = data['num_features'].ravel()[0]

    # Reduce the data set size so that this runs faster
    num_users = 4
    num_movies = 5
    num_features = 3
    X = X[0:num_movies, 0:num_features]
    theta = theta[0:num_users, 0:num_features]
    Y = Y[0:num_movies, 0:num_users]
    R = R[0:num_movies, 0:num_users]

    J, grad = collaborative_cost_function(np.hstack((X.flatten(), theta.flatten())),
                                          Y, R, num_users, num_movies, num_features, lamda=1.5)
    print ('Cost at loaded parameters:', J)

    # ============== Part 3: Entering ratings for a new user ===============
    movie_list = load_movie_list()

    # Initialize my ratings
    my_ratings = np.zeros(len(movie_list), dtype=np.int)
    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5

    print ('New user ratings:')
    for i in np.argwhere(my_ratings > 0).ravel():
        print ('Rated {} for {}'.format(my_ratings[i], movie_list[i]))

    # ================== Part 4: Learning Movie Ratings ====================
    print ('Training collaborative filtering...')
    data = io.loadmat('ex8_movies.mat')

    # Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
    Y = data['Y']
    # R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
    R = data['R']

    # Add our own ratings to the data matrix
    Y = np.hstack((my_ratings.reshape(len(movie_list), 1), Y))
    R = np.hstack((my_ratings.reshape(len(movie_list), 1) != 0, R))

    # Normalize Ratings
    Y_norm, Y_mean = normalize_ratings(Y, R)

    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10

    # Set Initial Parameters (Theta, X)
    X = np.random.randn(num_movies, num_features)
    theta = np.random.randn(num_users, num_features)

    initial_parameters = np.hstack((X.flatten(), theta.flatten()))

    # Set Regularization
    l = 10
    result = optimize.minimize(fun=collaborative_cost_function, x0=initial_parameters,
                                args=(Y_norm, R, num_users, num_movies, num_features, l),
                                method='CG', jac=True, options={'maxiter': 150})

    X = result.x[0:num_movies * num_features].reshape((num_movies, num_features))
    theta = result.x[num_movies * num_features:].reshape((num_users, num_features))

    print ('Recommender system learning completed.')

    # ================== Part 5: Recommendation for you ====================
    p = X.dot(theta.T)
    my_predictions = p[:, 0] + Y_mean
    idx = np.argsort(my_predictions)[::-1]
    print ('Top recommendations for you:')
    for i in range(10):
        print ('Predicting rating {0:.1f} for movie {1:s}'.format(my_predictions[idx[i]], movie_list[idx[i]]))

    print ('Original ratings provided:')
    for i in np.argwhere(my_ratings > 0).ravel():
        print ('Rated {} for {}'.format(my_ratings[i], movie_list[i]))
