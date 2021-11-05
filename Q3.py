import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Define random seed for numpy operations
seed = 1939671
np.random.seed(seed)

# Load data set
df = pd.read_csv('DATA.csv')

# Split data in train and test
train, test = train_test_split(df, test_size=0.25, random_state=seed)

# Extract features for train
X = np.array(train[['x1', 'x2']])

# Extract labels for test
y = np.array(train['y'])

# Extract features for train
X_test = np.array(test[['x1', 'x2']])

# Extract labels for test
y_test = np.array(test['y'])


def tanh(s, sigma):
    """
    Compute the tanh as in the Q1_1

    :param s: input variable
    :param sigma: hyperparameter

    :return tanh(x)
    """
    prod = 2 * sigma * s
    return (np.exp(prod) - 1) / (np.exp(prod) + 1)


def feedforward(X, W, b, v, sigma):
    """
    Compute the forward pass of the MLP.

    :param X: observations
    :param W: first layer weights
    :param b: bias
    :param v: output layer weights
    :param sigma: hyperparameter for tanh

    :return predictions
    """
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = np.dot(activation, v)

    return pred

def backpropagation_step1(x0, funcArgs):
    """
    Implement backpropagation to get the gradients wrt v
    :param x0: Contains initialization for v (output layer weights)
    :param funcArgs: list of additional parameters. Specifically:
            X: features
            y: labels
            sigma: hyperparameter for tanh
            N: Number of units
            rho:
            W: first layer weights
            b: bias
    :return the result of the minimization inside of the "res" object.
    """
    X = funcArgs[0]
    y = funcArgs[1]
    sigma = funcArgs[2]
    N = funcArgs[3]
    rho = funcArgs[4]
    W = funcArgs[5].reshape((X.shape[1], N))
    b = funcArgs[6]

    v = x0

    P = len(y)

    linear_layer = (np.dot(X, W) + b)
    a_2 = tanh(linear_layer, sigma)
    dJdf = (1 / P) * (np.dot(a_2, v) - y)
    dtanh = 1 - tanh(linear_layer, sigma) ** 2

    dW1_1 = np.tensordot(dJdf, np.transpose(v), axes=0)
    dW1_2 = dW1_1 * dtanh

    dv = np.dot(dJdf, a_2) + rho * v
    # db = np.sum(dW1_2, axis=0) + rho * b
    # dW = np.tensordot(np.transpose(X), dW1_2, axes=1) + rho * W

    return np.concatenate((dv), axis=None)

def backpropagation_step2(x0, funcArgs):
    """
    Implement backpropagation to get the gradients wrt W and b
    :param x0: Contains initialization for W and b(first layer weights and bias)
    :param funcArgs: list of additional parameters. Specifically:
            X: features
            y: labels
            sigma: hyperparameter for tanh
            N: Number of units
            rho:

            v: output layer weights

    :return the result of the minimization inside of the "res" object.
    """
    X = funcArgs[0]
    y = funcArgs[1]
    sigma = funcArgs[2]
    N = funcArgs[3]
    rho = funcArgs[4]
    v = funcArgs[5]

    P = len(y)

    W = x0[:int(X.shape[1] * N)].reshape((X.shape[1], N))
    b = x0[int(X.shape[1] * N):int(X.shape[1] * N + N)]

    linear_layer = (np.dot(X, W) + b)
    a_2 = tanh(linear_layer, sigma)
    dJdf = (1 / P) * (np.dot(a_2, v) - y)
    dtanh = 1 - tanh(linear_layer, sigma) ** 2

    dW1_1 = np.tensordot(dJdf, np.transpose(v), axes=0)
    dW1_2 = dW1_1 * dtanh

    db = np.sum(dW1_2, axis=0) + rho * b
    dW = np.tensordot(np.transpose(X), dW1_2, axes=1) + rho * W

    return np.concatenate((dW, db), axis=None)

def loss_step1(x0, funcArgs, test=False):
    """
    Compute the loss of the MLP for the first step (with respect to v).

    :param x0: Contains initialization for v (output layer weights)
    :param funcArgs: list of additional parameters. Specifically:
            X: features
            y: labels
            sigma: hyperparameter for tanh
            N: Number of units
            rho:
            W: first layer weights
            b: bias

    :return the result of the loss inside of the "res" object.
        """
    X = funcArgs[0]
    y = funcArgs[1]
    sigma = funcArgs[2]
    N = funcArgs[3]
    rho = funcArgs[4]
    W = funcArgs[5].reshape((X.shape[1], N))
    b = funcArgs[6]

    v = x0

    P = len(y)
    norm = np.linalg.norm(x0)
    pred = feedforward(X, W, b, v, sigma)
    if test:
        res = ((np.sum((pred - y) ** 2)) * P ** (-1)) * 0.5
    else:
        res = ((np.sum((pred - y) ** 2)) * P ** (-1) + rho * norm ** 2) * 0.5

    return res

def loss_step2(x0, funcArgs, test=False):
    """
    Compute the loss of the MLP for the second step (with respect to w and b).

    :param x0: Contains initialization for W and b(first layer weights and bias)
    :param funcArgs: list of additional parameters. Specifically: X, y, sigma, N, rho, W, b,
            X: features
            y: labels
            sigma: hyperparameter for tanh
            N: Number of units
            rho:
            v: output layer weights


    :return the result of the loss inside of the "res" object.
    """
    X = funcArgs[0]
    y = funcArgs[1]
    sigma = funcArgs[2]
    N = funcArgs[3]
    rho = funcArgs[4]
    v = funcArgs[5]

    W = x0[:int(X.shape[1] * N)].reshape((X.shape[1], N))
    b = x0[int(X.shape[1] * N):int(X.shape[1] * N + N)]

    P = len(y)
    norm = np.linalg.norm(x0)
    pred = feedforward(X, W, b, v, sigma)
    if test:
        res = ((np.sum((pred - y) ** 2)) * P ** (-1)) * 0.5
    else:
        res = ((np.sum((pred - y) ** 2)) * P ** (-1) + rho * norm ** 2) * 0.5

    return res

def loss_test(X, y, sigma, N, rho, W, b, v):
    """
    Compute the loss of the MLP for the test data set
    :param X: features
    :param y: labels
    :param sigma: hyperparameter for tanh
    :param N: Number of units
    :param rho:
    :param W: first layer weights
    :param b: bias
    :param v: output layer weights

    :return the result of the loss inside of the "res" object.
    """
    P = len(y)
    pred = feedforward(X, W, b, v, sigma)
    res = ((np.sum((pred - y) ** 2)) * P ** (-1)) * 0.5

    return res


def feedforwardplot(x1, x2, W, b, v, sigma):
    """
    Compute the forward pass of the MLP on a tuple (x1, x2).

    :param x1: first coordinate
    :param x2: second coordinate
    :param W: first layer weights
    :param b: bias
    :param v: output layer weights.
    :param sigma: hyperparameter for tanh

    :return predicted value f(x1,x2)
    """
    X = np.array([x1, x2])
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = np.dot(activation, v)

    return pred

def train_step1(X, y, sigma, N, rho, W_init, b_init, v_init, max_iter=1000,
          tol=1e-5, method='CG', func=loss_step1):
    """
    Train the MLP for the given hyperparameters for the step 1
    :param X: features
    :param y: labels
    :param sigma: hyperparameter for tanh
    :param N: Number of units
    :param rho:
    :param W_init: first layer weights
    :param b_init: bias
    :param v_init: output layer weights
    :param max_iter: Maximum number of iterations while minimizing
    :param tol: Tolerance for convergence
    :param method: Method for minimization
    :param func: Function to minimize

    :return the result of the minimization inside of the "res" object.
    """
    x0 = np.concatenate(v_init, axis=None)

    funcArgs = [X, y, sigma, N, rho, W_init, b_init]

    res = minimize(func,
                   x0,
                   args=funcArgs,
                   method=method,
                   tol=tol,
                   jac=backpropagation_step1,
                   options={'maxiter': max_iter})

    return res

def train_step2(X, y, sigma, N, rho, W_init, b_init, v, max_iter=1000,
          tol=1e-5, method='CG', func=loss_step2):
    """
    Train the MLP for the given hyperparameters for the step 1
    :param X: features
    :param y: labels
    :param sigma: hyperparameter for tanh
    :param N: Number of units
    :param rho:
    :param W_init: first layer weights
    :param b_init: bias
    :param v: output layer weights
    :param max_iter: Maximum number of iterations while minimizing
    :param tol: Tolerance for convergence
    :param method: Method for minimization
    :param func: Function to minimize

    :return the result of the minimization inside of the "res" object.
    """
    x0 = np.concatenate((W_init, b_init), axis=None)

    funcArgs = [X, y, sigma, N, rho, v]

    res = minimize(func,
                   x0,
                   args=funcArgs,
                   method=method,
                   tol=tol,
                   jac=backpropagation_step2,
                   options={'maxiter': max_iter})

    return res

def plotting(W, b, v, sigma):
    """
    Plot the estimated function with given parameters
    :param W: first layer weights
    :param b: bias
    :param v: output layer weights
    :param sigma: hyperparameter for tanh

    :return Show plot
    """
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    # create the grid
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-2, 2, 50)
    X_plot, Y_plot = np.meshgrid(x, y)

    Z = []
    for x1 in x:
        z = []
        for x2 in y:
            z.append(feedforwardplot(x1, x2, W, b, v, sigma))
        Z.append(z)
    Z = np.array(Z)

    ax.plot_surface(X_plot, Y_plot, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('F(x) learnt from MLP')
    plt.show()

# Define the best value obtained for N
N_best = 60

# Define the best value obtained for Sigma
sigma_best = 1

# Define the best value obtained for Rho
rho_best = 1e-05

# Set the number of random trials for W and b
trials = 12
best_val_loss = 1000

start0 = time.time()
# Iterate /trials/ times
for _ in tqdm(range(trials)):

    # Get random initialization for W
    W = np.random.randn(X.shape[1], N_best)

    # Get random initialization for b
    b = np.random.randn(N_best)

    # Get random initialization for v
    v = np.random.randn(N_best)

    ########################################
    ##### step1: convex minimization wrt v
    ########################################

    start = time.time()
    res_step1 = train_step1(X, y, sigma=sigma_best,
                N=N_best, rho=rho_best,
                W_init=W, b_init=b, v_init=v,
                max_iter=4000, tol=1e-6,
                method='CG', func=loss_step1)
    stop1 = time.time()
    # Extract the values for v after optimization
    v = res_step1.x

    ##################################################
    ##### step2: non-convex minimization wrt w and b
    ##################################################
    start2 = time.time()
    res_step2 = train_step2(X, y, sigma=sigma_best,
                N=N_best, rho=rho_best,
                W_init=W, b_init=b, v=v,
                max_iter=4000, tol=1e-6,
                method='CG', func=loss_step2)
    stop2 = time.time()

    # Get the loss for validation set
    funcArgs_test = [X_test, y_test, sigma_best, N_best, rho_best, v]

    current_val_loss = loss_step2(res_step2.x, funcArgs_test, test=True)

    # Extract the loss for the train set
    current_train_loss = res_step2.fun

    # Extract the values for W and b after optimization
    best_params = res_step2.x
    W = best_params[:X.shape[1] * N_best].reshape((X.shape[1], N_best))
    b = best_params[X.shape[1] * N_best:X.shape[1] * N_best + N_best]

    # Retain the best values
    if current_val_loss < best_val_loss:
        best_train_loss = current_train_loss
        best_val_loss = current_val_loss
        best_W = W
        best_b = b
        best_v = v
        convergence = res_step2.success

    stop = time.time()

    print('')
    print('Time required by optimization:', round(stop - start, 1), ' s')
    print('Time required by 1st step:', round(stop1 - start, 1), ' s')
    print('Time required by 2nd step:', round(stop2 - start2, 1), ' s')
    print('Minimal Loss Value on Train: ', res_step2.fun)
    print('Validation Loss: ', current_val_loss)
    print('Iterations: ', res_step2.nit)
    print('Did it converge?:', res_step2.success)
    print('===================')

stop0 = time.time()

print('N')
print(N_best)
print('')
print('sigma')
print(sigma_best)
print('')
print('rho')
print(rho_best)
print('')
print('W')
print(W)
print('')
print('b')
print(b)
print('')
print('v')
print(v)
print('')
print('Train Loss')
print(best_train_loss)
print('Validation Loss')
print(best_val_loss)
print('')
print('Convergence?')
print(convergence)
print('')
print('')
print('Time required by optimization:', round(stop0 - start0, 1), ' s')

# Plot function estimated from data
plotting(W, b, v, sigma_best)
