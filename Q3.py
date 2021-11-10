import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import optimize
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import json
import pickle
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

def backpropagation_block1(x0, funcArgs):
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

    return np.concatenate((dv), axis=None)

def backpropagation_block2(x0, funcArgs):
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

def loss_block1(x0, funcArgs, test=False):
    """
    Compute the loss of the MLP for the first block (with respect to v).

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



def loss_block2(x0, funcArgs, test=False):
    """
    Compute the loss of the MLP for the second block (with respect to w and b).

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

def train_block1(X, y, sigma, N, rho, W_init, b_init, v_init, max_iter=1000,
          tol=1e-5, method='CG', func=loss_block1):
    """
    Train the MLP for the given hyperparameters for the block 1
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
                   jac=backpropagation_block1,
                   options={'maxiter': max_iter})

    return res

def train_block2(X, y, sigma, N, rho, W_init, b_init, v, max_iter=1000,
          tol=1e-5, method='CG', func=loss_block2):
    """
    Train the MLP for the given hyperparameters for the block 1
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
                   jac=backpropagation_block2,
                   options={'maxiter': max_iter})

    return res

def plotting(title, W, b, v, sigma):
    """
    Plot the function in (-3,3)x(-2,2).
    :param title
    """
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    xs = np.linspace(-2, 2, 50)
    ys = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(xs, ys)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z = feedforward(XY, W, b, v, sigma).reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()

# Define the best value obtained for N
N_best = 70

# Define the best value obtained for Sigma
sigma_best = 1

# Define the best value obtained for Rho
rho_best = 1e-05

# Initialize current loss for early stopping

# Set the number of random trials for W and b
max_trials = 30
best_val_loss = 1000

# Get random initialization for W
W_init = np.random.randn(X.shape[1], N_best)

# Get random initialization for b
b_init = np.random.randn(N_best)

# Get random initialization for v
v_init = np.random.randn(N_best)

# Threshold for early stopping
thres = 1e-4

# Initialize the previous validation loss
losses = [1000]

# Initialize counters
niter_block1 = 0
nfev_block1 = 0
njev_block1 = 0
niter_block2 = 0
nfev_block2 = 0
njev_block2 = 0
time_block1 = 0
time_block2 = 0
time_total = 0

start0 = time.time()
# Iterate /trials/ times
for i in tqdm(range(max_trials)):

    ########################################
    ##### block1: convex minimization wrt v
    ########################################

    # Set the tolerance to use in the minimizations (change it in each iteration exponentially)
    tol = 1e-2 * (1 + 2)**(-i)
    print("Tolerance:",tol)
    start = time.time()
    res_block1 = train_block1(X, y, sigma=sigma_best,
                N=N_best, rho=rho_best,
                W_init=W_init, b_init=b_init, v_init=v_init,
                max_iter=4000, tol=tol,
                method='SLSQP', func=loss_block1)
    stop1 = time.time()
    # Extract the values for v after optimization
    v = res_block1.x

    # Number of iterations for block 1
    niter_block1 += res_block1.nit

    # Number of functions evaluation for block 1
    nfev_block1 += res_block1.nfev

    # Number of gradient evaluation for block 1
    njev_block1 += res_block1.njev

    ##################################################
    ##### block2: non-convex minimization wrt w and b
    ##################################################
    start2 = time.time()
    res_block2 = train_block2(X, y, sigma=sigma_best,
                N=N_best, rho=rho_best,
                W_init=W_init, b_init=b_init, v=v,
                max_iter=4000, tol=tol,
                method='L-BFGS-B', func=loss_block2)
    stop2 = time.time()

    # Number of iterations for block 2
    niter_block2 += res_block2.nit

    # Number of functions evaluation for block 2
    nfev_block2 += res_block2.nfev

    # Number of gradient evaluation for block 2
    njev_block2 += res_block2.njev

    # Get the loss for validation set
    funcArgs_test = [X_test, y_test, sigma_best, N_best, rho_best, v]

    current_val_loss = loss_block2(res_block2.x, funcArgs_test, test=True)
    losses.append(current_val_loss)

    # Extract the loss for the train set
    current_train_loss = res_block2.fun

    # Extract the values for W and b after optimization
    best_params = res_block2.x
    W = best_params[:X.shape[1] * N_best].reshape((X.shape[1], N_best))
    b = best_params[X.shape[1] * N_best:X.shape[1] * N_best + N_best]

    # Retain the best values
    if current_val_loss < best_val_loss:
        best_train_loss = current_train_loss
        best_val_loss = current_val_loss
        best_W = W
        best_b = b
        best_v = v
        convergence = res_block2.success
        best_iter_block1 = niter_block1
        best_iter_block2 = niter_block2
        best_nfev_block1 = nfev_block1
        best_nfev_block2 = nfev_block2
        best_njev_block1 = njev_block1
        best_njev_block2 = njev_block2
        time_block1 += round(stop1 - start, 1)
        time_block2 += round(stop2 - start, 1)

    stop = time.time()
    # time_total += round(stop - start, 1)
    print("loss actual:",losses[-1])
    print("loss anterior:",losses[-2])
    print("thres:",thres)

    # Define the early stopping criteria
    if abs(losses[-1] - losses[-2]) < thres:
        break

    print('')
    print('Time required by optimization:', round(stop - start, 1), ' s')
    print('Time required by 1st block:', round(stop1 - start, 1), ' s')
    print('Time required by 2nd block:', round(stop2 - start2, 1), ' s')
    print('Minimal Loss Value on Train: ', res_block2.fun)
    print('Validation Loss: ', current_val_loss)
    print('Iterations: ', res_block2.nit)
    print('Did it converge?:', res_block2.success)
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
print(best_W)
print('')
print('b')
print(best_b)
print('')
print('v')
print(best_v)
print('')
print('Iterations block 1')
print(best_iter_block1)
print('')
print('Iterations block 2')
print(best_iter_block2)
print('')
print('Outer Iterations')
print(best_iter_block1 + best_iter_block2)
print('')
print('nfev block 1')
print(best_nfev_block1)
print('')
print('nfev block 2')
print(best_nfev_block2)
print('')
print('nfev total')
print(best_nfev_block1 + best_nfev_block2)
print('')
print('njev block 1')
print(best_njev_block1)
print('')
print('njev block 2')
print(best_njev_block2)
print('')
print('njev total')
print(best_njev_block1 + best_njev_block2)
print('')
print('Time required by block 1:', time_block1, ' s')
print('')
print('Time required by block 2:', time_block2, ' s')
print('')
print('Time required by whole optimization:', time_block1 + time_block2, ' s')
print('')
print('Train Loss')
print(best_train_loss)
print('Validation Loss')
print(best_val_loss)
print('Convergence?')
print(convergence)

# Plot function estimated from data
plotting("F(x) learnt from MLP", best_W, best_b, best_v, sigma_best)

# Save values for future predictions
dict = {"W": best_W,
        "b": best_b,
        "v": best_v,
        "sigma": sigma_best }

with open('derivables/Question3/q3_values_for_prediction.pickle', 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



