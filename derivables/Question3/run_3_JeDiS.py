import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
change = parentdir + r'\functions'
sys.path.append(change)
from functions_q3 import *
import json

# Define random seed for numpy operations
seed = 1939671
np.random.seed(seed)

# Load data set
df = pd.read_csv('../../DATA.csv')

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

    # Define the early stopping criteria
    if abs(losses[-1] - losses[-2]) < thres:
        break

stop0 = time.time()

print('Number of neurons N chosen:', N_best)
print('Value of σ chosen:', sigma_best)
print('Value of ρ chosen:', rho_best)
print('Optimization solver chosen:', "block1: SLSQP, block2: L-BFGS-B")
print('Number of function evaluations:', best_nfev_block1 + best_nfev_block2)
print('Number of gradient evaluations:', best_njev_block1 + best_njev_block2)
print('Time for optimizing the network:', round(stop0 - start0, 1), 's')
print('Training Error:', best_train_loss)
print('Test Error', best_val_loss)