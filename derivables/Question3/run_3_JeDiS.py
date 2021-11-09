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
from functions import *
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
N_best = 80

# Define the best value obtained for Sigma
sigma_best = 1

# Define the best value obtained for Rho
rho_best = 1e-05

# Initialize current loss for early stopping
curr_loss_step2 = None
prev_loss_step2 = None
Nfeval_step2 = 1
global Nmaxit
Nmaxit = 4000

# Set the number of random trials
trials = 1
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
                max_iter=Nmaxit, tol=1e-6,
                method='BFGS', func=loss_step1)
    stop1 = time.time()
    # Extract the values for v after optimization
    v = res_step1.x

    # Number of iterations for step 1
    niter_step1 = res_step1.nit

    # Number of functions evaluation for step 1
    nfev_step1 = res_step1.nfev

    # Number of gradient evaluation for step 1
    njev_step1 = res_step1.njev

    ##################################################
    ##### step2: non-convex minimization wrt w and b
    ##################################################
    start2 = time.time()
    res_step2 = train_step2(X, y, sigma=sigma_best,
                N=N_best, rho=rho_best,
                W_init=W, b_init=b, v=v,
                max_iter=Nmaxit, tol=1e-6,
                method='BFGS', func=loss_step2)
    stop2 = time.time()

    # Number of iterations for step 2
    niter_step2 = res_step2.nit

    # Number of functions evaluation for step 2
    nfev_step2 = res_step2.nfev

    # Number of gradient evaluation for step 2
    njev_step2 = res_step2.njev
    
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
        best_iter_step1 = niter_step1
        best_iter_step2 = niter_step2
        best_nfev_step1 = nfev_step1
        best_nfev_step2 = nfev_step2
        best_njev_step1 = njev_step1
        best_njev_step2 = njev_step2
        time_step1 = round(stop1 - start, 1)
        time_step2 = round(stop2 - start, 1)

    stop = time.time()
    time_total = round(stop - start, 1)

stop0 = time.time()

print('Number of neurons N chosen:', N_best)
print('Value of σ chosen:', sigma_best)
print('Value of ρ chosen:', rho_best)
print('Optimization solver chosen:', "BFGS")
print('Number of function evaluations:', best_nfev_step1 + best_nfev_step2)
print('Number of gradient evaluations:', best_njev_step1 + best_njev_step2)
print('Time for optimizing the network:', time_total, 's')
print('Training Error:', best_train_loss)
print('Test Error', best_val_loss)


