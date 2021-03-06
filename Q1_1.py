import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold

df = pd.read_csv('DATA.csv')

train_df, test_df = train_test_split(df, test_size=0.255, random_state=1939671)

X = np.array(train_df[['x1', 'x2']])
y = np.array(train_df['y'])

X_test = np.array(test_df[['x1', 'x2']])
y_test = np.array(test_df['y'])


def tanh(s, sigma):
    prod = 2 * sigma * s
    return (np.exp(prod) - 1) / (np.exp(prod) + 1)


def feedforward(X, W, b, v, sigma):
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = np.dot(activation, v)

    return pred

def backpropagation(x0, funcArgs):
    
    X = funcArgs[0]
    y = funcArgs[1]
    sigma = funcArgs[2]
    N = funcArgs[3]
    rho = funcArgs[4]
    P = len(y)
    
    W = x0[:int(X.shape[1] * N)].reshape((X.shape[1], N))
    b = x0[int(X.shape[1] * N):int(X.shape[1] * N + N)]
    v = x0[int(X.shape[1] * N + N):]

    linear_layer = (np.dot(X, W) + b)
    a_2 = tanh(linear_layer, sigma)
    dJdf = (1 / P) * (np.dot(a_2, v) - y)
    dtanh = 1 - tanh(linear_layer, sigma) ** 2

    dW1_1 = np.tensordot(dJdf, np.transpose(v), axes=0)
    dW1_2 = dW1_1 * dtanh

    dv = np.dot(dJdf, a_2) + rho * v
    db = np.sum(dW1_2, axis=0) + rho * b
    dW = np.tensordot(np.transpose(X), dW1_2, axes=1) + rho * W

    return np.concatenate((dW, db, dv), axis=None)

def loss(x0, funcArgs, test=False):
    X = funcArgs[0]
    y = funcArgs[1]
    sigma = funcArgs[2]
    N = funcArgs[3]
    rho = funcArgs[4]

    W = x0[:int(X.shape[1] * N)].reshape((X.shape[1], N))
    b = x0[int(X.shape[1] * N):int(X.shape[1] * N + N)]
    v = x0[int(X.shape[1] * N + N):]

    P = len(y)
    norm = np.linalg.norm(x0)
    pred = feedforward(X, W, b, v, sigma)
    if test:
        res = ((np.sum((pred - y) ** 2)) * P ** (-1)) * 0.5
    else:
        res = ((np.sum((pred - y) ** 2)) * P ** (-1) + rho * norm ** 2) * 0.5

    return res


def train(X, y, sigma, N, rho, W, b, v, max_iter=1000,
          tol=1e-5, method='CG', func=loss):
          
    x0 = np.concatenate((W, b, v), axis=None)
    funcArgs = [X, y, sigma, N, rho]
    
    res = minimize(func,
                   x0,
                   args=funcArgs, 
                   method=method, 
                   tol=tol,
                   jac=backpropagation,
                   options={'maxiter':max_iter})  
    
    return res
    

def plotting(W, b, v, sigma):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    # create the grid

    xs = np.linspace(-2, 2, 50)
    ys = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(xs, ys)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z = feedforward(XY, W, b, v, sigma).reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('F(x) learnt from MLP')
    plt.show()

sigma_grid = [0.5, 1, 1.5]
N_grid = [40, 50, 60, 70, 80, 90]
rho_grid = np.linspace(1e-5, 1e-3, 3)
method_grid = ['CG', 'BFGS', 'L-BFGS-B']
iterables = [sigma_grid, N_grid, rho_grid, method_grid]
min_loss = 10000
k_fold = 5

kf5 = KFold(n_splits=k_fold, shuffle=False)

for t in itertools.product(*iterables):

    val_loss = 0
    N = t[1]

    print('===================')
    print('Sigma:', t[0])
    print('N:', N)
    print('Rho:', t[2])

    for train_index, test_index in kf5.split(train_df):

        X_ = np.array(train_df.iloc[train_index][['x1', 'x2']])
        X_val = np.array(train_df.iloc[test_index][['x1', 'x2']])
        y_ = np.array(train_df.iloc[train_index]['y'])
        y_val = np.array(train_df.iloc[test_index]['y'])

        W = np.random.normal(size=(X.shape[1], N))
        b = np.random.normal(size=N)
        v = np.random.normal(size=N)

        x0 = np.concatenate((W, b, v), axis=None)

        start = time.time()
        res = train(X_, y_, sigma=t[0],
                    N=N, rho=t[2],
                    W=W, b=b, v=v,
                    max_iter=5000, tol=1e-6,
                    method=t[3], func=loss)
        stop = time.time()

        funcArgs_test = [X_val, y_val, t[0], N, t[2]]

        val_loss += loss(res.x, funcArgs_test, test=True)

    print('')
    print('Time required by optimization:', round(stop - start, 1), ' s')
    print('Average Validation Loss: ', val_loss)
    print('Minimal Loss Value on Last Train Dataset: ', res.fun)
    print('Iterations: ', res.nit)
    print('Did it converge?:', res.success)
    print('===================')

    val_loss = val_loss/k_fold

    if val_loss < min_loss:
        N_best = N
        sigma_best = t[0]
        rho_best = t[2]
        min_loss = val_loss
        best_params = res.x
        convergence = res.success
        method_best = t[3]

W = best_params[:int(X.shape[1] * N_best)].reshape((X.shape[1], N_best))
b = best_params[int(X.shape[1] * N_best):int(X.shape[1] * N_best + N_best)]
v = best_params[int(X.shape[1] * N_best + N_best):]

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
print('Loss')
print(min_loss)
print('')
print('Convergence?')
print(convergence)
print('')
print('Best Method?')
print(method_best)

plotting(W, b, v, sigma_best)

# Save the best hyperparameters
import json
import pickle

dict = {"W": W,
        "b": b,
        "v": v,
        "sigma": sigma_best}

with open('q11_values_for_prediction.pickle', 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('config/q_1_1_cfg.json', 'w') as conf_file:
    json.dump({
        'SIGMA': sigma_best,
        'RHO': rho_best,
        'N': N_best
    }, conf_file)