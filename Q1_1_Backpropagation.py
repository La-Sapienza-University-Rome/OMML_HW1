import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import time

np.random.seed(42)

df = pd.read_csv('DATA.csv')

train, test = train_test_split(df, test_size=0.255, random_state=1939671)

X = np.array(train[['x1', 'x2']])
y = np.array(train['y'])

X_test = np.array(train[['x1', 'x2']])
y_test = np.array(train['y'])


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

# def backpropagationold(X, W, b, v, sigma, rho):
    # grads = {}

    # P = len(X)
    # linear_layer = (np.dot(X, W) + b)
    # a_2 = tanh(linear_layer, sigma)
    # dJdf = (1 / P) * (np.dot(a_2, v) - y)
    # dtanh = 1 - tanh(linear_layer, sigma) ** 2

    # dW1_1 = np.tensordot(dJdf, np.transpose(v), axes=0)
    # dW1_2 = dW1_1 * dtanh

    # grads['v'] = np.dot(dJdf, a_2) + rho * v
    # grads['b'] = np.sum(dW1_2, axis=0) + rho * b
    # grads['W'] = np.tensordot(np.transpose(X), dW1_2, axes=1) + rho * W

    # return grads


def loss(x0, funcArgs):
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
    res = ((np.sum((pred - y) ** 2)) * P ** (-1) + rho * norm**2) * 0.5

    return res


def loss_test(X, y, sigma, N, rho, W, b, v):
    P = len(y)
    pred = feedforward(X, W, b, v, sigma)
    res = ((np.sum((pred - y) ** 2)) * P ** (-1)) * 0.5

    return res


def feedforwardplot(x1, x2, W, b, v, sigma):
    X = np.array([x1, x2])
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = np.dot(activation, v)

    return pred


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
    ax.set_title('F(x) learnt from MLP BackPropagation')
    plt.show()


sigma_grid = [0.5, 1, 2, 5]
N_grid = [2, 5, 10, 20, 40]
rho_grid = np.linspace(1e-5, 1e-3, 3)
iterables = [sigma_grid, N_grid, rho_grid]
min_loss = 10000

for t in itertools.product(*iterables):

    N = t[1]
    W = np.random.randn(X.shape[1], N)
    b = np.random.randn(N)
    v = np.random.randn(N)
    
    
    x0 = np.concatenate((W, b, v), axis=None)

    print('===================')
    print('Sigma:', t[0])
    print('N:', N)
    print('Rho:', t[2])

    start = time.time()
    res = train(X, y, sigma=t[0], 
                N=N, rho=t[2], 
                W=W, b=b, v=v,
                max_iter=5000, tol=1e-6, 
                method='CG', func=loss)
    stop = time.time()

    res_loss = loss_test(X=X_test, y=y_test,
                         sigma=t[0], N=N,
                         rho=t[1],
                         W=res.x[:int(X.shape[1] * N)].reshape((X.shape[1], N)),
                         b=res.x[int(X.shape[1] * N):int(X.shape[1] * N + N)],
                         v=res.x[int(X.shape[1] * N + N):])

    print('')
    print('Time required by optimization:', round(stop - start, 1), ' s')
    print('Validation Loss: ', res_loss)
    print('Minimal Loss Value on Train: ', res.fun)
    print('Iterations: ', res.nit)
    print('Did it converge?:', res.success)
    print('===================')

    if res_loss < min_loss:
        N_best = N
        sigma_best = t[0]
        rho_best = t[2]
        min_loss = res_loss
        best_params = res
        convergence = res.success

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

plotting(W, b, v, sigma_best)

# Save the best hyperparameters
import json

with open('config/q_1_1_cfg.json', 'w') as conf_file:
    json.dump({
        'SIGMA': sigma_best,
        'RHO': rho_best,
        'N': N_best
    }, conf_file)