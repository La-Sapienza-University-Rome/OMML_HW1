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
    prod = 2*sigma*s
    return (np.exp(prod)-1)/(np.exp(prod)+1)

def feedforward(X, W, b, v, sigma):
    
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = np.dot(activation, v)
    
    return pred
    
def backpropagation(X, W, b, v, sigma, rho):
    
    grads = {}
    
    P = len(X)
    linear_layer = (np.dot(X, W) + b)
    a_2 = tanh(linear_layer, sigma)
    dJdf = (1/P)*(np.dot(a_2, v) - y)
    dtanh = 1 - tanh(linear_layer, sigma)**2
    
    dW1_1 = np.tensordot(dJdf, np.transpose(v), axes=0)
    dW1_2 = dW1_1*dtanh
    
    grads['v'] = np.dot(dJdf, a_2) + rho*v
    grads['b'] = np.sum(dW1_2, axis=0) + rho*b
    grads['W'] = np.tensordot(np.transpose(X), dW1_2, axes=1) + rho*W
    
    return grads
    
def loss(x0, funcArgs):
    
    X = funcArgs[0]
    y = funcArgs[1]
    sigma = funcArgs[2]
    N = funcArgs[3]
    rho = funcArgs[4]
    

    W = x0[:int(X.shape[1]*N)].reshape((X.shape[1],N))
    b = x0[int(X.shape[1]*N):int(X.shape[1]*N+N)]
    v = x0[int(X.shape[1]*N+N):]

    P = len(y)
    norm = np.linalg.norm(x0)
    pred = feedforward(X, W, b, v, sigma)
    res = ((np.sum((pred-y)**2))*P**(-1) + rho*norm)*0.5    
    
    return res
    
def loss_test(X, y, sigma, N, rho, W, b, v):
    
    P = len(y)
    pred = feedforward(X, W, b, v, sigma)
    res = ((np.sum((pred-y)**2))*P**(-1))*0.5    
    
    return res
    
def feedforwardplot(x1, x2, W, b, v, sigma):
    
    X = np.array([x1, x2])
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = np.dot(activation, v)
    
    return pred

def train(X, y, sigma, N, rho, W, b, v, max_iter=1000, 
          tol=1e-5, learning_rate=1e-3):
    
    funcArgs = [X, y, sigma, N, rho]
    loss_hist = 10000
    #norm_hist = 100000
    tol_bool = False
    for it in range(max_iter):
        x0 = np.concatenate((W, b, v), axis=None)
        
        # Compute loss and gradients using the current minibatch
        loss_train = loss(x0, funcArgs)
        grads = backpropagation(X=X, W=W, b=b, v=v, sigma=sigma, rho=rho)
        
        W += -learning_rate * grads['W']
        b += -learning_rate * grads['b']
        v += -learning_rate * grads['v'] 
        
        # Tolerance stop condition
        if abs(loss_train - loss_hist) < tol:    
            res = {'W': W, 'b': b, 'v': v, 'loss_train': loss_train, 'tol': True, 'iter': it}
            
            return res
        loss_hist = loss_train
        
    res = {'W': W, 'b': b, 'v': v, 'loss_train': loss_train, 'tol': tol_bool, 'iter': max_iter}
    
    return res
    
def plotting(W, b, v, sigma):
    

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    #create the grid
    x = np.linspace(-3, 3, 50) 
    y = np.linspace(-2, 2, 50)
    X_plot, Y_plot = np.meshgrid(x, y) 

    Z = []
    for x1 in x:
        z  = []
        for x2 in y:
            z.append(feedforwardplot(x1, x2, W, b, v, sigma))
        Z.append(z)
    Z = np.array(Z)


    ax.plot_surface(X_plot, Y_plot, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('F(x) learnt from MLP BackPropagation')
    plt.show()

sigma_grid = [1]
N_grid = [4]
rho_grid = [0]
#rho_grid = np.linspace(1e-5, 1e-3, 2)
iterables = [sigma_grid, N_grid, rho_grid]
min_loss = 10000
learning_rate=1e-2

W = np.random.randn(X.shape[1], N_grid[0])
b = np.random.randn(N_grid[0])
v = np.random.randn(N_grid[0])

print(W)
print(b)
print(v)

for t in itertools.product(*iterables):

    N = t[1]
    
    print('===================')
    print('Sigma:', t[0])
    print('N:', t[1])
    print('Rho:', t[2])

    start = time.time()
    res = train(X, y, sigma=t[0], 
                N=t[1], rho=t[2], 
                W=W, b=b, v=v,
                max_iter=5000, tol=1e-8,
                learning_rate=learning_rate)
    stop = time.time()
    
    res_loss = loss_test(X=X_test, y=y_test, 
                         sigma=t[0], N=t[1], 
                         rho=t[2], 
                         W=res['W'],
                         b=res['b'],
                         v=res['v'])
                   
    print('')   
    print('Time required by optimization:', round(stop-start, 1), ' s')
    print('Validation Loss: ', res_loss)
    print('Minimal Loss Value on Train: ', res['loss_train'])
    print('Iterations: ', res['iter'])
    print('===================')
                   
    if res_loss < min_loss:
        N_best = N
        sigma_best = t[0]
        rho_best = t[2]
        min_loss = res_loss
        best_params = res

W=best_params['W']
b=best_params['b']
v=best_params['v']

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
print('Tolerance Reached?')
print(res['tol'])


plotting(W, b, v, sigma_best)