import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import time

df = pd.read_csv('DATA.csv')

train, test = train_test_split(df, test_size=0.25, random_state=1939671)

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

def train(X, y, sigma, N, rho, W_init, b_init, v_init, max_iter=1000, 
          tol=1e-5, method='CG', func=loss):
    
    x0 = np.concatenate((W_init, b_init, v_init), axis=None)
    funcArgs = [X, y, sigma, N, rho]

    res = minimize(func,
                   x0,
                   args=funcArgs, 
                   method=method, 
                   tol=tol,
                   options={'maxiter':max_iter})    
    
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
    ax.set_title('F(x) learnt from MLP')
    plt.show()

sigma_grid = [1]
N_grid = [10]
rho_grid = np.linspace(1e-5, 1e-3, 5)
iterables = [sigma_grid, N_grid, rho_grid]
min_loss = 10000

for t in itertools.product(*iterables):

    N = t[1]
    W = np.random.randn(X.shape[1], N)
    b = np.random.randn(N)
    v = np.random.randn(N)
    
    print('===================')
    print('Sigma:', t[0])
    print('N:', t[1])
    print('Rho:', t[2])

    start = time.time()
    res = train(X, y, sigma=t[0], 
                N=t[1], rho=t[2], 
                W_init=W, b_init=b, v_init=v,
                max_iter=5000, tol=1e-6, 
                method='CG', func=loss)
    stop = time.time()
    
    res_loss = loss_test(X=X_test, y=y_test, 
                         sigma=t[0], N=t[1], 
                         rho=t[2], 
                         W=res.x[:X.shape[1]*N].reshape((X.shape[1],N)),
                         b=res.x[X.shape[1]*N:X.shape[1]*N+N],
                         v=res.x[X.shape[1]*N+N:])
                   
    print('')   
    print('Time required by optimization:', round(stop-start, 1), ' s')
    print('Validation Loss: ', res_loss)
    print('Minimal Loss Value: ', res.fun)
    print('Num Iterations: ', res.nit)
    print('Did it converge?: ', res.success)
    print('===================')
                   
    if res_loss < min_loss:
        N_best = N
        sigma_best = t[0]
        rho_best = t[2]
        min_loss = res.fun
        best_params = res.x

W=best_params[:X.shape[1]*N_best].reshape((X.shape[1],N_best))
b=best_params[X.shape[1]*N_best:X.shape[1]*N_best+N_best]
v=best_params[X.shape[1]*N_best+N_best:]

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

plotting(W, b, v, sigma_best)