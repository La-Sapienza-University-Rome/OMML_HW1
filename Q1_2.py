import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import time

df = pd.read_csv('DATA.csv')

train, test = train_test_split(df, test_size=0.255, random_state=1939671)

X = np.array(train[['x1', 'x2']])
y = np.array(train['y'])

X_test = np.array(train[['x1', 'x2']])
y_test = np.array(train['y'])

def rbf(X, c, sigma):
    """
    This function is only applied for a single observation
    x belongs to R^2
    c belongs to R^{2, 10}
    return R^10, 186
    """
    minus_matrix = []
    for i in range(len(c.T)):
        minus_matrix.append(X - c.T[i])
    minus_matrix = np.array(minus_matrix)

    return np.exp(-(np.linalg.norm(minus_matrix, axis=2)/sigma)**2)

def feedforward(X, c, v, sigma):
    """
    This function is only applied for a single observation
    x belongs to R^2
    c belongs to R^{2, 10}
    v belongs to R^N
    return float
    """
    
    pred = np.dot(rbf(X, c, sigma).T, v)
    return pred
    
def loss(x0, funcArgs):
    
    X = funcArgs[0]
    y = funcArgs[1]
    sigma = funcArgs[2]
    N = funcArgs[3]
    rho = funcArgs[4]
    
    c = x0[:int(X.shape[1]*N)].reshape((X.shape[1],N))
    v = x0[int(X.shape[1]*N):]

    P = len(y)
    sum_ = np.sum((feedforward(X, c, v, sigma) - y)**2)
    norm = np.linalg.norm(x0)
    res = (sum_*P**(-1) + rho*norm)*0.5 
    
    return res
    
def loss_test(X, y, sigma, N, rho, c, v):

    P = len(y)
    res = np.sum((feedforward(X, c, v, sigma) - y)**2)*0.5*P**(-1)
    
    return res
    
def feedforwardplot(x_i_1, x_i_2, c, v, sigma):
    x_i = np.array([x_i_1, x_i_2])
    pred = np.dot(np.exp(-(np.linalg.norm((x_i - c.T), axis=1)/sigma)**2), v)
    return pred
    
def train(X, y, sigma, N, rho, c_init, 
          v_init, max_iter=1000, tol=1e-5, method='CG', func=loss):
    
    x0 = np.concatenate((c_init, v_init), axis=None)
    funcArgs = [X, y, sigma, N, rho]

    res = minimize(func,
                   x0,
                   args=funcArgs, 
                   method=method, 
                   tol=tol,
                   options={'maxiter':max_iter})    
    
    return res
    
def plotting(c, v, sigma):
    

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
            z.append(feedforwardplot(x1, x2, c, v, sigma))
        Z.append(z)
    Z = np.array(Z)


    ax.plot_surface(X_plot, Y_plot, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('F(x) learnt from RBS')
    plt.show()
    
sigma_grid = [0.01, 1, 2]
N_grid = [2, 5, 10]
rho_grid = [1e-5, 1e-4, 1e-3]
iterables = [sigma_grid, N_grid, rho_grid]
min_loss = 10000

for t in itertools.product(*iterables):

    N = t[1]
    c = np.random.randn(X.shape[1], N)
    v = np.random.randn(N)

    x0 = np.concatenate((c, v), axis=None)
    
    print('===================')
    print('Sigma:', t[0])
    print('N:', t[1])
    print('Rho:', t[2])

    start = time.time()
    res = train(X, y, sigma=t[0], 
                N=t[1], rho=t[2], 
                c_init=c, v_init=v,
                max_iter=1000, tol=1e-6, 
                method='CG', func=loss)
    stop = time.time()
    
    res_loss = loss_test(X=X_test, y=y_test, 
                         sigma=t[0], N=t[1], 
                         rho=t[2], 
                         c=res.x[:X.shape[1]*N].reshape((X.shape[1],N)),
                         v=res.x[X.shape[1]*N:])
                   
    print('')    
    print('Time required by optimization:', round(stop-start, 1), ' s')
    print('Validation Loss: ', res_loss)
    print('Minimal Loss Value', res.fun)
    print('Num Iterations', res.nit)
    print('Did it converge?:', res.success)
    print('===================')
                   
    if res_loss < min_loss:
        N_best = N
        sigma_best = t[0]
        rho_best = t[2]
        min_loss = res.fun
        best_params = res.x

c=best_params[:X.shape[1]*N_best].reshape((X.shape[1],N_best))
v=best_params[X.shape[1]*N_best:]

print('N')
print(N_best)
print('')
print('sigma')
print(sigma_best)
print('')
print('rho')
print(rho_best)
print('')
print('c')
print(c)
print('')
print('v')
print(v)

plotting(c, v, sigma_best)