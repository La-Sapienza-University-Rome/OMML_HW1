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


def backpropagation(x0, funcArgs):

    X = funcArgs[0]
    y = funcArgs[1]
    sigma = funcArgs[2]
    N = funcArgs[3]
    rho = funcArgs[4]
    P = len(y)
    
    c = x0[:int(X.shape[1]*N)].reshape((X.shape[1],N))
    v = x0[int(X.shape[1]*N):]
    
    z_1 = rbf(X, c, sigma).T
    dJdf = (1/P)*(np.dot(z_1, v) - y)

    minus_matrix = []
    for i in range(len(c.T)):
        minus_matrix.append(X - c.T[i])
    minus_matrix = np.array(minus_matrix)

    dW1_1 = np.dot(dJdf.reshape((P, 1)), v.reshape((1,N)))
    dzdc = ((2*z_1)/(sigma**2))*minus_matrix.T

    dv = np.dot(dJdf, z_1) + rho*v
    dc = np.sum(dzdc*dW1_1, axis=1) + rho*c

    return np.concatenate((dc, dv), axis=None)


def loss(x0, funcArgs, test=False):
    
    X = funcArgs[0]
    y = funcArgs[1]
    sigma = funcArgs[2]
    N = funcArgs[3]
    rho = funcArgs[4]
    
    c = x0[:int(X.shape[1]*N)].reshape((X.shape[1],N))
    v = x0[int(X.shape[1]*N):]

    P = len(y)
    pred = feedforward(X, c, v, sigma)
    norm = np.linalg.norm(x0)
    if test:
        res = ((np.sum((pred - y) ** 2)) * P ** (-1)) * 0.5
    else:
        res = ((np.sum((pred - y) ** 2)) * P ** (-1) + rho * norm ** 2) * 0.5
    
    return res


def train(X, y, sigma, N, rho, c_init, 
          v_init, max_iter=1000, tol=1e-5, method='CG', func=loss):
    
    x0 = np.concatenate((c_init, v_init), axis=None)
    funcArgs = [X, y, sigma, N, rho]

    res = minimize(func,
                   x0,
                   args=funcArgs, 
                   method=method, 
                   tol=tol,
                   jac=backpropagation,
                   options={'maxiter':max_iter})    
    
    return res


def plotting(c, v, sigma):

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    #create the grid
    xs = np.linspace(-2, 2, 50)
    ys = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(xs, ys)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z = feedforward(XY, c, v, sigma).reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('F(x) learnt from RBS')
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
    print('N:', t[1])
    print('Rho:', t[2])

    for train_index, test_index in kf5.split(train_df):
        X_ = np.array(train_df.iloc[train_index][['x1', 'x2']])
        X_val = np.array(train_df.iloc[test_index][['x1', 'x2']])
        y_ = np.array(train_df.iloc[train_index]['y'])
        y_val = np.array(train_df.iloc[test_index]['y'])

        c = np.random.normal(size=(X.shape[1], N))
        v = np.random.normal(size=N)

        x0 = np.concatenate((c, v), axis=None)

        start = time.time()
        res = train(X_, y_, sigma=t[0],
                    N=t[1], rho=t[2],
                    c_init=c, v_init=v,
                    max_iter=5000, tol=1e-6,
                    method=t[3], func=loss)
        stop = time.time()

        funcArgs_test = [X_val, y_val, t[0], N, t[2]]

        val_loss += loss(res.x, funcArgs_test, test=True)

    print('')
    print('Time required by optimization:', round(stop-start, 1), ' s')
    print('Validation Loss: ', val_loss)
    print('Minimal Loss Value', res.fun)
    print('Num Iterations', res.nit)
    print('Did it converge?:', res.success)
    print('===================')

    if val_loss < min_loss:
        N_best = N
        sigma_best = t[0]
        rho_best = t[2]
        min_loss = val_loss
        best_params = res.x
        convergence = res.success
        method_best = t[3]

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
print('')
print('Validation Loss')
print(min_loss)
print('')
print('Convergence?')
print(convergence)
print('')
print('Best Method?')
print(method_best)

plotting(c, v, sigma_best)


# Save the best hyperparameters
import json
import pickle

dict = {"c": c,
        "v": v,
        "sigma": sigma_best}

with open('q12_values_for_prediction.pickle', 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('config/q_1_2_cfg.json', 'w') as conf_file:
    json.dump({
        'SIGMA': sigma_best,
        'RHO': rho_best,
        'N': N_best
    }, conf_file)