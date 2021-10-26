import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt

df = pd.read_csv('DATA.csv')

train, test = train_test_split(df, test_size=0.25, random_state=1939671)

X = np.array(train[['x1', 'x2']])
y = np.array(train['y'])

def rbf(x_i, c, sigma):
    """
    This function is only applied for a single observation
    x belongs to R^2
    c belongs to R^{2, 10}
    return R^10, 186
    """
    return np.exp(-(np.linalg.norm((x_i - c.T), axis=1)/sigma)**2)

def feedforward(x_i, c, v, sigma):
    """
    This function is only applied for a single observation
    x belongs to R^2
    c belongs to R^{2, 10}
    v belongs to R^N
    return float
    """
    
    pred = np.dot(rbf(x_i, c, sigma), v)
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
    sum_ = 0
    for i in range(P):
        sum_ += (feedforward(X[i], c, v, sigma) - y[i])**2
    norm = np.linalg.norm(x0)
    res = (sum_*P**(-1) + rho*norm)*0.5 
    
    return res
    
def feedforwardeval(x_i_1, x_i_2, c, v, sigma):
    """
    This function is only applied for a single observation
    x belongs to R^2
    c belongs to R^{2, 10}
    v belongs to R^N
    return float
    """
    x_i = np.array([x_i_1, x_i_2])
    pred = np.dot(rbf(x_i, c, sigma), v)
    return pred
        
sigma_grid = [1]
N_grid = [20]
rho_grid = [0]

iterables = [sigma_grid, N_grid, rho_grid]
min_loss = 10000

Nfeval = 1

def callbackF(Xi):
    global Nfeval
    if Nfeval % 10 == 0: 
        print(Nfeval, loss(Xi, funcArgs))
    Nfeval += 1

for t in itertools.product(*iterables):

    N = t[1]
    c = np.random.randn(X.shape[1], N)
    v = np.random.randn(N)

    x0 = np.concatenate((c, v), axis=None)
    
    print('===================')
    print('Sigma:', t[0])
    print('N:', t[1])
    print('Rho:', t[2])

    funcArgs = [X, y, t[0], t[1], t[2]] 

    res = minimize(loss,
                   x0,
                   args=funcArgs, 
                   method='CG', 
                   tol=1e-6,
                   callback=callbackF,
                   options={'maxiter':100})
                   
    print('')    
    print('Minimal Loss Value', res.fun)
    print('Num Iterations', res.nit)
    print('Did it converge?:', res.success)
    print('===================')
                   
    if res.fun < min_loss:
        N_best = N
        sigma_best = t[0]
        rho_best = t[2]
        min_loss = res.fun
        best_params = res.x

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
print(best_params[:X.shape[1]*N_best].reshape((X.shape[1],N_best)))
print('')
print('v')
print(best_params[X.shape[1]*N_best+N_best:])


c=best_params[:X.shape[1]*N_best].reshape((X.shape[1],N_best))
v=best_params[X.shape[1]*N_best+N_best:]
sigma=sigma_best

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
#create the grid
x = np.linspace(-3, 3, 50) 
y = np.linspace(-3, 3, 50)
X_plot, Y_plot = np.meshgrid(x, y) 

Z = []
for x1 in np.linspace(-3, 3, 50):
    z  = []
    for x2 in np.linspace(-3, 3, 50):
        z.append(feedforwardeval(x1, x2, c, v, sigma))
    Z.append(z)
Z = np.array(Z)


ax.plot_surface(X_plot, Y_plot, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('F(x) learnt from MLP')
plt.show()