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

def tanh(s, sigma):
    prod = 2*sigma*s
    return (np.exp(prod)-1)/(np.exp(prod)+1)

def feedforward(X, W, b, v, sigma):
    
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = np.dot(linear_layer, v)
    
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
    norm = np.linalg.norm(np.concatenate((b, W, v), axis=None))
    pred = feedforward(X, W, b, v, sigma)
    res = ((np.sum((pred-y)**2))*P**(-1) + rho*norm)*0.5    
    
    return res
    
def feedforward_eval(x1, x2, W, b, v, sigma):
    
    X = np.array([x1, x2])
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = np.dot(linear_layer, v)
    
    return pred
    
sigma_grid = np.linspace(0, 10, 5)
N_grid = [1, 5, 10, 15, 20]
rho_grid = np.linspace(1e-5, 1e-3, 5)

iterables = [sigma_grid, N_grid, rho_grid]
min_loss = 100

for t in itertools.product(*iterables):

    N = t[1]
    W = 1e-4 * np.random.randn(X.shape[1], N)
    b = np.zeros(N)
    v = 1e-4 * np.random.randn(N)

    x0 = np.concatenate((W, b, v), axis=None)
    
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
                   options={'maxiter':1000})
                   
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
print('W')
print(best_params[:X.shape[1]*N_best].reshape((X.shape[1],N_best)))
print('')
print('b')
print(best_params[X.shape[1]*N_best:X.shape[1]*N_best+N_best])
print('')
print('v')
print(best_params[X.shape[1]*N_best+N_best:])

# Plot 3D 

W=best_params[:X.shape[1]*N_best].reshape((X.shape[1],N_best))
b=best_params[X.shape[1]*N_best:X.shape[1]*N_best+N_best]
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
        z.append(feedforward_eval(x1, x2, W, b, v, sigma))
    Z.append(z)
Z = np.array(Z)


ax.plot_surface(X_plot, Y_plot, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('F(x) learnt from MLP')
plt.show()
