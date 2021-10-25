import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import itertools

df = pd.read_csv('DATA.csv')

train, test = train_test_split(df, test_size=0.25, random_state=1939671)

X = np.array(train[['x1', 'x2']])
y = np.array(train['y'])

N = 10
W = 1e-4 * np.random.randn(X.shape[1], N)
b = np.zeros(N)
v = 1e-4 * np.random.randn(N)

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
    rho = funcArgs[3]
    N = funcArgs[4]

    W = x0[:X.shape[1]*N].reshape((X.shape[1],N))
    b = x0[X.shape[1]*N:X.shape[1]*N+N]
    v = x0[X.shape[1]*N+N:]

    P = len(y)
    norm = np.linalg.norm(np.concatenate((b, W, v), axis=None))
    pred = feedforward(X, W, b, v, sigma)
    res = ((np.sum((pred-y)**2))*P**(-1) + rho*norm)*0.5    
    
    return res
    
def plotting(function, title='Plotting of the function'): #if you do not provide a title, 'Plotting...' will be used
    #create the object
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #create the grid
    x = np.linspace(-3, 3, 50) #create 50 points between [-5,5] evenly spaced  
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y) #create the grid for the plot

    Z = function(X, Y) #evaluate the function (note that X,Y,Z are matrix)


    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()
    
x0 = np.concatenate((W, b, v), axis=None)

sigma_grid = np.linspace(0, 2, 5)
N_grid = np.linspace(5, 20, 5)
rho_grid = np.linspace(1e-5, 1e-3, 5)

iterables = [sigma_grid, N_grid, rho_grid]
min_loss = 100

for t in itertools.product(*iterables):
    
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
    print('')
    print('W')
    print(res.x[:X.shape[1]*N].reshape((X.shape[1],N)))
    print('')
    print('b')
    print(res.x[X.shape[1]*N:X.shape[1]*N+N])
    print('')
    print('v')
    print(res.x[X.shape[1]*N+N:])
                   
    if res.fun < min_loss:
        N_best = N
        sigma_best = t[0]
        rho_best = t[2]
        min_loss = res.fun
        best_params = res.x

print('W')
print(best_params[:X.shape[1]*N_best].reshape((X.shape[1],N_best)))
print('')
print('b')
print(best_params[X.shape[1]*N_best:X.shape[1]*N_best+N_best])
print('')
print('v')
print(best_params[X.shape[1]*N_best+N_best:])

# TODO: Include the graph part
