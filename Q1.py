import numpy as np
import pandas as pd
from scipy.optimize import minimize

df = pd.read_csv('DATA.csv')

X = np.array(df[['x1', 'x2']])
y = np.array(df['y'])

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
    
x0 = np.concatenate((W, b, v), axis=None)
funcArgs = [X, y, 1, 1e-4, 10] 

res = minimize(loss,
               x0,
               args=funcArgs, 
               method='CG', 
               tol=1e-6,
               options={'maxiter':1000})
               
               
print(res.fun)
print('Did it converge?:', res.success)

