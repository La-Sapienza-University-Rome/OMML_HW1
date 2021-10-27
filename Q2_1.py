import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import cvxpy as cvx
from tqdm import tqdm

df = pd.read_csv('DATA.csv')

train, test = train_test_split(df, test_size=0.25, random_state=1939671)

X = np.array(train[['x1', 'x2']])
y = np.expand_dims(np.array(train['y']), axis=1)

def tanh(s, sigma):
    prod = 2*sigma*s
    return (np.exp(prod)-1)/(np.exp(prod)+1)

def feedforward(X, W, b, v, sigma):
    
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = cvx.matmul(activation, v)
    
    return pred
    
def loss(W, b, v, funcArgs):
    
    X = funcArgs[0]
    y = funcArgs[1]
    sigma = funcArgs[2]
    N = funcArgs[3]
    rho = funcArgs[4]

    P = len(y)
    norm = cvx.norm2(v)
    pred = feedforward(X, W, b, v, sigma)
    res = ((cvx.sum((pred-y)**2))*P**(-1) + rho*norm)*0.5    
    
    return res
    
def feedforward_eval(x1, x2, W, b, v, sigma):
    
    X = np.array([x1, x2])
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = cvx.matmul(activation, v)
    
    return pred.value[0,0]
    
SIGMA = 1
N = 20
RHO = 0

    
print('===================')
print('Sigma:', SIGMA)
print('N:', N)
print('Rho:', RHO)

funcArgs = [X, y, SIGMA, N, RHO] 

trials = 40
best_loss = 1000

for _ in tqdm(range(trials)):
    W = np.random.randn(X.shape[1], N)
    b = np.expand_dims(np.random.randn(N), axis=0)
    v = cvx.Variable(shape=(N,1), name='v')

    cvx_problem = cvx.Problem(cvx.Minimize(loss(W, b, v, funcArgs)))
    cvx_problem.solve(solver=cvx.SCS, verbose=False, eps=1e-6, max_iters=10000)
    
    if cvx_problem.value < best_loss:
        best_loss = cvx_problem.value
        best_W = W
        best_b = b

W = best_W
b = best_b                   

print('N')
print(N)
print('')
print('sigma')
print(SIGMA)
print('')
print('rho')
print(RHO)
print('')
print('W')
print(W)
print('')
print('b')
print(b)
print('')
print('v')
print(v.value.reshape((-1,)))
print('')
print('Loss')
print(best_loss)


# Plot 3D 

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
        z.append(feedforward_eval(x1, x2, W, b, v, SIGMA))
    Z.append(z)
Z = np.array(Z)



ax.plot_surface(X_plot, Y_plot, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('F(x) learnt from MLP')
plt.show()
 
