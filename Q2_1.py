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

X_test = np.array(test[['x1', 'x2']])
y_test = np.expand_dims(np.array(test['y']), axis=1)


def tanh(s:np.ndarray, sigma):
    """
    Compute the tanh as in the Q1_1

    :param s: input variable
    :param sigma: hyperparameter

    :return tanh(x)
    """

    prod = 2*sigma*s
    return (np.exp(prod)-1)/(np.exp(prod)+1)


def feedforward(X:np.ndarray, W:np.ndarray, b:np.ndarray, v:cvx.Variable, sigma) -> cvx.Variable:
    """
    Compute the forward pass of the MLP. Version adapted to cvxpy library.

    :param X: observations
    :param W: first layer weights
    :param b: bias
    :param v: output layer weights. cvxpy variable
    :param sigma: hyperparameter for tanh

    :return predictions
    """
    
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = cvx.matmul(activation, v)
    
    return pred

    
def loss(W:np.ndarray, b:np.ndarray, v:cvx.Variable, funcArgs) -> cvx.Expression:
    """
    Compute the loss of the MLP. Version adapted to cvxpy.

    :param W: first layer weights
    :param b: bias
    :param v: output layer weights. cvxpy variable
    :param funcArgs: list of additional parameters

    :return cvxpy expression. Use res.value to compute the expression and get the result.
    """
    
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

    
def feedforward_eval(x1:float, x2:float, W:np.ndarray, b:np.ndarray, v:cvx.Variable, sigma) -> float:
    """
    Compute the forward pass of the MLP on a tuple (x1, x2). Version adapted to cvxpy library.

    :param x1: first coordinate
    :param x2: second coordinate
    :param W: first layer weights
    :param b: bias
    :param v: output layer weights. cvxpy variable
    :param sigma: hyperparameter for tanh

    :return predicted value f(x1,x2)
    """
    
    X = np.array([x1, x2])
    linear_layer = (np.dot(X, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = cvx.matmul(activation, v)
    
    return pred.value[0,0]
    

# Fix the values of the hyperparameters according to the results of Q1_1
# TODO: make these definitions dynamic
SIGMA = 1
N = 10
RHO = 0.001

    
print('===================')
print('Sigma:', SIGMA)
print('N:', N)
print('Rho:', RHO)


# Define the parameters for the function according to the function's API
funcArgs = [X, y, SIGMA, N, RHO] 

# Set the number of random trials for W and b
trials = 50
best_val_loss = 1000


# Iterate /trials/ times  
for _ in tqdm(range(trials)):

    # Sample W and b from the given intervals
    # TODO: find optimal intervals
    W = np.random.randn(X.shape[1], N)
    b = np.expand_dims(np.random.randn(N), axis=0)

    # Define v as a cvxpy.Variable
    v = cvx.Variable(shape=(N,1), name='v')

    # Define the optimization problem
    cvx_problem = cvx.Problem(cvx.Minimize(loss(W, b, v, funcArgs)))

    # Solve the quadratic convex problem
    cvx_problem.solve(solver=cvx.SCS, verbose=False, eps=1e-6, max_iters=10000)
    
    # If the loss value is less than the current best value, save the parameters and update the best value
    current_train_loss = cvx_problem.value
    current_val_loss = loss(W, b, v, [X_test, y_test, SIGMA, N, RHO]).value
    if current_val_loss < best_val_loss:
        best_train_loss = current_train_loss
        best_val_loss = current_val_loss
        best_W = W
        best_b = b

# Set W and b
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
print(best_train_loss)
print('')
# TODO: compute the loss without the regularization term
print('Loss on Test')
print(best_val_loss)


# Plot 3D 
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
#create the grid
x = np.linspace(-2, 2, 50) 
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
 
