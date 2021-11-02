import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cvxpy as cvx
from tqdm import tqdm
import json


np.random.seed(1939671)

df = pd.read_csv('DATA.csv')

train, test = train_test_split(df, test_size=0.255, random_state=1939671)

X = np.array(train[['x1', 'x2']])
y = np.expand_dims(np.array(train['y']), axis=1)

X_test = np.array(test[['x1', 'x2']])
y_test = np.expand_dims(np.array(test['y']), axis=1)



def rbf(X:np.ndarray, c:np.ndarray, sigma:float) -> np.ndarray:
    """
    Compute the RBF kernel given the observations' matrix X and the centers c.

    :param X: observations' matrix. R^{n_obs, n_features}
    :params c: centers. R^{n_units, n_features}
    :param sigma: hyperparameter

    :return result of the kernel. Matrix of shape (n_units, X.shape[0])
    """
    N = c.shape[0]
    rows_idxs = np.array([(i,) for i in range(N)])
    minus_matrix = X - c[rows_idxs]
    return np.exp(-(np.linalg.norm(minus_matrix, axis=2) / sigma)**2)


def feedforward(X:np.ndarray, c:np.ndarray, v:cvx.Variable, sigma:float) -> cvx.Variable:
    """
    Compute the forward pass of the MLP with RBF kernel.

    :param X: observations' matrix in R^{n_obs, n_features}
    :param c: centers' matrix in R^{n_units, n_features}
    :param v: output layer weights
    :param sigma: hyperparameter 

    :return
    """
    
    assert sigma>0, 'Sigma must be positive.'
    assert X.shape[1] == c.shape[1], 'The shapes of X and c don\'t match.'

    return cvx.matmul(v.T, rbf(X, c, sigma))


def loss(X:np.ndarray, y:np.ndarray, c:np.ndarray, v:cvx.Variable, sigma:float, rho:float, test=False) -> cvx.Expression:
    """
    Compute the loss of the RBF MLP. Version adapted to cvxpy.

    :param X: observations' matrix
    :param y: target variable
    :param c: centers' matrix
    :param v: output layer weights. cvxpy variable
    :param sigma: hyperparameter
    :param rho: regularization hyperparameter
    :param test: compute the loss without the regularization term. Boolean

    :return cvxpy expression. Use res.value to compute the expression and get the result.
    """
    
    P = y.shape[0]
    pred = feedforward(X, c, v, sigma)
    res = cvx.sum((pred - y)**2) / (2*P)

    if not test:
        res = res + 0.5 * rho * cvx.norm2(v)**2

    return res



# Define the hyperparameters according to Q1_2
with open('config/q_1_2_cfg.json', 'r') as conf_file:
    hyperparameter_cfg = json.load(conf_file)
SIGMA = hyperparameter_cfg['SIGMA']
N = hyperparameter_cfg['N']
RHO = hyperparameter_cfg['RHO']


print('===================')
print('Sigma:', SIGMA)
print('N:', N)
print('Rho:', RHO)


# Set the number of random trials for the centers' random selection
trials = 20
best_val_loss = 1000

P = X.shape[0]


# Iterate /trials/ times  
for _ in tqdm(range(trials)):

    # Sample the centers c among the P observations
    c_idxs = np.random.choice(np.arange(P), size=N, replace=False)
    c = X[c_idxs].copy()

    # Define v as a cvxpy.Variable
    v = cvx.Variable(shape=(N,1), name='v')

    # Define the optimization problem
    cvx_problem = cvx.Problem(cvx.Minimize(loss(X, y, c, v, SIGMA, RHO)))

    # Solve the quadratic convex problem
    cvx_problem.solve(solver=cvx.ECOS, verbose=False, max_iters=10000)
    
    # If the loss value is less than the current best value, save the parameters and update the best value
    current_train_loss = cvx_problem.value
    current_val_loss = loss(X_test, y_test, c, v, SIGMA, RHO, test=True).value
    if current_val_loss < best_val_loss:
        best_train_loss = current_train_loss
        best_val_loss = current_val_loss
        best_c_idxs = c_idxs
        best_v = v


# Set c and v
c = X[best_c_idxs,]
v = best_v 


print('N')
print(N)
print('')
print('sigma')
print(SIGMA)
print('')
print('rho')
print(RHO)
print('')
print('v')
print(v.value.reshape((-1,)))
print('')
print('Loss')
print(best_train_loss)
print('')
print('Loss on Test')
print(best_val_loss)


# Plot 3D 
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
#create the grid
x = np.linspace(-3, 3, 50) 
y = np.linspace(-2, 2, 50)
X_plot, Y_plot = np.meshgrid(x, y) 

Z = []
for x1 in np.linspace(-3, 3, 50):
    z  = []
    for x2 in np.linspace(-3, 3, 50):
        z.append(feedforward(np.array([x1, x2]).reshape(1,-1), c, v, SIGMA).value[0,0])
    Z.append(z)
Z = np.array(Z)



ax.plot_surface(X_plot, Y_plot, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('F(x) learnt from MLP')
plt.show()

