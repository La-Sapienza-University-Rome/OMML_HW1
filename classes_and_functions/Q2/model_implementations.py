"""
This file contains the concrete models ModelCVX and ModelNumpy
"""


import time

import cvxpy as cvx
from scipy.optimize import minimize
from tqdm import tqdm

from classes_and_functions.Q2.model_interface import *



np.random.seed(1939671)




class ModelCVX(Model):
    """
    Inherit from Model abstract class
    """

    ###########################
    # Magic Methods
    ###########################

    def __init__(self, algorithm, hyper_param_cfg_file):
        super().__init__()
        self._set_hyperparameters(self._load_hyperparameters(hyper_param_cfg_file))
        self.algorithm = algorithm


    ###########################
    # Public Methods
    ###########################

    def eval(self, Xy_test):
        X_test, y_test = Xy_test
        return self.loss(X_test, y_test, test=True).value



    def feedforward(self, X):
        return np.squeeze(np.array(self._feedforward(X).value))



    def fit(self, Xy, Xy_test, trials=1, **kwargs):
        self.X, self.y = Xy
        X_test, y_test = Xy_test
        max_iters = kwargs['max_iters'] if 'max_iters' in kwargs.keys() else 10000
        solver = kwargs['solver'] if 'solver' in kwargs.keys() else None
        best_test_loss = 1e4
        start = time.time()
        for _ in tqdm(range(trials)):
            self._set_state(**kwargs)
            cvx_problem = cvx.Problem(cvx.Minimize(self.loss(X=self.X, y=self.y, test=False)))
            cvx_problem.solve(solver=solver, verbose=False, max_iters=max_iters)
            current_test_loss = self.loss(X_test, y_test, test=True).value
            if current_test_loss < best_test_loss:
                best_test_loss = current_test_loss
                best_train_loss = self.loss(self.X, self.y, test=True).value # train loss without the regularization term
                self._save_state(cvx_problem, train_loss=best_train_loss, test_loss=best_test_loss, restore=False)
        stop = time.time()
        self._save_state(None, restore=True, total_elapsed_time=f'{round(stop-start, 3)}s')
        return self



    def loss(self, X, y, test=False):
        P = y.shape[0]
        pred = self._feedforward(X)
        res = cvx.sum_squares(pred - y) / (2*P)
        if not test:
            res = res + 0.5 * self.RHO * cvx.square(cvx.norm(self.v, 2))
        return res


    ###########################
    # Protected Methods
    ###########################

    def _feedforward(self, X):
        if self.algorithm == 'MLP':
            return self._feedforward_MLP(X)
        else:
            return self._feedforward_RBF(X)



    def _feedforward_MLP(self, X):
        linear_layer = np.dot(X, self.W) + self.b
        activation = self._tanh(linear_layer)
        return cvx.matmul(activation, self.v)



    def _feedforward_RBF(self, X):
        return cvx.matmul(self._rbf(X).T, self.v)


    
    def _set_state(self, **kwargs):
        self.v = cvx.Variable(shape=(self.N,1), value=np.random.normal(size=(self.N,1)), name='v')
        super()._set_state(**kwargs)



    def _save_state(self, state, **kwargs):
        if kwargs['restore']:
            if self.algorithm == 'MLP':
                self.W = self.state['best_W']
                self.b = self.state['best_b']
            else:
                self.c = self.state['best_c']
            self.v = self.state['best_v'].reshape(self.v.shape)
            self.state['total_elapsed_time'] = kwargs['total_elapsed_time']
        else:
            if self.algorithm == 'MLP':
                self.state = {'best_W': self.W.copy(),
                              'best_b': self.b.copy(),
                              'printable_info':{}}
            else:
                self.state = {'best_c': self.c.copy(), 'printable_info':{}}
            self.state['best_v'] = np.array(self.v.value).ravel()
            self.state['printable_info']['Number of neurons N chosen'] = self.N
            self.state['printable_info']['Value of σ chosen'] = self.SIGMA
            self.state['printable_info']['Value of ρ chosen'] = self.RHO
            self.state['printable_info']['Optimization solver chosen'] = state.solver_stats.solver_name
            self.state['printable_info']['Number of function evaluations'] = state.solver_stats.num_iters
            self.state['printable_info']['Time for optimizing the network'] = round(state.solver_stats.solve_time, 6)
            self.state['printable_info']['Training Error'] = round(kwargs['train_loss'], 6)
            self.state['printable_info']['Test Error'] = round(kwargs['test_loss'], 6)

    



class ModelNumpy(Model):
    """
    Inherit from Model abstract class
    """

    ###########################
    # Magic Methods
    ###########################

    def __init__(self, algorithm, hyper_param_cfg_file):
        super().__init__()
        self._set_hyperparameters(self._load_hyperparameters(hyper_param_cfg_file))
        self.algorithm = algorithm



    ###########################
    # Public Methods
    ###########################

    def eval(self, Xy_test):
        X_test, y_test = Xy_test
        return self.loss(X_test, y_test, test=True)



    def feedforward(self, X):
        return np.squeeze(self._feedforward(X))



    def fit(self, Xy, Xy_test, trials=1, **kwargs):
        self.X, self.y = Xy
        X_test, y_test = Xy_test
        best_test_loss = 1e4
        start = time.time()
        for _ in tqdm(range(trials)):
            t0 = time.time()
            self._set_state(**kwargs)
            problem_res = minimize(self._loss, self.v, jac=self._gradient, method='SLSQP')
            self.v = np.expand_dims(problem_res.x, axis=1)
            current_test_loss = self.loss(X_test, y_test, test=True)
            t1 = time.time()
            if current_test_loss < best_test_loss:
                best_test_loss = current_test_loss
                self._save_state(problem_res, test_loss=best_test_loss, solver_time=round(t1-t0, 5), restore=False)
        stop = time.time()
        self._save_state(problem_res, restore=True, total_elapsed_time=f'{round(stop-start, 3)}s')
        return self



    def loss(self, X, y, test=False):
        P = y.shape[0]
        pred = self._feedforward(X)
        res = np.sum((pred - y)**2) / (2*P)
        if not test:
            res = res + 0.5 * self.RHO * np.linalg.norm(self.v)**2
        return res



    ###########################
    # Protected Methods
    ###########################

    def _feedforward(self, X):
        if self.algorithm == 'MLP':
            return self._feedforward_MLP(X)
        else:
            return self._feedforward_RBF(X)



    def _feedforward_MLP(self, X):
        linear_layer = np.dot(X, self.W) + self.b
        activation = self._tanh(linear_layer)
        return np.dot(activation, self.v)



    def _feedforward_RBF(self, X):
        return np.dot(self._rbf(X).T, self.v)


    
    def _gradient(self, x0, funcArgs=[]):
        self.v = np.expand_dims(x0, axis=1)
        P = self.y.shape[0]
        if self.algorithm == 'MLP':
            a = self._tanh(np.dot(self.X, self.W) + self.b)
        else:
            a = self._rbf(self.X).T
        dJdf = (1 / P) * (np.dot(a, self.v) - self.y)
        dv = np.dot(a.T, dJdf) + self.RHO * self.v
        return np.squeeze(dv)

    

    def _loss(self, x0, funcArgs=[], test=False):
        self.v = np.expand_dims(x0, axis=1)
        P = self.y.shape[0]
        pred = self._feedforward(self.X)
        res = (np.sum((pred - self.y)**2)) / (2*P)
        if not test:
            res += 0.5 * self.RHO * np.linalg.norm(self.v)**2
        return res



    def _save_state(self, state, **kwargs):
        if kwargs['restore']:
            if self.algorithm == 'MLP':
                self.W, self.b = self.state['best_W'], self.state['best_b']
            else:
                self.c = self.state['best_c']
            self.v = self.state['best_v'].reshape(self.v.shape)
            self.state['total_elapsed_time'] = kwargs['total_elapsed_time']
        else:
            if self.algorithm == 'MLP':
                self.state = {'best_W': self.W.copy(),
                              'best_b': self.b.copy(),
                              'printable_info':{}}
            else:
                self.state = {'best_c': self.c.copy(), 'printable_info':{}}
            self.state['best_v'] = state.x.copy()
            self.state['printable_info']['Number of neurons N chosen'] = self.N
            self.state['printable_info']['Value of σ chosen'] = self.SIGMA
            self.state['printable_info']['Value of ρ chosen'] = self.RHO
            self.state['printable_info']['Optimization solver chosen'] = 'SLSQP'
            self.state['printable_info']['Number of function evaluations'] = state.nfev
            self.state['printable_info']['Time for optimizing the network'] = kwargs['solver_time']
            self.state['printable_info']['Training Error'] = state.fun
            self.state['printable_info']['Test Error'] = kwargs['test_loss']



    def _set_state(self, **kwargs):
        self.v = np.random.normal(size=(self.N,1))
        super()._set_state(**kwargs)