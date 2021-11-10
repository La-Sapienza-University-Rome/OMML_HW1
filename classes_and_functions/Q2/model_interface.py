"""
This file contains only the abstract class which provides the interface and the implementation of common methods
[__str__(), _tanh(..), _rbf(..), _set_state(..), plot(..)] to the two concrete subclasses ModelCVX (relying on CVX library) 
and ModelNumpy (relying on Numpy). These sublcasses are located in ./model_implementations.py.

The hierarchy

            Model
            /    /
    ModelCVX    ModelNumpy

follows a State design pattern.
"""


import abc
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
from sklearn.cluster import KMeans


np.random.seed(1939671)




class Model(abc.ABC):
    """
    Abstract class that provides the common interface for subclasses using different libraries.
    Currently, it is the parent class of two classes: ModelCVX, which implements the algorithms MLP and
    RBF by means of the library CVXPY, ModelNumpy, which implements them in Numpy and Scipy.

    The public API is [see the description of each function for details]:
    - eval(tuple(X_test,y_test))
    - feedforward(X)
    - fit(tuple(X,y), tuple(X_test,y_test), trials=1, **kwargs)
    - loss(X, y, test=False)
    - plot()
    - __str__() -> returns the string representation of the state of the object
    """

    ###########################
    # Magic Methods
    ###########################

    def __init__(self):
        self.state = {}



    def __str__(self):
        """
        Return a nice string representation of the object's state
        """
        str_repr = ''
        for key, value in self.state.items():
            str_repr += '\n## {key}: {value}\n'.format(key=key, value=value)
        return str_repr



    ###########################
    # Public Methods
    ###########################

    @abc.abstractmethod
    def feedforward(self, X):
        """
        Return the feedforward pass according to the selected algorithm.
        :param X: observations[n_samples, n_features]
        :return f(X)
        """
        pass


    
    @abc.abstractmethod
    def fit(self, Xy, Xy_test, trials=1, **kwargs):
        """
        Run the model /trials/ times on the training data (X,y). Select the best configuration 
        through evaluating the model om the test data (X_test,y_test).
        :param Xy: tuple(X,y)
        :param Xy_test: tuple(X_test,y_test)
        :param trials: number of random samplings
        :param kwargs: variable parameters; see TwoBlocksContext.fit() for details
        :return self 
        """
        pass



    @abc.abstractmethod
    def loss(self, X, y, test=False):
        """
        Compute the loss on X. If test=False, the regularization term is not added.
        :param X: observations[n_samples, n_features]
        :param y: target values
        :param test: boolean
        :return loss value (float)
        """
        pass



    def plot(self, title):
        """
        Plot the function in (-3,3)x(-2,2).
        :param title
        """
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection='3d')
        xs = np.linspace(-2, 2, 50)
        ys = np.linspace(-3, 3, 50) 
        X, Y = np.meshgrid(xs, ys)
        XY = np.column_stack([X.ravel(), Y.ravel()])
        Z = self.feedforward(XY).reshape(X.shape)
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        plt.show()



    ###########################
    # Protected Methods
    ###########################

    @abc.abstractmethod
    def _feedforward_MLP(self, X):
        """
        Implement the feedforward pass of the Multilayer Perceptron
        :param X: observations[n_samples, n_features]
        :return f(X) 
        """
        pass



    @abc.abstractmethod
    def _feedforward_RBF(self, X):
        """
        Implement the feedforward pass of the Radial Basis Function network
        :param X: observations[n_samples, n_features]
        return f(X)
        """
        pass



    def _load_hyperparameters(self, hyper_param_cfg_file):
        """
        Load the hyperparameters' values of Q1.
        :param hyper_param_cfg_file: path to the configuration file
        :return configuration dictionary
        """
        with open(hyper_param_cfg_file, 'r') as h_cfg_file:
            hyper_cfg = json.load(h_cfg_file)
        return hyper_cfg



    def _rbf(self, X):
        """
        Compute the rbf with variable SIGMA.
        :param X: argument to rbf(X)
        :return rbf(X)
        """
        rows_idxs = np.array([(i,) for i in range(self.N)])
        minus_matrix = X - self.c[rows_idxs]
        return np.exp(-(np.linalg.norm(minus_matrix, ord=2, axis=2) / self.SIGMA)**2)



    @abc.abstractmethod
    def _save_state(self, state, **kwargs):
        """
        Save the state of the model, e.g. the best parameters, the training time,
        the training loss, the test loss, ...
        :param state: current state
        :param kwargs: variable arguments; see overrided functions for details.
        """
        pass

    

    def _set_hyperparameters(self, hyper_cfg_dict):
        """
        Set the hyperparameters' values.
        :param hyper_cfg_dict: configuration dictionary
        """
        self.N = hyper_cfg_dict['N']
        self.SIGMA = hyper_cfg_dict['SIGMA']
        self.RHO = hyper_cfg_dict['RHO']



    def _set_state(self, **kwargs):
        """
        Set the state of the model. Sample W,b,c
        :param kwargs: variable arguments, see TwoBlocksContext.fit() for details
        """
        if self.algorithm == 'MLP':
            lbound_W = kwargs['lbound_W'] if 'lbound_W' in kwargs.keys() else -4
            ubound_W = kwargs['ubound_W'] if 'ubound_W' in kwargs.keys() else 3
            lbound_b = kwargs['lbound_b'] if 'lbound_b' in kwargs.keys() else -5
            ubound_b = kwargs['ubound_b'] if 'ubound_b' in kwargs.keys() else 5
            mean_W = kwargs['mean_W'] if 'mean_W' in kwargs.keys() else 0
            std_W = kwargs['std_W'] if 'std_W' in kwargs.keys() else 2.5
            mean_b = kwargs['mean_b'] if 'mean_b' in kwargs.keys() else 0
            std_b = kwargs['std_b'] if 'std_b' in kwargs.keys() else 3.5
            self.W = truncnorm.rvs(a=lbound_W, b=ubound_W, 
                                    loc=mean_W, scale=std_W, size=(self.X.shape[1], self.N))
            self.b = truncnorm.rvs(a=lbound_b, b=ubound_b, loc=mean_b, scale=std_b, size=(1, self.N))
                    #np.random.uniform(lbound_b, ubound_b, size=(1, self.N))
        else:
            if kwargs['centers_selection'] == 'random':
                c_idxs = np.random.choice(np.arange(self.X.shape[0]), 
                                            size=self.N, replace=False) 
                self.c = self.X[c_idxs].copy()
            else:
                kmeans = KMeans(n_clusters=self.N, random_state=1939671).fit(self.X)
                self.c = kmeans.cluster_centers_

    
    
    def _tanh(self, s):
        """
        Compute the tanh with variable SIGMA.
        :param s: argument to tanh(s)
        :retuern tanh(s)
        """
        prod = 2 * self.SIGMA * s
        return (np.exp(prod) - 1) / (np.exp(prod) + 1)
