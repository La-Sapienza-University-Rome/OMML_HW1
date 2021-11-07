import numpy as np
import cvxpy as cvx
import time
import json
from scipy.optimize import least_squares
from scipy.stats import truncnorm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import abc


np.random.seed(1939671)


class Model(abc.ABC):
    """
    Abstract class that provides the common interface for subclasses using different libraries.
    Currently, it is the parent class of two classes: ModelCVX, which implements the algorithms MLP and
    RBF by means of the library CVXPY, ModelNumpy, which implements them in Numpy and Scipy.

    The public API is [see the description of each function for details]:
    - eval(tuple(X_test,y_test))
    - feedforward(X)
    - fit(tuple(X,y), tuple(X_val,y_val), trials=1, **kwargs)
    - loss(X, y, test=False)
    - plot()
    - __str__() -> returns the string representation of the state of the object
    """

    def __init__(self):
        self.state = {}

    
    def _load_hyperparameters(self, hyper_param_cfg_file):
        """
        Load the hyperparameters' values of Q1.

        :param hyper_param_cfg_file: path to the configuration file

        :return configuration dictionary
        """
        with open(hyper_param_cfg_file, 'r') as h_cfg_file:
            hyper_cfg = json.load(h_cfg_file)
        return hyper_cfg

    
    def _set_hyperparameters(self, hyper_cfg_dict):
        """
        Set the hyperparameters' values.

        :param hyper_cfg_dict: configuration dictionary
        """
        self.N = hyper_cfg_dict['N']
        self.SIGMA = hyper_cfg_dict['SIGMA']
        self.RHO = hyper_cfg_dict['RHO']

    
    def _tanh(self, s):
        """
        Compute the tanh with variable SIGMA.

        :param s: argument to tanh(s)

        :retuern tanh(s)
        """
        prod = 2 * self.SIGMA * s
        return (np.exp(prod) - 1) / (np.exp(prod) + 1)

    
    def _rbf(self, X):
        """
        Compute the rbf with variable SIGMA.

        :param X: argument to rbf(X)

        :return rbf(X)
        """
        rows_idxs = np.array([(i,) for i in range(self.N)])
        minus_matrix = X - self.c[rows_idxs]
        return np.exp(-(np.linalg.norm(minus_matrix, ord=2, axis=2) / self.SIGMA)**2)


    def __str__(self):
        """
        Return fancy string representation of the object's state
        """
        str_repr = ''
        for key, value in self.state.items():
            str_repr += '\n## {key}: {value}\n'.format(key=key, value=value)
        return str_repr


    def _set_state(self, **kwargs):
        """
        Set the state of the model. Sample W,b,c

        :param kwargs: variable arguments, see ExtremeLearning.fit() for details
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
            self.b = np.random.uniform(lbound_b, ubound_b, size=(1, self.N))#truncnorm.rvs(a=lbound_b, b=ubound_b, 
                     #               loc=mean_b, scale=std_b, size=(1, self.N))
        else:
            if kwargs['centers_selection'] == 'random':
                c_idxs = np.random.choice(np.arange(self.X.shape[0]), 
                                            size=self.N, replace=False) 
                self.c = self.X[c_idxs].copy()
            else:
                kmeans = KMeans(n_clusters=self.N, random_state=1939671).fit(self.X)
                self.c = kmeans.cluster_centers_


    def plot(self, title):
        """
        Plot the function in (-3,3)x(-2,2).

        :param title
        """
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection='3d')
        x = np.linspace(-3, 3, 50) 
        y = np.linspace(-2, 2, 50)
        X_plot, Y_plot = np.meshgrid(x, y) 
        Z = np.array( [[self.feedforward(np.array([x1, x2])) for x2 in y] for x1 in x ])

        ax.plot_surface(X_plot, Y_plot, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        plt.show()


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


    @abc.abstractmethod
    def _save_state(self, state, **kwargs):
        """
        Save the state of the model, e.g. the best parameters, the training time,
        the training loss, the validation loss, ...
        
        :param state: current state
        :param kwargs: variable arguments; see overrided functions for details.
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


    @abc.abstractmethod
    def feedforward(self, X):
        """
        Return the feedforward pass according to the selected algorithm.

        :param X: observations[n_samples, n_features]

        :return f(X)
        """
        pass

    
    @abc.abstractmethod
    def fit(self, Xy, Xy_val, trials=1, **kwargs):
        """
        Run the model /trials/ times on the training data (X,y). Select the best configuration 
        through evaluating the model om the validation data (X_val,y_val).

        :param Xy: tuple(X,y)
        :param Xy_val: tuple(X_val,y_val)
        :param trials: number of random samplings
        :param kwargs: variable parameters; see ExtremeLearning.fit() for details

        :return self 
        """
        pass


    @abc.abstractmethod
    def eval(self, Xy_test):
        """
        Evaluate the model on the new data (X_test, y_test). No regulazation is added to the loss

        :param Xy_test: tuple(X_test, y_test)

        :return loss value without the regularization term
        """
        pass




class ModelCVX(Model):

    def __init__(self, algorithm, hyper_param_cfg_file):
        super().__init__()
        self._set_hyperparameters(self._load_hyperparameters(hyper_param_cfg_file))
        self.algorithm = algorithm


    def _feedforward_MLP(self, X):
        linear_layer = np.dot(X, self.W) + self.b
        activation = self._tanh(linear_layer)
        return cvx.matmul(activation, self.v)


    def _feedforward_RBF(self, X):
        return cvx.matmul(self._rbf(X).T, self.v)

    
    def _set_state(self, **kwargs):
        self.v = cvx.Variable(shape=(self.N, 1), name='v')
        super()._set_state(**kwargs)


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
                                'best_b': self.b.copy()}
            else:
                self.state = {'best_c': self.c.copy()}
            self.state['best_train_loss'] = state.value
            self.state['best_val_loss'] = kwargs['val_loss']
            self.state['N'] = self.N
            self.state['SIGMA'] = self.SIGMA
            self.state['RHO'] = self.RHO
            self.state['solver_name'] = state.solver_stats.solver_name
            self.state['solve_time'] = state.solver_stats.solve_time
            self.state['num_iters'] = state.solver_stats.num_iters
            self.state['problem_status'] = state.status
            self.state['best_v'] = np.array(self.v.value).ravel()


    def _feedforward(self, X):
        if self.algorithm == 'MLP':
            return self._feedforward_MLP(X)
        else:
            return self._feedforward_RBF(X)


    def feedforward(self, X):
        return self._feedforward(X).value[0,0]


    def loss(self, X, y, test=False):
        P = y.shape[0]
        pred = self._feedforward(X)
        res = cvx.sum_squares(pred - y) / (2*P)
        if not test:
            res = res + 0.5 * self.RHO * cvx.square(cvx.norm2(self.v))
        return res


    def fit(self, Xy, Xy_val, trials=1, **kwargs):
        self.X, self.y = Xy
        X_val, y_val = Xy_val
        max_iters = kwargs['max_iters'] if 'max_iters' in kwargs.keys() else 10000
        solver = kwargs['solver'] if 'solver' in kwargs.keys() else None
        best_val_loss = 1e4
        start = time.time()
        for _ in tqdm(range(trials)):
            self._set_state(**kwargs)
            cvx_problem = cvx.Problem(cvx.Minimize(self.loss(X=self.X, y=self.y, test=False)))
            cvx_problem.solve(solver=solver, verbose=False, max_iters=max_iters)
            current_val_loss = self.loss(X_val, y_val, test=False).value
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self._save_state(cvx_problem, val_loss=best_val_loss, restore=False)
        stop = time.time()
        self._save_state(None, restore=True, total_elapsed_time=f'{round(stop-start, 3)}s')
        return self

    
    def eval(self, Xy_test):
        X_test, y_test = Xy_test
        return self.loss(X_test, y_test, test=True).value



class ModelNumpy(Model):

    def __init__(self, algorithm, hyper_param_cfg_file):
        super().__init__()
        self._set_hyperparameters(self._load_hyperparameters(hyper_param_cfg_file))
        self.algorithm = algorithm


    def _feedforward_MLP(self, X):
        linear_layer = np.dot(X, self.W) + self.b
        activation = self._tanh(linear_layer)
        return np.dot(activation, self.v)


    def _feedforward_RBF(self, X):
        return np.dot(self._rbf(X).T, self.v)

    
    def _set_state(self, **kwargs):
        self.v = np.random.normal(size=self.N)
        super()._set_state(**kwargs)


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
                              'best_b': self.b.copy()}
            else:
                self.state = {'best_c': self.c.copy()}
            self.state['best_v'] = state.x
            self.state['best_train_loss'] = state.cost
            self.state['best_val_loss'] = kwargs['val_loss']
            self.state['N'] = self.N
            self.state['SIGMA'] = self.SIGMA
            self.state['RHO'] = self.RHO
            self.state['solve_time'] = kwargs['solver_time']
            self.state['num_iters'] = state.nfev
            self.state['problem_status'] = state.success


    def feedforward(self, X):
        if self.algorithm == 'MLP':
            return self._feedforward_MLP(X)[0]
        else:
            return self._feedforward_RBF(X)[0]


    def loss(self, X, y, test=False):
        P = y.shape[0]
        pred = self.feedforward(X)
        res = np.sum((pred - y)**2) / (2*P)

        if not test:
            res = res + 0.5 * self.RHO * np.linalg.norm(self.v)**2

        return res


    def _loss(self, v, **kwargs):
        self.v = np.expand_dims(v, axis=1)
        return self.loss(self.X, self.y, test=False)


    def _gradient(self, v, **kwargs):
        self.v = np.expand_dims(v, axis=1)
        if self.algorithm == 'MLP':
            linear_layer = np.dot(self.X, self.W) + self.b
            a1 = self._tanh(linear_layer)
        else:
            a1 = self._rbf(self.X).T
        P = len(self.y)
        z2 = np.dot(a1, self.v)
        dJdf = (z2 - self.y) / P
        dz2dv = a1
        return np.squeeze(np.dot(dz2dv.T, dJdf) + self.RHO * self.v)


    def fit(self, Xy, Xy_val, trials=1, **kwargs):
        self.X, self.y = Xy
        X_val, y_val = Xy_val
        best_val_loss = 1e4
        start = time.time()
        for _ in tqdm(range(trials)):
            t0 = time.time()
            self._set_state(**kwargs)
            problem_res = least_squares(self._loss, self.v, jac=self._gradient)
            self.v = problem_res.x
            current_val_loss = self.loss(X_val, y_val, test=False)
            t1 = time.time()
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self._save_state(problem_res, val_loss=best_val_loss, solver_time=round(t1-t0, 5), restore=False)
        stop = time.time()
        self._save_state(problem_res, restore=True, total_elapsed_time=f'{round(stop-start, 3)}s')
        return self

    
    def eval(self, Xy_test):
        X_test, y_test = Xy_test
        return self.loss(X_test, y_test, test=True)


class ExtremeLearning:
    """
    Context class for the Question 2, internally selects the library (CVXPY or Numpy) and 
    the algorithm (MLP or RBF) and provides a unique API to the client.
    This is the only class that should be explicitly instantiated.

    The public API is:
    - eval(tuple(X_test,y_test))
    - fit(tuple(X,y), tuple(X_val,y_val), trials=1, **kwargs)
    - get_state()
    - loss(X, y, test=False)
    - plot()
    - predict(X)
    - __str__() -> string representation of the state of the object ModelCVX or ModelNumpy
    """

    def __init__(self, library, algorithm, hyper_param_cfg_file):
        """
        :param library: either cvx or numpy
        :param algorithm: either MLP or RBF
        :param hyper_param_cfg_file: path to the configuration file
        """
        if library == 'cvx':
            self.model = ModelCVX(algorithm, hyper_param_cfg_file)
        else:
            self.model = ModelNumpy(algorithm, hyper_param_cfg_file)


    def get_state(self):
        """
        Get the model state. It must be called after fit().

        :param

        :return state dictionary
        """
        return self.model.state

    
    def predict(self, X):
        """
        Return the predictions y on the data X

        :param X: observations[n_samples, n_features]

        :return y[n_samples,]
        """
        return self.model.feedforward(X)


    def fit(self, Xy, Xy_val, trials=1, **kwargs):
        """
        Run the model /trials/ times on the training data Xy and select the best
        trial through evaluating the model on the validation data Xy_val.

        :param Xy: tuple (X, y); training data
        :param Xy_val: tuple (X_val, y_val); validation data
        :param trials: number of random samplings of W,b,c
        :param kwargs: further options for the fit:
                :key lbound_W: lower bound of the interval of W
                :key ubound_W: upper bound of the interval of W
                :key lbound_b: lower bound of the interval of b
                :key ubound_b: upper bound of the interval of b
                :key mean_W: mean of the truncated Gaussian of W
                :key std_W: standard deviation of the truncated Gaussian of W
                :key mean_b: mean of the truncated Gaussian of b
                :key std_b: standard deviation of the truncated Gaussian of b
                :key centers_selection: either random or kmeans
                :key max_iters: maximum number of iterations (cvx only)
                :key solver: solver to use (cvx only)
        
        :return model
        """
        return self.model.fit(Xy, Xy_val, trials, **kwargs)

    
    def loss(self, X, y, test=False):
        """
        Compute the loss on X and y. If test=False, add the regularization term.

        :param X: observations
        :param y: target values
        :param test: boolean

        :return loss value (float)
        """
        return self.model.loss(X, y, test)


    def eval(self, Xy_test):
        """
        Evaluate the model on new data. No regularization is added to the loss.

        :param Xy_test: tuple (observations[n_samples, n_features], target)

        :return loss value (float)
        """
        return self.model.eval(Xy_test)


    def plot(self):
        """
        Plot the function in (-3,3)x(-2,2).
        """
        self.model.plot(f'Extreme Learning {self.model.algorithm}')

    
    def __str__(self):
        return self.model.__str__()