from classes_and_functions.Q2.model_implementations import *
import pickle






class TwoPhaseContext:
    """
    Context class for the Question 2, internally selects the library (CVXPY or Numpy) and 
    the algorithm (MLP or RBF) and provides a unique API to the client.
    This is the only class that should be explicitly instantiated.

    The public API is:
    - eval(tuple(X_test,y_test))
    - fit(tuple(X,y), tuple(X_test,y_test), trials=1, **kwargs)
    - get_state()
    - load_from_file(file_path)
    - loss(X, y, test=False)
    - plot()
    - predict(X)
    - save_to_file(file_path)
    - __str__() -> string representation of the state of the object ModelCVX or ModelNumpy
    """

    ###########################
    # Static Methods
    ###########################

    @staticmethod
    def load_from_file(file_path):
        """
        Load and return the pre-trained object in file_path.

        :param file_path: path to the object file

        :return new TwoPhaseContext instance 
        """
        with open(file_path, 'rb') as in_file:
            obj = pickle.load(in_file)
        return obj



    ###########################
    # Magic Methods
    ###########################

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



    def __str__(self):
        return str(self.model)



    ###########################
    # Public Methods
    ###########################

    def eval(self, Xy_test):
        """
        Evaluate the model on new data. No regularization is added to the loss.

        :param Xy_test: tuple (observations[n_samples, n_features], target)

        :return loss value (float)
        """
        return self.model.eval(Xy_test)



    def fit(self, Xy, Xy_test, trials=1, **kwargs):
        """
        Run the model /trials/ times on the training data Xy and select the best
        trial through evaluating the model on the test data Xy_test.

        :param Xy: tuple (X, y); training data
        :param Xy_test: tuple (X_test, y_test); test data
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
        return self.model.fit(Xy, Xy_test, trials, **kwargs)



    def get_state(self):
        """
        Get the model state. It must be called after fit().

        :param

        :return state dictionary
        """
        return self.model.state


    
    def loss(self, X, y, test=False):
        """
        Compute the loss on X and y. If test=False, add the regularization term.

        :param X: observations
        :param y: target values
        :param test: boolean

        :return loss value (float)
        """
        return self.model.loss(X, y, test)



    def plot(self):
        """
        Plot the function in (-2,2)x(-3,3).
        """
        self.model.plot(f'Two-Phase method {self.model.algorithm}')
        

    
    def predict(self, X):
        """
        Return the predictions y on the data X

        :param X: observations[n_samples, n_features]

        :return y[n_samples,]
        """
        return self.model.feedforward(X)



    def save_to_file(self, file_path):
        """
        Save the current object to a pickle file.

        :param file_path
        """
        with open(file_path, 'wb') as out_file:
            pickle.dump(self, out_file, protocol=pickle.HIGHEST_PROTOCOL)
