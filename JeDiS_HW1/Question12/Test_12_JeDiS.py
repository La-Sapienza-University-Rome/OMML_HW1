import numpy as np
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
change = parentdir + r'\functions'
sys.path.append(change)
import pickle
import pandas as pd

# Load data set (change here by the desired one)
X_test = pd.read_csv('../../DATA.csv')
X_test = np.array(X_test[['x1', 'x2']])


def rbf(X, c, sigma):
    minus_matrix = []
    for i in range(len(c.T)):
        minus_matrix.append(X - c.T[i])
    minus_matrix = np.array(minus_matrix)

    return np.exp(-(np.linalg.norm(minus_matrix, axis=2)/sigma)**2)


def ICanGeneralize(x_new):
    """
    Make prediction with defined parameters and new data
    :param x_new: Input data with two features
    :return predictions
    """

    # Load the parameters needed
    with open('q12_values_for_prediction.pickle', 'rb') as handle:
        dict = pickle.load(handle)

    c = dict['c']
    v = dict['v']
    sigma = dict['sigma']
    x_new = np.array(x_new)

    # Compute prediction
    pred = np.dot(rbf(x_new, c, sigma).T, v)
    return pred


# Execute function to make predictions
y_pred = ICanGeneralize(X_test)
print(y_pred)