import numpy as np
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
change = parentdir + r'\functions'
sys.path.append(change)
from functions_q3 import *
import pickle
import pandas as pd

# Load data set (change here by the desired one)
X_test = pd.read_csv('../../DATA.csv')
X_test = np.array(X_test[['x1', 'x2']])

def ICanGeneralize(x_new):
    """
    Make prediction with defined parameters and new data

    :param x_new: Input data with two features

    :return predictions
    """

    # Load the parameters needed
    with open('q3_values_for_prediction.pickle', 'rb') as handle:
        dict = pickle.load(handle)
    W = dict['W']
    b = dict['b']
    v = dict['v']
    sigma = dict['sigma']
    x_new = np.array(x_new)

    # Compute prediction
    linear_layer = (np.dot(x_new, W) + b)
    activation = tanh(linear_layer, sigma)
    pred = np.dot(activation, v)
    return pred

# Execute function to make predictions
y_pred = ICanGeneralize(X_test)
print(y_pred)
