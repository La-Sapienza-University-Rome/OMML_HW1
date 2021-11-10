from classes_and_functions.Q2.two_blocks import *



def ICanGeneralize(X):
    """
    Take the data X and return the predictions Y.

    :param X: observations[n_samples, n_features]

    :return
    """
    tbc = TwoBlocksContext.load_from_file('./config/model_q_2_2.pickle')
    return tbc.predict(X)