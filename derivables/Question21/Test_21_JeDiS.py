"""
For making the script run:
- set the parent folder as the current directory
- run: python ./Question21/run_21_JeDiS.py
"""


import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from functions.Q2.two_phase_class import TwoPhaseContext




def ICanGeneralize(X):
    """
    Take the data X and return the predictions Y.

    :param X: observations[n_samples, n_features]

    :return
    """
    two_phase = TwoPhaseContext.load_from_file(r'./config/model_q_2_1.pickle')
    return two_phase.predict(X)