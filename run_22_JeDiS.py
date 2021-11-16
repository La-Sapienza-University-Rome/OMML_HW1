from classes_and_functions.Q2.two_phase_class import TwoPhaseContext
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np





# Read and prepare the data
df = pd.read_csv('DATA.csv')

train, test = train_test_split(df, test_size=0.255, random_state=1939671)

X = np.array(train[['x1', 'x2']])
y = np.expand_dims(np.array(train['y']), axis=1)

X_test = np.array(test[['x1', 'x2']])
y_test = np.expand_dims(np.array(test['y']), axis=1)



# Get the object instance
library = 'cvx' # 'numpy'
two_phase = TwoPhaseContext(library=library, algorithm='RBF', hyper_param_cfg_file='config/q_1_2_cfg.json')


# Define the optional parameters. Check classes_and_functions.Q2.two_block.py for details 
if library == 'cvx':
    options = {
        'solver':'ECOS',
        'max_iters':10000,
        'centers_selection':'random' # 'kmeans'
    }
else:
    options = {
        'solver_options':{'method':'SLSQP'},
        'solver':'ECOS',
        'max_iters':10000,
        'centers_selection':'random' # 'kmeans'
    }


# Fit the data --> optimize v /trials/ times with different centers c
trials = 1
random_state = 409473 if trials == 1 else 1939671
two_phase.fit((X,y), (X_test, y_test), trials=trials, random_state=random_state, **options)


# Print the required information 
print(two_phase)


# Plot the function in (-2,2)x(-3,3)
two_phase.plot()


# Save the trained object
two_phase.save_to_file(file_path='./config/model_q_2_2.pickle')