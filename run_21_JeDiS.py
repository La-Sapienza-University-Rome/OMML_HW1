from classes_and_functions.Q2.two_blocks import TwoBlocksContext
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np




# Set seed
np.random.seed(1939671)

# Read and prepare the data
df = pd.read_csv('DATA.csv')

train, test = train_test_split(df, test_size=0.255, random_state=1939671)

X = np.array(train[['x1', 'x2']])
y = np.expand_dims(np.array(train['y']), axis=1)

X_test = np.array(test[['x1', 'x2']])
y_test = np.expand_dims(np.array(test['y']), axis=1)



# Get the object instance
library = 'cvx' # 'numpy'
tbc = TwoBlocksContext(library=library, algorithm='MLP', hyper_param_cfg_file='config/q_1_1_cfg.json')


# Define the optional parameters. Check classes_and_functions.Q2.two_block.py for details 
options = {
    'solver':'ECOS',
    'max_iters':10000,
    'lbound_W':-1.8,
    'ubound_W':1.558,
    'mean_W':0.02,
    'std_W':1,
    'lbound_b':-2.9,
    'ubound_b':3.8,
    'mean_b':-0.105,
    'std_b':0.97
}
# # if library=numpy
# options = {
#     'solver_options':{'method':'BFGS'},
#     'max_iters':10000,
#     'lbound_W':-1.8,
#     'ubound_W':1.558,
#     'mean_W':0.02,
#     'std_W':1,
#     'lbound_b':-2.9,
#     'ubound_b':3.8,
#     'mean_b':-0.105,
#     'std_b':0.97
# }


# Fit the data --> optimize v /trials/ times with different realizations of W and b
tbc.fit((X,y), (X_test, y_test), trials=150, **options)


# Print the required information 
print(tbc)


# Plot the function in (-2,2)x(-3,3)
tbc.plot()


# Save the trained object
tbc.save_to_file(file_path='./config/model_q_2_1.pickle')