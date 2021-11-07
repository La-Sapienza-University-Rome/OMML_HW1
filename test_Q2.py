from Q2_experimental import *
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(1939671)

df = pd.read_csv('DATA.csv')

train, test = train_test_split(df, test_size=0.255, random_state=1939671)

X = np.array(train[['x1', 'x2']])
y = np.expand_dims(np.array(train['y']), axis=1)

X_test = np.array(test[['x1', 'x2']])
y_test = np.expand_dims(np.array(test['y']), axis=1)



el = ExtremeLearning(library='cvx', algorithm='RBF', hyper_param_cfg_file='config/q_1_1_cfg.json')


options = {
    'solver':'ECOS',
    'lbound_W':-2.5,
    'ubound_W':1,
    'mean_W':0,
    'std_W':1,
    'lbound_b':-3,
    'ubound_b':1.8,
    'mean_b':-0.1,
    'std_b':0.97,
    'centers_selection':'random'
}


el.fit((X,y), (X_test, y_test), trials=150, **options)

print(el)

print(f'Val loss without regularization {round(el.eval((X_test, y_test)), 5)}')

el.plot()