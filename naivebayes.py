import numpy as np
import util
from sklearn.naive_bayes import MultinomialNB, CategoricalNB




def low(cat=False):
    y, train, val, test = util.get_labels_and_split()

    y_train, y_val, y_test = y[train], y[val], y[test]

    # Make this file once with util.save_high_res()
    low_res = np.load('low_res.npy')

    low_res = low_res.reshape(low_res.shape[0], -1).astype(int)

    X_train = low_res[train]
    X_val = low_res[val]
    X_test = low_res[test]
    
    if cat:
        nb = CategoricalNB(force_alpha=True)
    else: 
        nb = MultinomialNB(force_alpha=True)
  
    nb.fit(X_train, y_train)

    print('train acc', nb.score(X_train, y_train))
    print('test acc', nb.score(X_test, y_test))
    X_test = np.concatenate((X_test, X_val))
    y_test = np.concatenate((y_test, y_val))
    print('test+val acc', nb.score(X_test, y_test))


low()
low(cat=True)