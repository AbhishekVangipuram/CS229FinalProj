import numpy as np
import util
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA


def low(cat=False, augment=False):
    if augment:
        y, train, val, test = util.get_labels_and_split_augmented()
    else: 
        y, train, val, test = util.get_labels_and_split()

    y_train, y_val, y_test = y[train], y[val], y[test]
    low_res = np.load('low_res_aug.npy') if augment else np.load('low_res.npy')
    print('here 1')
    low_res = low_res.reshape(low_res.shape[0], -1).astype(int)

    X_train = low_res[train]
    X_val = low_res[val]
    X_test = low_res[test]

    # ros = RandomOverSampler(random_state=0)
    # rus = RandomUnderSampler(random_state=0)
    # freqs = {1:500, 2:500, 4:500, 7:500}
    # rus = RandomUnderSampler(random_state=0, sampling_strategy=freqs, replacement=True)
    # X_under, y_under = rus.fit_resample(X_train, y_train)


    # print('here 2')
    # X_train_resample, y_train_resample = ros.fit_resample(X_under, y_under)
    # print('here 3')
    # X_val_resample, y_val_resample = ros.fit_resample(X_val, y_val)
    # print('here 4')
    # X_test_resample, y_test_resample = ros.fit_resample(X_test, y_test)
    # print('here 5')

    # pca = PCA(n_components=5)
    # pca.fit(X_val_resample, y_val_resample)
    # print('here pca')
    # print(pca.explained_variance_ratio_)
    # pca.transform(X_train_resample)
    # print('here pca2')
    # pca.transform(X_val_resample)
    # pca.transform(X_test_resample)

    
    if cat:
        nb = CategoricalNB(force_alpha=True)
    else: 
        nb = MultinomialNB(force_alpha=True)
  
    nb.fit(X_train, y_train)
    
    print('here 6')
    print('train acc', nb.score(X_train, y_train))
    print('test acc', nb.score(X_test, y_test))
    X_test = np.concatenate((X_test, X_val))
    y_test = np.concatenate((y_test, y_val))
    print('test+val acc', nb.score(X_test, y_test))

    # TRY WITH FULL OVERSAMPLING:
    # print(len(X_train_resample))
    # nb.fit(X_train_resample, y_train_resample)
    # print('train acc', nb.score(X_train_resample, y_train_resample))
    # print('test acc', nb.score(X_test_resample, y_test_resample))
    # X_test = np.concatenate((X_test_resample, X_val_resample))
    # y_test = np.concatenate((y_test_resample, y_val_resample))
    # print('test+val acc', nb.score(X_test_resample, y_test_resample))


low(augment=True)
# low(cat=True, augment=True)