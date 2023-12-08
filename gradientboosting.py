# testing pca with multinomial regression, naivebayes, random forest

import numpy as np
import util
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


low_res_aug = np.load('low_res_aug.npy')

y, train, val, test = util.get_labels_and_split_augmented()

y_train, y_val, y_test = y[train], y[val], y[test]

flat = low_res_aug.reshape((low_res_aug.shape[0], -1)).astype('uint8')

X_train = flat[train]
X_val = flat[val]
X_test = flat[test]

best_rf_val_acc, best_lr_val_acc, best_nb_val_acc = 0, 0, 0
best_rf_dim, best_lr_dim, best_nb_dim = 0,0,0

for pca_dim in [20]:
    print('PCA DIM ==', pca_dim)
    if pca_dim == 0:
        X_train_p, X_val_p, X_test_p = X_train, X_val, X_test
    else:
        pca = PCA(n_components=pca_dim)
        pca.fit(X_val)
        X_train_p, X_val_p, X_test_p = pca.transform(X_train), pca.transform(X_val), pca.transform(X_test)
    # gradient boosting
    rf = GradientBoostingClassifier(n_estimators = 100, max_depth = 2, learning_rate=0.1, random_state=0)
    rf.fit(X_train_p, y_train)
    print('GRADIENT BOOSTING:', 'test:', rf.score(X_test_p, y_test), 'val:', rf.score(X_val_p, y_val), 'train:', rf.score(X_train_p, y_train))
    if rf.score(X_val_p, y_val) >= best_rf_val_acc:
        best_rf_val_acc = rf.score(X_val_p, y_val)
        best_rf_dim = pca_dim
    # multinomial regression
    # lr = LogisticRegression(multi_class='multinomial', max_iter=1000)
    # lr.fit(X_train_p, y_train)
    # print('MULTINOMIAL:', 'test:', lr.score(X_test_p, y_test), 'val:', lr.score(X_val_p, y_val), 'train:', lr.score(X_train_p, y_train))
    # if lr.score(X_val_p, y_val) >= best_lr_val_acc:
    #     best_lr_val_acc = lr.score(X_val_p, y_val)
    #     best_lr_dim = pca_dim
    # naive bayes
    # nb = MultinomialNB(force_alpha=True)
    # nb.fit(X_train_p, y_train)
    # print('NAIVE BAYES:', 'test:', nb.score(X_test_p, y_test), 'val:', nb.score(X_val_p, y_val), 'train:', nb.score(X_train_p, y_train))
    # if nb.score(X_val_p, y_val) > best_nb_val_acc:
    #     best_nb_val_acc = nb.score(X_val_p, y_val)
    #     best_nb_dim = pca_dim
    

print('RF:', best_rf_dim)
# print('LR:', best_lr_dim)

