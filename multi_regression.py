# Basic multinomial regression treating images as 3 dimensional vectors
# Uses whole high resolution image as a baseline

import util
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

MAX_ITER = 10000

# def images_to_grayscale(images):
#     return 0.3*images[:,:,:,0] + 0.59*images[:,:,:,1] + 0.11*images[:,:,:,2]

# def high():
#     y, train, val, test = util.get_labels_and_split()

#     y_train, y_val, y_test = y[train], y[val], y[test]

#     # Make this file once with util.save_high_res()
#     high_res = np.load("high_res.npy")


#     # grayscale = 0.3R + 0.59G + 0.11B
#     high_res = images_to_grayscale(high_res)
#     # high_res = 0.3*high_res[:,:,:,0] + 0.59*high_res[:,:,:,1] + 0.11*high_res[:,:,:,2]
#     print(high_res.shape)

#     high_res = high_res.reshape(high_res.shape[0], -1)

#     X_train = high_res[train]
#     X_val = high_res[val]
#     X_test = high_res[test]

#     lr = LogisticRegression(multi_class='multinomial', max_iter = MAX_ITER, verbose=1)

#     print("training")

#     # lr.fit(X_train, y_train)
#     # best_acc = 0
#     # best_reg = 0
#     # best_lr = lr
#     # for reg in 10 ** np.linspace(-5,1,num=10):
#     #     lr = LogisticRegression(multi_class='multinomial', max_iter=MAX_ITER, C=1/reg, verbose=True)
#     #     lr.fit(X_train, y_train)
#     #     val_acc = (lr.predict(X_val)==y_val).mean()
#     #     if val_acc > best_acc:
#     #         best_acc = val_acc
#     #         best_reg = reg
#     #         best_lr = lr
#     #     print(reg, val_acc)
    
#     # # lr = LogisticRegression(multi_class='multinomial', max_iter=10000, penalty=best_pen)
#     # # lr.fit(X_train, y_train)
#     # lr = best_lr

#     lr.fit(X_train, y_train)

#     print("predicting")

#     predictions = lr.predict(X_val)

#     print("val error: " + str((predictions == y_val).mean()))

#     train_preds = lr.predict(X_train)

#     print("train error: " + str((train_preds == y_train).mean()))
    
#     test_preds = lr.predict(X_test)

#     print('test accuracy:', lr.score(X_test, y_test))

#     dump(lr, 'high_multinomial.joblib')


def low(augment=False):
    if augment:
        y, train, val, test = util.get_labels_and_split_augmented()
    else: 
        y, train, val, test = util.get_labels_and_split()

    y_train, y_val, y_test = y[train], y[val], y[test]
    low_res = np.load('low_res_aug.npy') if augment else np.load('low_res.npy')
    low_res = low_res.reshape(low_res.shape[0], -1).astype('uint8')
    
    X_train = low_res[train]
    X_val = low_res[val]
    X_test = low_res[test]

    lr = LogisticRegression(multi_class='multinomial', max_iter=MAX_ITER, verbose=1)

    print("training")

    # lr.fit(X_train, y_train)
    # best_acc = 0
    # best_reg = 0
    # best_lr = lr
    # for reg in 10 ** np.linspace(-5,1,num=10):
    #     lr = LogisticRegression(multi_class='multinomial', max_iter=MAX_ITER, C=1/reg, verbose=True)
    #     lr.fit(X_train, y_train)
    #     val_acc = (lr.predict(X_val)==y_val).mean()
    #     if val_acc > best_acc:
    #         best_acc = val_acc
    #         best_reg = reg
    #         best_lr = lr
    #     print(reg, val_acc)
    
    # lr = LogisticRegression(multi_class='multinomial', max_iter=10000, penalty=best_pen)
    # lr.fit(X_train, y_train)
    # lr = best_lr

    lr.fit(X_train, y_train)

    print("predicting")

    val_preds = lr.predict(X_val)

    print("val accuracy: " + str((val_preds == y_val).mean()))

    train_preds = lr.predict(X_train)

    print("train accuracy: " + str((train_preds == y_train).mean()))

    test_preds = lr.predict(X_test)

    print('test accuracy:', lr.score(X_test, y_test))

    # dump(lr, 'low_multinomial.joblib')




# low()
low(augment=True)

# high()