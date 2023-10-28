# Basic multinomial regression treating images as 3 dimensional vectors
# Uses whole high resolution image as a baseline

import util
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.linear_model import LogisticRegression

def high():
    y, train, val, test = util.get_labels_and_split()

    y_train, y_val, y_test = y[train], y[val], y[test]

    # Make this file once with util.save_high_res()
    high_res = np.load("high_res.npy")

    high_res.reshape(high_res.shape[0], -1)

    X_train = high_res[train]
    X_val = high_res[val]
    X_test = high_res[test]

    lr = LogisticRegression(multi_class='multinomial')

    print("training")

    lr.fit(X_train, y_train)

    print("predicting")

    predictions = lr.predict(X_val)

    print("val error: " + str((predictions == y_val).mean()))

    train_preds = lr.predict(X_train)

    print("train error: " + str((train_preds == y_train).mean()))


def low():
    y, train, val, test = util.get_labels_and_split()

    y_train, y_val, y_test = y[train], y[val], y[test]

    # Make this file once with util.save_high_res()
    low_res = util.get_low_res()

    low_res = low_res.reshape(low_res.shape[0], -1)

    X_train = low_res[train]
    X_val = low_res[val]
    X_test = low_res[test]

    lr = LogisticRegression(multi_class='multinomial', max_iter=10000)

    print("training")

    lr.fit(X_train, y_train)

    print("predicting")

    predictions = lr.predict(X_val)

    print("val accuracy: " + str((predictions == y_val).mean()))

    train_preds = lr.predict(X_train)

    print("train accuracy: " + str((train_preds == y_train).mean()))


low()