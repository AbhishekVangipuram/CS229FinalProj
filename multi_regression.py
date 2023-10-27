# Basic multinomial regression treating images as 3 dimensional vectors
# Uses whole high resolution image as a baseline

import util
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.linear_model import LogisticRegression

y, train, val, test = util.get_labels_and_split()

y_train, y_val, y_test = y[train], y[val], y[test]

# Make this file once with util.save_high_res()
high_res = np.load("high_res.npy")

X_train = high_res[train]
X_val = high_res[val]
X_test = high_res[test]

lr = LogisticRegression(multi_class='multinomial')

print("training")

X_train = X_train.reshape(X_train.shape[0], -1)

lr.fit(X_train, y_train)

print("predicting")

X_val = X_val.reshape(X_val.shape[0], -1)

predictions = lr.predict(X_val)

print((predictions == y_val).mean())

