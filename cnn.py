# Basic multinomial regression treating images as 3 dimensional vectors
# Uses whole high resolution image as a baseline

import util
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

import tensorflow as tf
from tensorflow.keras import layers, models



def high():
    y, train, val, test = util.get_labels_and_split()

    y_train, y_val, y_test = y[train], y[val], y[test]

    y_train = tf.one_hot(y_train, depth=8)
    y_val   = tf.one_hot(y_val,   depth=8)
    y_test  = tf.one_hot(y_test,  depth=8)


    # Make this file once with util.save_high_res()
    high_res = util.get_high_res()

    # high_res.reshape(high_res.shape[0], -1)

    X_train = high_res[train]
    X_val = high_res[val]
    X_test = high_res[test]

    print(X_train.shape)

    cnn = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(8, activation='softmax')
        ])
    
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("training")

    cnn.fit(X_train, y_train, epochs=10, batch_size=32)

    print("predicting")

    val_loss, val_accuracy = cnn.evaluate(X_val, y_val)

    print("val acc: " + str(val_accuracy))

    train_loss, train_accuracy = cnn.evaluate(X_train, y_train)

    print("train acc: " + str(train_accuracy))


def low():
    y, train, val, test = util.get_labels_and_split()

    y_train, y_val, y_test = y[train], y[val], y[test]

    y_train = tf.one_hot(y_train, depth=8)
    y_val   = tf.one_hot(y_val,   depth=8)
    y_test  = tf.one_hot(y_test,  depth=8)


    # Make this file once with util.save_high_res()
    high_res = util.get_low_res()

    high_res.reshape(high_res.shape[0], -1)

    X_train = high_res[train]
    X_val = high_res[val]
    X_test = high_res[test]

    print(X_train.shape)

    cnn = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(8, activation='softmax')
        ])
    
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("training")

    cnn.fit(X_train, y_train, epochs=10, batch_size=32)

    print("predicting")

    val_loss, val_accuracy = cnn.evaluate(X_val, y_val)

    print("val acc: " + str(val_accuracy))

    train_loss, train_accuracy = cnn.evaluate(X_train, y_train)

    print("train acc: " + str(train_accuracy))


high()