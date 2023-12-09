# Basic multinomial regression treating images as 3 dimensional vectors
# Uses whole high resolution image as a baseline

import util
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
\

import matplotlib.pyplot as plt


counts = [ 204, 1736,  771, 405,  624,  240,  474, 1413]
props = np.array(counts) / sum(counts)
weights = 1 / props
def wcce():
    def wcce_(y_true, y_pred):
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce.__call__(y_true=y_true, y_pred=y_pred, sample_weight=weights)
    return wcce_

def high():
    y, train, val, test = util.get_labels_and_split()

    y_train, y_val, y_test = y[train], y[val], y[test]

    y_train = tf.one_hot(y_train, depth=8)
    y_val   = tf.one_hot(y_val,   depth=8)
    y_test  = tf.one_hot(y_test,  depth=8)


    # Make this file once with util.save_high_res()
    high_res = np.load('high_res.npy')

    # high_res.reshape(high_res.shape[0], -1)

    X_train = high_res[train]
    X_val = high_res[val]
    X_test = high_res[test]

    print(X_train.shape)

    cnn = models.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(500, 500, 3)),
        layers.Dropout(0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(8, activation='softmax')
        ])
    
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("training")

    history = cnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

    print("predicting")

    val_loss, val_accuracy = cnn.evaluate(X_val, y_val)

    print("val acc: " + str(val_accuracy))

    train_loss, train_accuracy = cnn.evaluate(X_train, y_train)

    print("train acc: " + str(train_accuracy))

    training_loss = history.history["accuracy"]
    validation_loss = history.history["val_accuracy"]

    plt.figure(figsize=(10,5))
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # test_loss, test_accuracy = cnn.evaluate(X_test, y_test)

    # print("test acc: " + str(test_accuracy))

    cnn.save('high_cnn.keras')
def low():

    y, train, val, test = util.get_labels_and_split_augmented()
    
    y_train, y_val, y_test = y[train], y[val], y[test]
    low_res = np.load('low_res_aug.npy')

    X_train = low_res[train]
    X_val = low_res[val]
    X_test = low_res[test]

    y_train = tf.one_hot(y_train, depth=8)
    y_val   = tf.one_hot(y_val,   depth=8)
    y_test  = tf.one_hot(y_test,  depth=8)


    

    # low_res.reshape(low_res.shape[0], -1)

    

    print(X_train.shape)

    # cnn = models.Sequential([
    #     layers.Conv2D(32, (5, 5), activation='relu', input_shape=(147, 147, 3)),
    #     layers.Dropout(0.3),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    #     layers.Dropout(0.3),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    #     layers.Dropout(0.3),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(64, (3, 3), activation='relu'),
    #     layers.Dropout(0.3),
    #     layers.Flatten(),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dropout(0.5),
    #     layers.Dense(8, activation='softmax')
    #     ])
    cnn = models.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(147, 147, 3)),
        layers.Dropout(0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu',),
        layers.Dropout(0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu',),
        layers.Dropout(0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu',),
        layers.Dropout(0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu',),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(8, activation='softmax')
        ])
    
    cnn.compile(optimizer='adam', loss=wcce() , metrics=['accuracy'])

    print("training")

    history = cnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
    # possibel things to remedy the majority of the one urban density class
    # add weigths in loss function to make unrepresented class moreimportant
    # within each batch, try to have about an equal proportion of each ubran class
    # coudl add augmentations i.e. flip, random noise, crop and zoom

    # maybe do pca for dimension reduction for the traiditional models (miltinomial, NB)

    print("predicting")

    val_loss, val_accuracy = cnn.evaluate(X_val, y_val)

    print("val acc: " + str(val_accuracy))

    train_loss, train_accuracy = cnn.evaluate(X_train, y_train)

    print("train acc: " + str(train_accuracy))

    test_loss, test_accuracy = cnn.evaluate(X_test, y_test)

    print("test acc: " + str(test_accuracy))

    training_loss = history.history["accuracy"]
    validation_loss = history.history["val_accuracy"]

    plt.figure(figsize=(10,5))
    plt.plot(training_loss, label="Training Accuracy")
    plt.plot(validation_loss, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    y, train, val, test = util.get_labels_and_split()

    y_train, y_val, y_test = y[train], y[val], y[test]

    y_train = tf.one_hot(y_train, depth=8)
    y_val   = tf.one_hot(y_val,   depth=8)
    y_test  = tf.one_hot(y_test,  depth=8)

    low_res = util.get_low_res()
   # low_res = np.load('low_res.npy')

    # low_res.reshape(low_res.shape[0], -1)

    X_train = low_res[train]
    X_val = low_res[val]
    X_test = low_res[test]

    val_loss, val_accuracy = cnn.evaluate(X_val, y_val)

    print("unaugmented val acc: " + str(val_accuracy))


    # Only run at end
    # test_loss, test_accuracy = cnn.evaluate(X_test, y_test)

    # print("unaugmented test acc: " + str(test_accuracy))



# low(augment=True)
low()
# high()