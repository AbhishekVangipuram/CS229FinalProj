import util
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


y, train, val, test = util.get_labels_and_split_augmented() # Can use anything here

def high():
    y, train, val, test = util.get_labels_and_split()

    y_train, y_val, y_test = y[train], y[val], y[test]

    low_res = util.get_low_res_augmented() # Match abvoe
    # low_res = np.load('low_res.npy')
    y_train = tf.one_hot(y_train, depth=8)
    y_val   = tf.one_hot(y_val,   depth=8)
    y_test  = tf.one_hot(y_test,  depth=8)

    high_res = np.load('high_res.npy')

    # high_res.reshape(high_res.shape[0], -1)

    X_train = high_res[train]
    X_val = high_res[val]
    X_test = high_res[test]

    print(X_train.shape)

    base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(147, 147, 3))

    # Freeze the pre-trained layers
    for layer in base_model.layers[:-10]: #new
        layer.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    print("predicting")

    val_loss, val_accuracy = model.evaluate(X_val, y_val)

    print("val acc: " + str(val_accuracy))


    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print("test acc: " + str(test_accuracy))

    # cnn.save('low_cnn.keras')

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

    model.save('hires_image_classifier.h5')


def low():
    y, train, val, test = util.get_labels_and_split_augmented() # Can use anything here

    y_train, y_val, y_test = y[train], y[val], y[test]

    y_train = tf.one_hot(y_train, depth=8)
    y_val   = tf.one_hot(y_val,   depth=8)
    y_test  = tf.one_hot(y_test,  depth=8)

    low_res = util.get_low_res_augmented() # Match abvoe
    #low_res = np.load('low_res_aug.npy')

    # low_res.reshape(low_res.shape[0], -1)

    X_train = low_res[train]
    X_val = low_res[val]
    X_test = low_res[test]

    print(X_train.shape)

    base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(147, 147, 3))

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32)

    print("predicting")

    val_loss, val_accuracy = model.evaluate(X_val, y_val)

    print("val acc: " + str(val_accuracy))

    train_loss, train_accuracy = model.evaluate(X_train, y_train)

    print("train acc: " + str(train_accuracy))

    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print("test acc: " + str(test_accuracy))

    # cnn.save('low_cnn.keras')

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

    val_loss, val_accuracy = model.evaluate(X_val, y_val)

    print("unaugmented val acc: " + str(val_accuracy))

    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    # Save the model
    model.save('low_res_image_classifier.h5')




low()
high()