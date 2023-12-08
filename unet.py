import util
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam

y, train, val, test = util.get_labels_and_split_augmented() # Can use anything here

y_train, y_val, y_test = y[train], y[val], y[test]

y_train = tf.one_hot(y_train, depth=8)
new_y_train = np.tile(y_train[:, np.newaxis, np.newaxis, :], (1, 144, 144, 1))
y_val   = tf.one_hot(y_val,   depth=8)
new_y_val = np.tile(y_val[:, np.newaxis, np.newaxis, :], (1, 144, 144, 1))
y_test  = tf.one_hot(y_test,  depth=8)
new_y_test = np.tile(y_test[:, np.newaxis, np.newaxis, :], (1, 144, 144, 1))

low_res = util.get_low_res_augmented()
# low_res = np.load('low_res.npy')

# low_res.reshape(low_res.shape[0], -1)

X_train = low_res[train]
X_val = low_res[val]
X_test = low_res[test]

print(X_train.shape)

def unet_model(input_shape=(147, 147, 3), num_classes=8):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Middle
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    crop6 = layers.Cropping2D(cropping=((0, 0), (0, 0)))(conv4)  # Adjust cropping as needed
    concat6 = layers.Concatenate()([up6, crop6])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(concat6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    crop7 = layers.Cropping2D(cropping=((0, 0), (0, 0)))(conv3)  # Adjust cropping as needed
    concat7 = layers.Concatenate()([up7, crop7])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    crop8 = layers.Cropping2D(cropping=((1, 0), (1, 0)))(conv2)  # Adjust cropping as needed
    concat8 = layers.Concatenate()([up8, crop8])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    crop9 = layers.Cropping2D(cropping=((3, 0), (3, 0)))(conv1)  # Adjust cropping as needed
    concat9 = layers.Concatenate()([up9, crop9])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # dense10 = layers.Dense(8, activation='softmax')(conv9)

    # outputs = layers.Dense(8, activation='softmax')(dense10)

    outputs = layers.Conv2D(8, 1, activation='sigmoid')(conv9)

    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


model = unet_model()


model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, new_y_train, validation_data=(X_val, new_y_val), epochs=1, batch_size=32)

print("predicting")

val_loss, val_accuracy = model.evaluate(X_val, new_y_val)

print("val acc: " + str(val_accuracy))

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

preds = model.predict(X_val)
np.save("preds.npy", preds)

comb_preds = preds.sum(axis=(1,2))
comb_guesses = comb_preds.argmax(axis=1)

real_values = np.array(y_val).argmax(axis=1)

print("unaugmented val acc: " + str((real_values == comb_guesses).mean()))


# Save the model
