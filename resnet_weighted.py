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
from sklearn.utils.class_weight import compute_class_weight


y, train, val, test = util.get_labels_and_split() # Can use anything here

y_train, y_val, y_test = y[train], y[val], y[test]

y_keep = y_train

y_train = tf.one_hot(y_train, depth=8)
y_val   = tf.one_hot(y_val,   depth=8)
y_test  = tf.one_hot(y_test,  depth=8)

low_res = util.get_low_res() # Match abvoe
# low_res = np.load('low_res.npy')

# low_res.reshape(low_res.shape[0], -1)

X_train = low_res[train]
X_val = low_res[val]
X_test = low_res[test]

print(X_train.shape)

base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(147, 147, 3))

for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))

model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

class_weights = compute_class_weight('balanced', classes=range(8), y=np.array(y_keep))

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32, class_weight=dict(enumerate(class_weights)))

print("predicting")

val_loss, val_accuracy = model.evaluate(X_val, y_val)

print("val acc: " + str(val_accuracy))

# test_loss, test_accuracy = cnn.evaluate(X_test, y_test)

# print("test acc: " + str(test_accuracy))


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

print("unaugmented test acc: " + str(test_accuracy))



