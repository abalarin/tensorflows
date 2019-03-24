# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os

# Maximum Amount of word we'll take into account
max_vocabulary = 10000

# Create Save Model Callback
# checkpoint_path = "models/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # Create checkpoint callback
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1)


def extract_data(filename):
    data = pd.read_excel(filename)

    features = [row[2] for row in data.values.astype('U')]
    labels = [row[3] for row in data.values]

    print("Normal Text")
    print(features[0])
    for i, text in enumerate(features):
        features[i] = keras.preprocessing.text.one_hot(text, max_vocabulary, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

    print("One Hot: ")
    print(features[0])

    return features, labels


def create_model():
    model = keras.Sequential([
        keras.layers.Embedding(max_vocabulary, 32, input_length=250),
        keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(250, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())

    return model


x_train, y_train = extract_data('./temp/datasets/dataset_train.xlsx')
x_test, y_test = extract_data('./temp/dataset_test.xlsx')

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

print("Padding:")
print(x_train[0])
print(x_test[0])

model = create_model()

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=2)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
