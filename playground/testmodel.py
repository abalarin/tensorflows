import tensorflow as tf
from tensorflow import keras
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

max_words = 10000


def extract_data(filename):
    data = pd.read_excel(filename)

    features = [row[2] for row in data.values.astype('U')]
    labels = [row[3] for row in data.values]

    return features, labels


x_test, y_test = extract_data('./temp/dataset3.xlsx')

for i, text in enumerate(x_test):
    x_test[i] = keras.preprocessing.text.one_hot(text, max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=500)

model = keras.Sequential([
    keras.layers.Embedding(max_words, 32, input_length=500),
    keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

loss, acc = model.evaluate(x_test, y_test)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

model.load_weights("./models/cp.ckpt")
loss, acc = model.evaluate(x_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
