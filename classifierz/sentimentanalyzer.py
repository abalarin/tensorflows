# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

max_words = 10000

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_words)

print(x_train[0])

X = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# summarize size
print("Training data: ")
print(X.shape)
print(y.shape)

# Summarize number of classes
print("Classes: ")
print(np.unique(y))

# Summarize number of words
print("Number of words: ")
print(len(np.unique(np.hstack(X))))

print("Review lenght: ")
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))
# plot review length
# plt.boxplot(result)
# plt.show()

seed = 7
np.random.seed(seed)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=500)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=500)
print(x_train[0])

model = keras.Sequential([
    keras.layers.Embedding(max_words, 32, input_length=500),
    keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=2)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
