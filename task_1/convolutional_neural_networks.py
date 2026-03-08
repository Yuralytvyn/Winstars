from interface import MnistClassifierInterface
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np


class ConvolutionalNeuralNetwork(MnistClassifierInterface):

    def train(self, X_train, y_train):
        X = X_train.reshape(len(X_train), 28, 28, 1)
        X = X / 255.0

        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model.fit(X, y_train, epochs=5, batch_size=32)

    def predict(self, X_test):
        X = X_test.reshape(len(X_test), 28, 28, 1)
        X = X / 255.0

        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)