from interface import MnistClassifierInterface
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


class FeedForwardNeuralNetworks(MnistClassifierInterface):

    def train(self, X_train, y_train):

        X = X_train.reshape(len(X_train), 784)
        X = X / 255.0

        self.model = Sequential([
            Dense(128, activation="relu", input_shape=(784,)),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model.fit(X, y_train, epochs=5, batch_size=32)

    def predict(self, X_test):

        X = X_test.reshape(len(X_test), 784)
        X = X / 255.0

        preds = self.model.predict(X)

        return np.argmax(preds, axis=1)