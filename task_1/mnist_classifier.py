from interface import MnistClassifierInterface

from random_forest import RandomForest
from neural_networks import FeedForwardNeuralNetworks
from convolutional_neural_networks import ConvolutionalNeuralNetwork


class MnistClassifier(MnistClassifierInterface):

    MODELS = {
        "rf": RandomForest,
        "nn": FeedForwardNeuralNetworks,
        "cnn": ConvolutionalNeuralNetwork
    }
    def __init__(self, algorithm):
        self.model = self.MODELS[algorithm]()



    def train(self, x,y):
        return self.model.train(x,y)

    def predict(self, x):
        return self.model.predict(x)
