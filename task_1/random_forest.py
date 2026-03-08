from interface import *
from sklearn.ensemble import RandomForestClassifier

class RandomForest(MnistClassifierInterface):

    def train(self,X_train, y_train):
        X = X_train.reshape(len(X_train), -1)
        self.model = RandomForestClassifier()
        self.model.fit(X, y_train)

    def predict(self, X_test):
        X = X_test.reshape(len(X_test), -1)
        return self.model.predict(X)
