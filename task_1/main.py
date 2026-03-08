from tensorflow.keras.datasets import mnist
from mnist_classifier import MnistClassifier
from sklearn.metrics import accuracy_score


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    classifier = MnistClassifier("cnn")   #here can be trained rf / nn / cnn
    classifier.train(x_train, y_train)
    predictions = classifier.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()
