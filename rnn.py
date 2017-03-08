import numpy as np

class RNN(object):
    def __init__(self):
        self.hiddenLayers = 5

        self.WH = np.random.randn(self.hiddenLayers, self.hiddenLayers)

        self.h = {}
        self.h[-1] = np.zeros(self.hiddenLayers)

        self.o = {}

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def loss(self, prediction, target):
        return -np.log(prediction[np.argmax(target)])

    def forward(self, X, oh):
        # Update hidden layer
        h = self.softmax(np.dot(X, self.WX) + np.dot(oh, self.WH))
        # Activity of Output Layer
        y = self.softmax(np.dot(h, self.WY))
        return h, y

    def step(self, X):
        err = 0

        # Forward propogation for all inputs
        for i in range(self.T):
            self.h[i], self.o[i] = self.forward(X[i], self.h[i - 1])
            err += self.loss(self.o[i], self.y[i])

        return self.o, self.h, err

    def train(self, inputs, outputs):
        self.X = np.array(inputs, dtype=float)
        self.y = np.array(outputs, dtype=float)

        self.inputLayers = len(self.X[0])
        self.outputLayers = len(self.y[0])

        self.WX = np.random.randn(self.inputLayers, self.hiddenLayers)
        self.WY = np.random.randn(self.hiddenLayers, self.outputLayers)

        self.T = len(self.X)
