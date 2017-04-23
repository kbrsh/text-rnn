import numpy as np
import pickle

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def softmax(x, temperature=1.0):
    exp_x = np.exp(x / temperature)
    return exp_x / np.sum(exp_x)

class TextRNN(object):
    def __init__(self, hiddenLayers=10, sequenceLength=100):
        # Hidden Layers
        self.hiddenLayers = hiddenLayers

        # Learning Rate
        self.learningRate = 1e-2

        # Hidden State
        self.h = {}

        # Internal cursor
        self.cursor = 0

        # Sequence Length
        self.sequenceLength = sequenceLength

    def train(self, text, ngrams=7, delimiter=" "):
        # Setup delimiter
        self.delimiter = delimiter

        # Split by delimiter
        grams = text.split(delimiter) if delimiter != "" else list(text)

        # Setup Data by Ngrams
        self.data = [delimiter.join(grams[i:i+ngrams]) for i in range(len(grams))[::ngrams]]

        # Get Unique Data
        self.uniqueData = unique(self.data)

        # Get Vocab Maps
        self.indexToGram = {i:gram for i, gram in enumerate(self.uniqueData)}
        self.gramToIndex = {gram:i for i, gram in enumerate(self.uniqueData)}

        # Get vocab size
        self.vocabSize = len(self.uniqueData)

        # Setup Inputs
        inputs = []
        outputs = []
        inputGrams = [self.gramToIndex[gram] for gram in self.data]
        outputGrams = [self.gramToIndex[gram] for gram in self.data[1:]]

        for i, inputGram in enumerate(inputGrams[0:-1]):
            X = np.zeros((self.vocabSize, 1))
            X[inputGram, 0] = 1

            y = np.zeros((self.vocabSize, 1))
            y[outputGrams[i], 0] = 1

            inputs.append(X)
            outputs.append(y)

        self.inputs = inputs
        self.outputs = outputs

        # Setup Weights
        self.WX = np.random.randn(self.hiddenLayers, self.vocabSize) * 0.1
        self.WH = np.random.randn(self.hiddenLayers, self.hiddenLayers) * 0.1
        self.WY = np.random.randn(self.vocabSize, self.hiddenLayers) * 0.1

        # Setup Gradient Cache for Adam Update
        self.dXM = np.zeros_like(self.WX)
        self.dXV = np.zeros_like(self.WX)

        self.dHM = np.zeros_like(self.WH)
        self.dHV = np.zeros_like(self.WH)

        self.dYM = np.zeros_like(self.WY)
        self.dYV = np.zeros_like(self.WY)

    def forward(self, X, hPrev, temperature=1.0):
        h = np.tanh(np.dot(self.WX, X) + np.dot(self.WH, hPrev))
        o = softmax(np.dot(self.WY, h), temperature)
        return h, o

    def step(self):
        # Hidden State
        self.h = {}
        self.h[-1] = np.zeros((self.hiddenLayers, 1))

        # Outputs
        o = {}

        # Target Indexes
        targets = {}

        # Timesteps to Unroll
        totalLen = self.vocabSize - 1
        if self.cursor + self.sequenceLength > totalLen:
            self.cursor = 0

        # Total Loss
        loss = 0

        for i in xrange(self.sequenceLength):
            # Get inputs and outputs
            X = self.inputs[self.cursor + i]
            y = self.outputs[self.cursor + i]

            # Move inputs forward through network
            self.h[i], o[i] = self.forward(X, self.h[i - 1])

            # Calculate loss
            target = np.argmax(y)
            loss += -np.log(o[i][target, 0])
            targets[i] = target

        # Back Propagation
        dX = np.zeros_like(self.WX)
        dH = np.zeros_like(self.WH)
        dY = np.zeros_like(self.WY)
        for i in reversed(xrange(self.sequenceLength)):
            dSY = np.copy(o[i])
            dSY[targets[i]] -= 1
            dY += np.dot(dSY, self.h[i].T)

            dHRaw = (1 - self.h[i] * self.h[i]) * np.dot(self.WY.T, dSY)

            dH += np.dot(dHRaw, self.h[i-1].T)

            dX += np.dot(dHRaw, self.inputs[i].T)

        # Parameter Update (Adam)
        for param, delta, m, v in zip([self.WX,  self.WH,  self.WY],
                                      [dX,       dH,       dY],
                                      [self.dXM, self.dHM, self.dYM],
                                      [self.dXV, self.dHV, self.dYV]):
            m = 0.9 * m + 0.1 * delta
            v = 0.99 * v + 0.01 * (delta ** 2)
            param += -self.learningRate * m / (np.sqrt(v) + 1e-8)

        # Update cursor
        self.cursor += self.sequenceLength

        return loss


    def sample(self, num=100, temperature=1.0):
        # Output
        output = ""

        # Sample hidden state
        h = {}
        h[-1] = np.zeros((self.hiddenLayers, 1))

        # Make inputs from seed
        X = np.zeros((self.vocabSize, 1))
        X[self.cursor, 0] = 1

        # Add seed to output
        output += self.indexToGram[self.cursor]

        # Generate sample
        for i in xrange(num - 1):
            # Move through network
            h[i], prediction = self.forward(X, h[i - 1], temperature)

            # Pick ngram using probabilities
            idx = np.random.choice(range(self.vocabSize), p=prediction.ravel())

            # Add to output
            output += self.delimiter + self.indexToGram[idx]

            # Update input to feed back in
            X = np.zeros((self.vocabSize, 1))
            X[idx, 0] = 1


        return output

    def run(self, epochs=10, iterations=100, size=100, temperatures=[1.0]):
        for i in xrange(epochs):
            for j in xrange(iterations):
                loss = bot.step()
            for temperature in temperatures:
                print '======= Temperature: ' + str(temperature) + ' ======='
                print bot.sample(size, temperature)

            print '======= Epoch ' + str(i + 1) + ' ======='
            print '======= Loss: ' + str(loss) + ' ======='
            print '\n'

    def save(self):
        pickle.dump(self, open("TEXT_RNN_DUMP", "w+"))

    def load(self, dump):
        return pickle.load(dump)


bot = TextRNN()
bot.train(open('data.txt').read(), 1)
bot.run(epochs=100, iterations=10, size=25, temperatures=[0.7, 1.0])
bot.save()
