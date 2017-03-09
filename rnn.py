import numpy as np

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

class RNN(object):
    def __init__(self):
        # Hyperparameters
        self.hiddenLayers = 100

        self.WH = np.random.randn(self.hiddenLayers, self.hiddenLayers)

        self.h = {}
        self.h[-1] = np.zeros((self.hiddenLayers))

        self.o = {}

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def forward_step(self, X, i):
        # Compute hidden state
        self.h[i] = np.tanh(np.dot(X, self.WX) + np.dot(self.h[i - 1], self.WH))
        # Compute output
        return self.softmax(np.dot(self.h[i], self.WY))

    def step(self):
        outputs = []
        for i in xrange(self.T):
            X = self.data[i]
            normalized_input = np.zeros(self.vocab_size)
            normalized_input[self.gram_to_vocab[X]] = 1

            normalized_output = np.zeros(self.vocab_size)
            normalized_output[self.gram_to_vocab[X] + 1] = 1

            outputs.append(self.forward_step(normalized_input, i))

    def train(self, data, ngrams=7):
        # Split Data by ngrams
        self.data = [data[i:i+ngrams] for i in range(0, len(data), ngrams)]

        # Generate map to access gram from index
        self.vocab = {i:gram for i,gram in enumerate(unique(self.data))}

        # Generate map to access index from gram
        self.gram_to_vocab = {gram:i for i,gram in enumerate(unique(self.data))}

        # Total Vocabulary Size
        self.vocab_size = len(self.vocab)

        # Timesteps to Move through Network
        self.T = len(self.data) - 1

        # Initialize Weights
        self.WX = np.random.randn(self.vocab_size, self.hiddenLayers) # Input to Hidden
        self.WY = np.random.randn(self.hiddenLayers, self.vocab_size) # Hidden to Output


    def sample(self, n=100):
        # Sample
        sample = ""

        # Seed (start letter)
        seed = 0

        # Generate sample input
        sample_input = np.zeros((self.vocab_size))
        sample_input[0] = 1

        # Index of current hidden state
        h_idx = 0

        # Reset Hidden State
        self.h = {}
        self.h[-1] = np.zeros((self.hiddenLayers))

        sample += self.vocab[0]

        for i in range(n):
            # Move Inputs forward and Get Output
            sample_output = self.forward_step(sample_input, h_idx)
            vocab_idx = np.argmax(sample_output)
            sample_input = np.zeros((self.vocab_size))
            sample_input[vocab_idx] = 1
            sample += self.vocab[vocab_idx]
            h_idx += 1

        return sample

epochs = 1
bot = RNN()
bot.train(open("data.txt").read())
for i in xrange(epochs):
    if(i % 100 == 0):
        print bot.sample(1000)
