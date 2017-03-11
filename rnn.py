import numpy as np

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

class RNN(object):
    def __init__(self):
        # Hyperparameters

        # Number of Hidden Layers in Network
        self.hiddenLayers = 75

        # Learning Rate
        self.learning_rate = 1e-3

        # Weights for Hidden Layer to Hidden Layer
        self.WH = np.random.randn(self.hiddenLayers, self.hiddenLayers)

        # Initial Hidden State
        self.h = {}
        self.h[-1] = np.zeros((self.hiddenLayers, 1))

        # Initial Hidden State for Samples
        self.sample_h = np.zeros((self.hiddenLayers, 1))

        # Internal Cursor
        self.cursor = 0

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def loss(self, prediction, targets):
        return -np.log(prediction[targets, 0])

    def forward_step(self, X, prev_h):
        # Compute hidden state
        h = np.tanh(np.dot(self.WX, X) + np.dot(self.WH, prev_h))
        # Compute output
        return self.softmax(np.dot(self.WY, prev_h)), h

    def step(self):
        loss = 0
        return loss

    def train(self, data, ngrams=7):
        # Split Data by ngrams
        self.data = [data[i:i+ngrams] for i in range(0, len(data), ngrams)]

        # Get Unique Data
        self.unique_data = unique(self.data)

        # Generate map to access gram from index
        self.vocab = {i:gram for i,gram in enumerate(self.unique_data)}

        # Generate map to access index from gram
        self.gram_to_vocab = {gram:i for i,gram in enumerate(self.unique_data)}

        # Total Vocabulary Size
        self.vocab_size = len(self.vocab)

        # Timesteps to Move through Network
        data_len = len(self.data) - 1
        self.T = data_len if data_len < 10 else 10

        # Initialize Weights
        self.WX = np.random.randn(self.hiddenLayers, self.vocab_size) # Input to Hidden
        self.WY = np.random.randn(self.vocab_size, self.hiddenLayers) # Hidden to Output


    def sample(self, n=100):
        # Sample
        sample = ""

        # Seed (Start Letter)
        seed = self.gram_to_vocab[self.data[self.cursor]]

        # Populate Sample with Seed
        sample += self.data[self.cursor]

        # Generate sample input
        sample_input = np.zeros((self.vocab_size, 1))
        sample_input[seed, 0] = 1

        for i in xrange(n):
            # Move Inputs Through Neural Network
            sample_output, self.sample_h = self.forward_step(sample_input, self.sample_h)
            idx = np.argmax(sample_output)
            sample += self.vocab[idx]

            # Generate new Inputs
            sample_input = np.zeros((self.vocab_size, 1))
            sample_input[idx] = 1

        return sample

iterations = 1000
bot = RNN()
bot.train(open("data.txt").read())
for i in xrange(iterations):
    loss = bot.step()
    if(i % 100 == 0):
        print '======= Iteration: ' + str(i) + '  Loss: ' + str(loss) + ' ======='
        print bot.sample()
