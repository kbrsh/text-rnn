import numpy as np

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

class RNN(object):
    def __init__(self):
        # Hyperparameters

        # Number of Hidden Layers in Network
        self.hiddenLayers = 100

        # Learning Rate
        self.learning_rate = 1e-0

        # Weights for Hidden Layer to Hidden Layer
        self.WH = np.random.randn(self.hiddenLayers, self.hiddenLayers)

        # Initial Hidden State
        self.h = {}
        self.h[-1] = np.zeros((self.hiddenLayers, 1))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def loss(self, prediction, targets):
        return -np.log(prediction[targets, 0])

    def forward_step(self, X, i):
        # Compute hidden state
        self.h[i] = np.tanh(np.dot(self.WX, X) + np.dot(self.WH, self.h[i - 1]))
        # Compute output
        return self.softmax(np.dot(self.WY, self.h[i]))

    def step(self):
        hot_inputs = []
        outputs = []
        target_idxs = []
        loss = 0

        # Forward propagation
        for i in xrange(self.T):
            # Encode In One hot Encoding
            hot_input = np.zeros((self.vocab_size, 1))
            hot_input[self.gram_to_vocab[self.data[i]]][0] = 1
            hot_inputs.append(hot_input)

            hot_output = np.zeros((self.vocab_size, 1))
            target_idx = self.gram_to_vocab[self.data[i + 1]]
            hot_output[target_idx][0] = 1
            target_idxs.append(target_idx)

            # Make Prediction (Compute Hidden State)
            prediction = self.forward_step(hot_input, i)
            outputs.append(prediction)

            # Compute Loss
            output_loss = self.loss(prediction, target_idxs[i])
            loss += output_loss


        # Back Propagation Through Time (BPTT)
        dW = np.zeros_like(self.WX)
        dW2 = np.zeros_like(self.WY)
        dWh = np.zeros_like(self.WH)
        for i in reversed(xrange(self.T)):
            # Compute gradient for outputs
            d_output = np.copy(outputs[i])
            d_output[target_idxs[i]] -= 1

            # Compute Gradient for Hidden to Output
            dW2 += np.dot(d_output, self.h[i].T)

            # Compute Gradient for Hidden to Hidden
            dhidden = np.dot(self.WY.T, d_output)
            dhidden = (1 - self.h[i] * self.h[i]) * dhidden
            dWh += np.dot(dhidden, self.h[i - 1].T)

            # Compute Gradient for Hidden to Output
            dW += np.dot(dhidden, hot_inputs[i].T)

        # Perform Parameter Update
        self.WX += -self.learning_rate * dW
        self.WY += -self.learning_rate * dW2
        self.WH += -self.learning_rate * dWh

        return loss

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
        data_len = len(self.data) - 1
        self.T = data_len

        # Initialize Weights
        self.WX = np.random.randn(self.hiddenLayers, self.vocab_size) # Input to Hidden
        self.WY = np.random.randn(self.vocab_size, self.hiddenLayers) # Hidden to Output


    def sample(self, n=100):
        # Sample
        sample = ""

        # Seed (start letter)
        seed = 0

        # Generate sample input
        sample_input = np.zeros((self.vocab_size, 1))
        sample_input[seed, 0] = 1

        # Index of current hidden state
        h_idx = 0

        sample += self.vocab[seed]

        for i in range(n):
            # Move Inputs forward and Get Output
            sample_output = self.forward_step(sample_input, h_idx)
            vocab_idx = np.argmax(sample_output)
            sample_input = np.zeros((self.vocab_size, 1))
            sample_input[vocab_idx, 0] = 1
            sample += self.vocab[vocab_idx]
            h_idx += 1

        return sample

iterations = 1000
bot = RNN()
bot.train(open("data.txt").read())
for i in xrange(iterations):
    loss = bot.step()
    if(i % 100 == 0):
        print '======= Iteration: ' + str(i) + '  Loss: ' + str(loss) + ' ======='
        print bot.sample()
