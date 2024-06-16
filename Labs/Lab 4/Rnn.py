import numpy as np
"""
Simple code to implement a RNN and code taken from lab3
"""
class RNN:
    def __init__(self, k=1, m=100, learning_rate=0.1, seq_length=25, sigma=0.01):
        # Setting up the parameters
        self.M = m
        self.eta = learning_rate
        self.seq_length = seq_length
        # Bias vector B and C and weight matrices U, W and V
        self.B = np.zeros((m, 1))
        self.C = np.zeros((k, 1))  # Ensure that grads.C has the correct shape
        self.U = np.random.rand(m, k) * sigma
        self.W = np.random.rand(m, m) * sigma
        self.V = np.random.rand(k, m) * sigma


# softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# Loss function
def compute_loss(target, probs):
    """
    target: target values
    probs: probabilities
    Returns the loss
    """
    return -np.sum(np.log(np.sum(target * probs, axis=0)))


# One-hot encoding
def one_hot(vec, conversor):
    """
    vec: vector to convert
    conversor: dictionary to convert
    Returns the one-hot encoding of the vector
    """
    one_hotter = np.zeros((len(conversor), len(vec)))
    for i in range(len(vec)):
        one_hotter[conversor[vec[i]], i] = 1
    return one_hotter


# Gets the data from the text file
def get_data():
    file = open("goblet_book.txt", "r")
    all_text = file.read()
    unique_chars = set(all_text)
    char_to_int = {char: i for i, char in enumerate(unique_chars)}
    int_to_char = {i: char for i, char in enumerate(unique_chars)}
    return all_text, char_to_int, int_to_char
