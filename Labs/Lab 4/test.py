"""
Porsev Aslan
porsev@kth.se
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Exercise 1 ------------------

def read_book_data(file_path):
    with open(file_path, 'r') as f:
        # Read the text file
        book_data = f.read()
    return book_data

def get_unique_characters(book_data):
    book_chars = sorted(set(book_data))
    K = len(book_chars)
    return book_chars, K

def create_char_mappings(book_chars):
    char_to_ind = {char: i for i, char in enumerate(book_chars)}
    ind_to_char = {i: char for i, char in enumerate(book_chars)}
    return char_to_ind, ind_to_char

def chars_to_one_hot(char_seq, char_to_ind, K):
    N = len(char_seq)
    one_hot_seq = np.zeros((K, N))
    for i, char in enumerate(char_seq):
        one_hot_seq[char_to_ind[char], i] = 1
    return one_hot_seq

def one_hot_to_chars(one_hot_seq, ind_to_char):
    N = one_hot_seq.shape[1]
    char_seq = ''.join([ind_to_char[np.argmax(one_hot_seq[:, i])] for i in range(N)])
    return char_seq

# Read the data from the file
book_data = read_book_data('goblet_book.txt')

# Get unique characters and their count
book_chars, K = get_unique_characters(book_data)

# Create character mappings
char_to_ind, ind_to_char = create_char_mappings(book_chars)

# Test encoding and decoding
test_char_seq = 'Hello, world!'
test_one_hot_seq = chars_to_one_hot(test_char_seq, char_to_ind, K)
test_reconstructed_char_seq = one_hot_to_chars(test_one_hot_seq, ind_to_char)

# Assert that the reconstruction is correct (prints True if correct)
print(test_char_seq == test_reconstructed_char_seq)


# ------------------ Exercise 2 ------------------
class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1, seq_length=25, sigma=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        
        self.b = np.zeros((hidden_dim, 1))
        self.c = np.zeros((output_dim, 1))
        
        self.U = np.random.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
        self.W = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.V = np.random.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)

# Set hyperparameters
m = 100
eta = 0.1
seq_length = 25

# Initialize the RNN
rnn = RNN(K, m, K, eta, seq_length)

# ------------------ Exercise 3 ------------------
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def synthesize_text(rnn, h0, x0, n):
    """ Synthesize text from the RNN model.

    Args:
        rnn: RNN model
        h0: Initial hidden state of shape (m, 1)
        x0: One-hot encoding of initial character of shape (K, 1)
        n: Number of characters to synthesize

    Returns:
        Y: One-hot encodings of synthesized characters of shape (K, n)"""
    # Initialize hidden state
    h = h0
    # Initialize one-hot encoding of first character
    x = x0
    # Initialize synthesized text
    Y = np.zeros((rnn.output_dim, n))

    for t in range(n):
        # Compute the next hidden state
        h = np.tanh(np.dot(rnn.U, x) + np.dot(rnn.W, h) + rnn.b)
        # Compute output
        o = np.dot(rnn.V, h) + rnn.c
        # Compute probabilities
        p = softmax(o)
        # Sample the next character
        x_next = np.random.choice(rnn.output_dim, p=p.ravel())
        # Convert to one-hot encoding
        x = np.zeros((rnn.output_dim, 1))
        x[x_next] = 1
        # Store result
        Y[:, t] = x.ravel()

    return Y

# Test synthesize_text
h0 = np.zeros((m, 1))
x0 = chars_to_one_hot('A', char_to_ind, K)
n = 100

Y = synthesize_text(rnn, h0, x0, n)
synthesized_text = one_hot_to_chars(Y, ind_to_char)
print(synthesized_text)

# ------------------ Exercise 4 ------------------
#forwards pass
def forward_pass(rnn, X, Y, h0):
    """ Forward pass for the RNN model.
    
    Args:
        rnn: RNN model
        X: Input sequence of shape (K, N)
        Y: Output sequence of shape (K, N)
        h0: Initial hidden state of shape (m, 1)
        
        Returns:
        loss: Loss of the model
        h : Hidden states of shape (m, N)
        p : Probabilities of shape (K, N)"""

    # Initialize loss
    loss = 0
    # Initialize list of probabilities
    p = {}
    # Initialize list of hidden states
    h = {}
    # Initialize first hidden state
    h[-1] = np.copy(h0)
    # Initialize list of outputs
    o = {}

    # Loop over sequence
    for t in range(rnn.seq_length):
        # Compute the next hidden state
        h[t] = np.tanh(np.dot(rnn.U, X[:, t:t+1]) + np.dot(rnn.W, h[t-1]) + rnn.b)
        # Compute output
        o[t] = np.dot(rnn.V, h[t]) + rnn.c
        # Compute probabilities
        p[t] = softmax(o[t])
        # Compute loss and get the argmax
        loss += -np.log(p[t][Y[:, t].argmax(), 0])

    return loss, h, p

#backwards pass
def backward_pass(rnn_model, X, Y, hidden_state_seq, prob_seq):
    """ Backward pass for the RNN model.

    Args:
        rnn_model: RNN model  
        X: Input sequence of shape (K, N)
        Y: Output sequence of shape (K, N)
        hidden_state_seq: Hidden state sequence of shape (m, N)
        prob_seq: Probability sequence of shape (K, N)

    Returns:
        gradient_collection: Dictionary containing gradients of the model parameters
        """
    gradient_collection = {
        'U': np.zeros(rnn_model.U.shape),
        'W': np.zeros(rnn_model.W.shape),
        'V': np.zeros(rnn_model.V.shape),
        'b': np.zeros(rnn_model.b.shape),
        'c': np.zeros(rnn_model.c.shape)
    }

    next_hidden_state_gradient = np.zeros(hidden_state_seq[0].shape)
    seq_len = X.shape[1]

    for seq_position in reversed(range(seq_len)):
        grad_o = np.copy(prob_seq[seq_position])
        grad_o[Y[:, seq_position].argmax()] -= 1

        gradient_collection['V'] += np.dot(grad_o, hidden_state_seq[seq_position].T)
        gradient_collection['c'] += grad_o

        grad_h = np.dot(rnn_model.V.T, grad_o) + next_hidden_state_gradient
        grad_a = grad_h * (1 - np.square(hidden_state_seq[seq_position]))

        gradient_collection['U'] += np.dot(grad_a, X[:, seq_position:seq_position+1].T)
        gradient_collection['W'] += np.dot(grad_a, hidden_state_seq[seq_position-1].T) if seq_position != 0 else np.zeros_like(rnn_model.W)
        gradient_collection['b'] += grad_a

        next_hidden_state_gradient = np.dot(rnn_model.W.T, grad_a)

    return gradient_collection

def gradient_clip(grads, max_value=5):
    """Clips gradients to improve training stability"""
    for grad in grads:
        grads[grad] = np.clip(grads[grad], -max_value, max_value)
    return grads



print("Rnn seq length: ", rnn.seq_length)
X_chars = book_data[:rnn.seq_length]
Y_chars = book_data[1:rnn.seq_length+1]

X = chars_to_one_hot(X_chars, char_to_ind, K)
Y = chars_to_one_hot(Y_chars, char_to_ind, K)

loss, h, p = forward_pass(rnn, X, Y, h0)
print('loss: ', loss)

# Compute numerical gradient
def compute_numerical_gradient(rnn, X, Y, h0, parameter_name, h=1e-5):
    """
    Computes the numerical gradient of the loss function with respect to a parameter.
    
    Arguments:
    rnn -- the RNN
    X -- the input data
    Y -- the output data
    h0 -- the initial hidden state
    parameter_name -- the name of the parameter
    h -- the value of the perturbation

    Returns:
    grad -- the numerical gradient
    """

    original_value = getattr(rnn, parameter_name)
    grad = np.zeros_like(original_value)
    it = np.nditer(original_value, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        # Add h to the current parameter element
        original_value[ix] += h
        setattr(rnn, parameter_name, original_value)
        loss2, _, _ = forward_pass(rnn, X, Y, h0)
        
        # Subtract h from the current parameter element
        original_value[ix] -= 2 * h
        setattr(rnn, parameter_name, original_value)
        loss1, _, _ = forward_pass(rnn, X, Y, h0)
        
        # Compute the gradient
        grad[ix] = (loss2 - loss1) / (2 * h)
        
        # Reset the parameter to its original value
        original_value[ix] += h
        setattr(rnn, parameter_name, original_value)
        it.iternext()

    return grad

def gradient_check(rnn, X, Y, h0, computed_grads):
    """
    Computes the numerical gradient and compares it with the backprop gradient

    Arguments:
    rnn -- the RNN
    X -- the input data
    Y -- the output data
    h0 -- the initial hidden state
    computed_grads -- the backprop gradient
    """
    parameter_names = ['U', 'W', 'V', 'b', 'c']
    for parameter_name in parameter_names:
        computed_grad = computed_grads[parameter_name]
        numerical_grad = compute_numerical_gradient(rnn, X, Y, h0, parameter_name)
        
        # Compute relative error
        epsilon = 1e-8
        relative_error = np.abs(computed_grad - numerical_grad) / (np.abs(computed_grad) + np.abs(numerical_grad) + epsilon)
        
        print('Max relative error for {}: {}'.format(parameter_name, np.max(relative_error)))

# Compute the gradients
computed_grads = backward_pass(rnn, X, Y, h, p)
computed_grads = gradient_clip(computed_grads)

# Check the gradients
gradient_check(rnn, X, Y, h0, computed_grads)

# ------------------ Exercise 5 ------------------
def plot(losses):
    plt.plot(losses)
    plt.xlabel('Update Step')
    plt.ylabel('Smooth Loss')
    plt.title('Smooth Loss vs. Update Step')
    plt.show()

def rnn_training(model, text_data, char2idx, idx2char, epochs=4):
    """
    Trains the RNN model on the given text data
    
    Arguments:
    model -- the RNN model
    text_data -- the book text data
    char2idx -- a dictionary mapping characters to indices
    idx2char -- a dictionary mapping indices to characters
    epochs -- the number of epochs to train for
    
    Returns:
    loss_history -- a list containing the loss at each update step
    """
    out_dim = model.output_dim
    seq_len = model.seq_length

    iterations_per_epoch = len(text_data) // seq_len
    total_updates = epochs * iterations_per_epoch

    avg_loss = -np.log(1.0 / out_dim) * seq_len
    prev_hidden_state = np.zeros((model.hidden_dim, 1))
    ada_grads = {param: np.zeros_like(getattr(model, param)) for param in ['U', 'W', 'V']}

    pos = 0
    loss_history = []
    #Uncomment these lines to save the best model
    #best_rnn = None
    #lowest_smooth_loss = float('inf')
    for update in range(total_updates):
        if pos == 0 or pos + seq_len + 1 > len(text_data):
            pos = 1
            prev_hidden_state = np.zeros((model.hidden_dim, 1))

        input_chars = text_data[pos - 1 : pos - 1 + seq_len]
        target_chars = text_data[pos : pos + seq_len]

        input_data = chars_to_one_hot(input_chars, char2idx, out_dim)
        target_data = chars_to_one_hot(target_chars, char2idx, out_dim)

        loss_val, hidden_state, prob = forward_pass(model, input_data, target_data, prev_hidden_state)
        grad_values = backward_pass(model, input_data, target_data, hidden_state, prob)
        grad_values = gradient_clip(grad_values)

        for param in ada_grads.keys():
            ada_grads[param] += grad_values[param] * grad_values[param]
            new_param_value = getattr(model, param) - model.learning_rate * grad_values[param] / (np.sqrt(ada_grads[param]) + 1e-8)
            setattr(model, param, new_param_value)

        avg_loss = 0.999 * avg_loss + 0.001 * loss_val
        loss_history.append(avg_loss)

        if update % 10000 == 0:
            print(f"Average loss at update {update}: {avg_loss}")

        if update % 100000 == 0:
            synthesized_output = synthesize_text(model, prev_hidden_state, input_data[:, :1], 200)
            synthesized_text = one_hot_to_chars(synthesized_output, idx2char)
            print(f"Generated text:\n {synthesized_text}")


        # if loss_history < lowest_smooth_loss:
        #     lowest_smooth_loss = loss_history
        #     best_rnn = rnn
        pos += seq_len

    return loss_history

# Call the train_rnn function to train the RNN
smooth_losses = rnn_training(rnn, book_data, char_to_ind, ind_to_char, epochs=7)
#plot(smooth_losses)


# Call the train_rnn function to train the RNN
#smooth_losses, best_rnn = rnn_training2(rnn, book_data, char_to_ind, ind_to_char, n_epochs=7)
plot(smooth_losses)

# Generate a sequence of 1000 characters
h0 = np.zeros((m, 1)) # reset the initial hidden state
x0 = chars_to_one_hot('A', char_to_ind, K) # let's start with the character 'A'
n = 1000

# Synthesize the text
Y = synthesize_text(rnn, h0, x0, n)
synthesized_text = one_hot_to_chars(Y, ind_to_char)
print(synthesized_text)