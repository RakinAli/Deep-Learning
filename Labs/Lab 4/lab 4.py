"""
Assignment 4 at DD2424 Deep Learning in Data Science by @RakinAli
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
import copy

import math


class RNN:
    def __init__(self, k=1, m=100, learning_rate=0.1, seq_length=25, sigma=0.01):
        # Setting up the parameters
        self.m = m
        self.eta = learning_rate
        self.seq_length = seq_length
        # Bias vector B and C and weight matrices U, W and V
        self.B = np.zeros((m, 1))
        self.C = np.zeros((k, 1))  # Ensure that grads.C has the correct shape
        self.U = np.random.rand(m, k) * sigma
        self.W = np.random.rand(m, m) * sigma
        self.V = np.random.rand(k, m) * sigma


# Gets the data from the text file
def get_data():
    file = open("goblet_book.txt", "r")
    all_text = file.read()
    unique_chars = set(all_text)
    char_to_int = {char: i for i, char in enumerate(unique_chars)}
    int_to_char = {i: char for i, char in enumerate(unique_chars)}
    return all_text, char_to_int, int_to_char


# softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# Synthesize text
def synthesize_text(model, h0, x0, characters_to_generate):
    """
    model: RNN model
    h0: initial hidden state
    x0: initial input
    n: number of characters to generate
    Does equations 1-4 in the assignment
    """
    print("Running synthesize_text function")
    data = np.copy(x0)
    n = characters_to_generate
    # Dim weights = (m, 1)
    hidden_weights = np.copy(h0)[:, np.newaxis]
    generated_samples = np.zeros((x0.shape[0], n))

    for t in range(n):
        a = model.W @ hidden_weights + model.U @ data + model.B
        h = np.tanh(a)
        o = model.V @ h + model.C
        p = softmax(o)
        # Select random character based on the probability distribution
        choice = np.random.choice(range(data.shape[0]), 1, p=p.flatten())
        data = np.zeros((data.shape))
        data[choice] = 1
        generated_samples[:, t] = data.flatten()

    return generated_samples


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


# Forward pass
def forward_pass(model, h_prev, x):
    """
    model: RNN model
    h_prev: previous hidden state
    x: input
    Returns the probabilities, hidden states and activations"""
    h = np.zeros((h_prev.shape[0], x.shape[1]))
    a = np.zeros((h_prev.shape[0], x.shape[1]))
    probs = np.zeros(x.shape)
    for t in range(x.shape[1]):
        # special case --> No previous hidden state
        if t == 0:
            a[:, t] = (
                model.W @ h_prev[:, np.newaxis]
                + model.U @ x[:, t][:, np.newaxis]
                + model.B
            ).flatten()
        else:
            a[:, t] = (
                model.W @ h[:, t - 1][:, np.newaxis]
                + model.U @ x[:, t][:, np.newaxis]
                + model.B
            ).flatten()
        h[:, t] = np.tanh(a[:, t])
        o = model.V @ h[:, t][:, np.newaxis] + model.C
        probs[:, t] = softmax(o).flatten()
    return probs, h, a


def backpass(rnn, target, probs, h, h_prev, aids, x):
    """
    rnn = RNN model
    target = target values
    probs = probabilities
    h = hidden states
    h_prev = previous hidden states
    a = activations
    x = input
    Returns the gradients of the weights and biases

    Admittedly this part was the hardest to implement. I had to look at the slides, ask CHATGPT4 and look All code written is my own although some parts were strictly from chatgpt4.
    """
    grad_h = list()
    grad_a = list()

    # Computation of the last gradient
    grad_o = -(target - probs).T

    # Last gradients of h and a
    grad_h.append(grad_o[-1][np.newaxis, :] @ rnn.V)
    grad_a.append((grad_h[-1] @ np.diag(1 - np.power(np.tanh(aids[:, -1]), 2))))

    # Computation of the remaining gradients
    for t in reversed(range(target.shape[1] - 1)):
        grad_h.append(grad_o[t][np.newaxis, :] @ rnn.V + grad_a[-1] @ rnn.W)
        grad_a.append(grad_h[-1] @ np.diag(1 - np.power(np.tanh(aids[:, t]), 2)))

    # Reverse it
    grad_a.reverse()
    grad_a = np.vstack(grad_a)
    rnn_grads = RNN()
    rnn_grads.V = grad_o.T @ h.T
    h_aux = np.zeros(h.shape)
    h_aux[:, 0] = h_prev
    h_aux[:, 1:] = h[:, 0:-1]
    rnn_grads.W = grad_a.T @ h_aux.T
    rnn_grads.U = grad_a.T @ x.T
    rnn_grads.B = np.sum(grad_a, axis=0)[:, np.newaxis]
    rnn_grads.C = np.sum(grad_o, axis=0)[:, np.newaxis]

    return rnn_grads


# Numerically
def gradients_numerical(rnn, x, y, h_prev):
    """
    rnn = RNN model
    x = input
    y = target values
    h_prev = previous hidden states
    Returns the numerical gradients

    @note: This code was generated by taking the TA's matlab code and asking chatgpt4 to convert it to python. Thereafter adjusting it to fit my needs and the codebase. This code was not entirely written by me, but I have adjusted it to fit the codebase and the assignment.
    """

    grads = {}
    eps = 1e-4

    for param_name in ["W", "U", "V", "B", "C"]:
        param = getattr(rnn, param_name)  # Get the parameter
        param_grad = np.zeros_like(param)

        # Create an iterable for tqdm
        iterable = tqdm(range(param.size), desc=f"Computing gradients for {param_name}")

        # Use np.nditer for efficient element-wise operation without explicitly creating tqdm wrapper around it
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

        for _ in iterable:  # Use tqdm iterable for progress display
            ix = it.multi_index
            original_value = param[ix]

            # Perturb parameter positively
            param[ix] = original_value + eps
            probs, _, _ = forward_pass(rnn, h_prev, x)
            loss1 = compute_loss(y, probs)

            # Perturb parameter negatively
            param[ix] = original_value - eps
            probs, _, _ = forward_pass(rnn, h_prev, x)
            loss2 = compute_loss(y, probs)

            # Compute numerical gradient
            param_grad[ix] = (loss1 - loss2) / (2 * eps)
            param[ix] = original_value  # Restore original value

            it.iternext()  # Move to the next index

        grads[param_name] = param_grad

    return grads


# Compares grads
def compare_gradients(do_it=False, m_value=10):
    if do_it:
        all_text, char_to_int, int_to_char = get_data()
        unique_chars = len(char_to_int)
        rnn = RNN(k=unique_chars, seq_length=1000, m=m_value)
        book_pointer = 0

        print("Checking gradients")
        # Do forward pass on some data
        x = one_hot(all_text[book_pointer : book_pointer + rnn.seq_length], char_to_int)
        y = one_hot(
            all_text[book_pointer + 1 : book_pointer + rnn.seq_length + 1], char_to_int
        )
        print("Length of x: ", x.shape)
        print("Length of y: ", y.shape)

        probs, h, a = forward_pass(rnn, np.zeros((rnn.m)), x)

        # Calculate the gradients
        print("Doing backpass")
        rnn_grads = backpass(rnn, y, probs, h, np.zeros((rnn.m)), a, x)
        # Check the gradients
        print("Doing numerical gradients")
        grads = gradients_numerical(rnn, x, y, np.zeros((rnn.m)))
        # Compare the gradients
        for key in grads:  # Iterate through keys in grads dictionary
            print(f"Checking gradient for {key}")
            grad_diff = np.linalg.norm(
                grads[key] - getattr(rnn_grads, key)
            )  # Use getattr to access rnn_grads attributes
            print(f"Gradient difference: {grad_diff}")


def adagrad(squared_grads, grads, old_params, eta):
    # Update the squared gradients
    m_new = squared_grads + np.square(grads)
    # Update the parameters
    new_params = old_params - eta * grads / np.sqrt(m_new + 1e-8)
    return m_new, new_params


def main():
    all_text, char_to_int, int_to_char = get_data()
    print("Characters in all_text: ", len(all_text))

    # Get 10% of all_text
    all_text = all_text[: int(len(all_text) * 0.1)]

    unique_chars = len(char_to_int)
    rnn = RNN(k=unique_chars, seq_length=25)

    # Initalize the learning
    book_pointer = 0
    loss_list = []
    steps_list = []
    steps=0
    h_prev = np.zeros((rnn.m))
    squared_grads = [0, 0, 0, 0, 0]

    # Used to update model
    best_loss = 0

    # Check the args when compiling. If you want to check the gradients, set do_it to True
    compare_gradients(do_it=False, m_value=10)

    # Starting the learning
    for epoch in trange(2, desc="Epoch"):
        # In new epoch you want to set the h_prev to zero
        h_prev = np.zeros((rnn.m))

        for idx, book_pointer in enumerate(
            (trange(0, len(all_text) - rnn.seq_length, rnn.seq_length, desc="iteration"))
        ):
            steps+=1
            steps_list.append(steps)

            # One-hot encoding of the sequence
            data = one_hot(
                all_text[book_pointer : book_pointer + rnn.seq_length], char_to_int
            )
            target = one_hot(
                all_text[book_pointer + 1 : book_pointer + rnn.seq_length + 1],
                char_to_int,
            )

            # Forward and backward pass
            probs, h, a = forward_pass(rnn, h_prev, data)
            rnn_grads = backpass(rnn, target, probs, h, h_prev, a, data)

            # Handle exploding gradients
            for grad_x, att in enumerate(["W", "U", "V", "B", "C"]):
                grad = getattr(rnn_grads, att)
                grad = np.clip(grad, -5, 5)
                # Do adagrad and update the parameters and weights
                squared_grads[grad_x], new_param = adagrad(
                    squared_grads[grad_x], grad, getattr(rnn, att), rnn.eta
                )
                setattr(rnn, att, new_param)
                squared_grads[grad_x] = squared_grads[grad_x]
                

            if idx == 0 and epoch == 0:
                smooth_loss = compute_loss(target, probs)
                loss_list.append(smooth_loss)
                best_loss = smooth_loss
                rnn = copy.deepcopy(rnn)
            
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * compute_loss(target, probs)
                # If this loss is smaller, update the model and best loss
                if smooth_loss < best_loss:
                    best_loss = smooth_loss
                    new_best = copy.deepcopy(rnn)
                    rnn = new_best
                    loss_list.append(smooth_loss)
                                

            if idx % 100 == 0:
                smooth_loss = 0.999 * smooth_loss + 0.001 * compute_loss(target, probs)
                loss_list.append(smooth_loss)

            # Update the weights
            h_prev = h[:, -1]
    
    # Loss the loss (y) over steps(x)
    plt.plot(np.arange(len(loss_list)) * 100, loss_list)
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
