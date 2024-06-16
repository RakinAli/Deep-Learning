"""
Assignment 4 at DD2424 Deep Learning in Data Science by @RakinAli
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
import copy
from Rnn import RNN, softmax, compute_loss, one_hot
import sys

def preprocess():
    """
    Preprocesses the text data by reading from a file, creating mappings between characters and integers,
    and returning the processed data and mappings.

    Returns:
        all_text (str): The entire text read from the file.
        char_to_int (dict): A dictionary mapping each unique character to its corresponding integer index.
        int_to_char (dict): A dictionary mapping each integer index to its corresponding unique character.
    """
    with open("goblet_book.txt", "r") as file:
        all_text = file.read()
        unique_chars = sorted(set(all_text))
        char_to_int = {char: i for i, char in enumerate(unique_chars)}
        int_to_char = {i: char for i, char in enumerate(unique_chars)}

    return all_text, char_to_int, int_to_char


def check_preprocess():
    """
    Asserts the preprocess function works correctly
    
    This function tests the preprocess function by converting a given text to its integer representation
    using the char_to_int dictionary returned by the preprocess function. It then converts the integer
    representation back to the original text using the int_to_char dictionary and asserts that the original
    text is equal to the converted text. If the assertion passes, it prints a success message.
    """
    all_text, char_to_int, int_to_char = preprocess()
    
    # Convert Harry Potter to
    text_test = "Harry Potter"
    text_test_converted = [char_to_int[char] for char in text_test]
    text_test_converted_back = [int_to_char[i] for i in text_test_converted]
    assert text_test == "".join(text_test_converted_back) # Redo preprocessing
    print("Preprocess test passed")


# Loss function
def compute_loss(target, probs):
    """
    target: target values
    probs: probabilities
    Returns the loss
    """
    return -np.sum(np.log(np.sum(target * probs, axis=0)))


# Forward pass
def forward_pass(model, h_prev, x):
    """
    Input:
    model: RNN model
    h_prev: previous hidden state
    x: input
    Returns the probabilities, hidden states and activations
    Output:
    probs: probabilities
    h: hidden states
    a: activations
    """
    h = np.zeros((h_prev.shape[0], x.shape[1]))
    a = np.zeros((h_prev.shape[0], x.shape[1]))
    probs = np.zeros(x.shape)

    for t in range(x.shape[1]):
        a[:, t] = compute_activation(model, h_prev if t == 0 else h[:, t - 1], x[:, t])
        h[:, t] = np.tanh(a[:, t])
        o = model.V @ h[:, t][:, np.newaxis] + model.C
        probs[:, t] = softmax(o).flatten()
    return probs, h, a


def compute_activation(model, h_prev, x_t):
    """
    This helper function calculates the activation for each time step.

    Args:
        model: The RNN model object.
        h_prev: The previous hidden state (for t=0) or the current hidden state (for t>0).
        x_t: The input vector at the current time step.

    Returns:
        The activation vector after flattening.
    """
    return (
        model.W @ h_prev[:, np.newaxis] + model.U @ x_t[:, np.newaxis] + model.B
    ).flatten()


# Numerically
def gradients_numerical(rnn, x, y, h_prev):
    """
    rnn = RNN model
    x = input
    y = target values
    h_prev = previous hidden states
    Returns the numerical gradients

    @Note: Took the TA:s code and modified it to fit the assignment

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


def check_forward_pass():
    """
    Performs a forward pass test for the RNN model.

    Returns:
    None
    """
    all_text, char_to_int, int_to_char = preprocess()
    unique_chars = len(char_to_int)

    rnn  = RNN(k = unique_chars, m = 10)

    h_prev = np.zeros(rnn.M)
    data = one_hot(all_text[0 : 1 + rnn.seq_length], char_to_int)
    target = one_hot(all_text[0 + 1 : 0 + rnn.seq_length + 1], char_to_int)

    probs, h, a = forward_pass(rnn, h_prev, data)
    print("Forward pass test passed")


def backpass(
    model,
    targets,
    predictions,
    hidden_states,
    initial_hidden_state,
    activations,
    input_sequence,
):
    """
    Calculate gradients for the RNN model parameters.

    Args:
        model: RNN model
        targets: Target sequence of shape (K, N)
        predictions: Predictions of shape (K, N)
        hidden_states: Hidden state sequence of shape (m, N)
        initial_hidden_state: Initial hidden state of shape (m,)
        activations: Activation sequence of shape (m, N)
        input_sequence: Input sequence of shape (K, N)

    Returns:
        gradient_collection: Dictionary containing gradients of the model parameters
    """
    # Initialize gradient collection
    gradient_collection = {
        "U": np.zeros(model.U.shape),
        "W": np.zeros(model.W.shape),
        "V": np.zeros(model.V.shape),
        "B": np.zeros(model.B.shape),
        "C": np.zeros(model.C.shape),
    }

    seq_len = targets.shape[1]
    grad_output = -(targets - predictions).T

    # Calculate gradients for V and C directly
    gradient_collection["V"] = grad_output.T @ hidden_states.T
    gradient_collection["C"] = np.sum(grad_output, axis=0)[:, np.newaxis]

    next_hidden_state_gradient = np.zeros_like(initial_hidden_state)

    # Loop to calculate gradients for W, U, and B
    for seq_position in reversed(range(seq_len)):
        grad_h = np.dot(grad_output[seq_position], model.V) * (
            1 - np.tanh(activations[:, seq_position]) ** 2
        )
        grad_h += next_hidden_state_gradient

        if seq_position > 0:
            prev_h = hidden_states[:, seq_position - 1]
        else:
            prev_h = initial_hidden_state

        gradient_collection["W"] += np.outer(grad_h, prev_h)
        gradient_collection["U"] += np.outer(grad_h, input_sequence[:, seq_position])
        gradient_collection["B"] += grad_h[:, np.newaxis]

        next_hidden_state_gradient = np.dot(model.W.T, grad_h)

    return gradient_collection


def check_gradients(do_it=False, m_value = 10):
    if do_it:
        all_text, char_to_int, int_to_char = preprocess()
        unique_chars = len(char_to_int)
        rnn = RNN(k=unique_chars, seq_length=1000, m=m_value)
        book_pointer = 0

        print("Checking gradients")
        # Do forward pass on some data
        x = one_hot(all_text[book_pointer : book_pointer + rnn.seq_length], char_to_int)
        y = one_hot(
            all_text[book_pointer + 1 : book_pointer + rnn.seq_length + 1], char_to_int
        )

        probs, h, a = forward_pass(rnn, np.zeros((rnn.M)), x)

        # Calculate the gradients
        grads_faster = backpass(rnn, y, probs, h, np.zeros((rnn.M)), a, x)
        # just print out the keys of the dictionary
        print(grads_faster.keys())
        # Check the gradients
        grads = gradients_numerical(rnn, x, y, np.zeros((rnn.M)))
        print(grads.keys())
        # Compare the gradients
        for key in grads:  # Iterate through keys in grads dictionary
            print(f"Checking gradient for {key}")
            grad_diff = np.linalg.norm(
                grads[key] - grads_faster[key]
            )  # Use getattr to access rnn_grads attributes
            print(f"Gradient difference: {grad_diff}")
            if grad_diff > 1e-5:
                print(f"Gradient check failed for {key}")
                sys.exit(1)
        
        print("Gradient check passed")



def run_sanity_checks():
    """
    Runs the sanity checks for the RNN model.

    Returns:
    None
    """
    print("Running sanity checks...")
    print("Checking if preprocess works correctly...")
    all_text, char_to_int, int_to_char = preprocess()
    check_preprocess()
    print("Checking if forward pass works correctly...")
    check_forward_pass()
    print("Checking if numerically & analytical allign...")
    check_gradients(do_it=True)
    print("Checking if ")



def main():
    print("Run checks or not? (y/n): ")
    run_checks = input()
    if run_checks == "y":
        run_sanity_checks()
    else:
        print("Running the main program...")
    
    all_text, char_to_int, int_to_char = preprocess()
    unique_chars = len(char_to_int)


if __name__ == "__main__":
    main()
