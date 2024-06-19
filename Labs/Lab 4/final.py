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
    gradient_collection = {
        "U": np.zeros(model.U.shape),
        "W": np.zeros(model.W.shape),
        "V": np.zeros(model.V.shape),
        "B": np.zeros(model.B.shape),
        "C": np.zeros(model.C.shape),
    }

    seq_len = targets.shape[1]
    grad_output = -(targets - predictions).T

    gradient_collection["V"] = grad_output.T @ hidden_states.T
    gradient_collection["C"] = np.sum(grad_output, axis=0)[:, np.newaxis]

    next_hidden_state_gradient = np.zeros_like(initial_hidden_state)

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

        # Inner loop for full backpropagation through time
        for step_back in range(seq_position - 1, -1, -1):
            grad_h = np.dot(grad_h, model.W) * (
                1 - np.tanh(activations[:, step_back]) ** 2
            )
            if step_back > 0:
                prev_h = hidden_states[:, step_back - 1]
            else:
                prev_h = initial_hidden_state

            gradient_collection["W"] += np.outer(grad_h, prev_h)
            gradient_collection["U"] += np.outer(grad_h, input_sequence[:, step_back])
            gradient_collection["B"] += grad_h[:, np.newaxis]

        next_hidden_state_gradient = np.dot(model.W.T, grad_h)

    return gradient_collection


def check_gradients(do_it=False, m_value = 15):
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
        grads_slower = gradients_numerical(rnn, x, y, np.zeros((rnn.M)))
        print(grads_slower.keys())
        # Compare the gradients
        for key in grads_slower:  # Iterate through keys in grads dictionary
            print(f"Checking gradient for {key}")
            grad_diff = np.linalg.norm(
                grads_slower[key] - grads_faster[key]
            )  # Use getattr to access rnn_grads attributes
            print(f"Gradient difference: {grad_diff}")
            if grad_diff > 1e-5:
                print(f"Gradient check failed for {key}")
                sys.exit(1)
        
        print("Gradient check passed")

def test_gradients():
    all_text, char_to_int, int_to_char = preprocess()
    unique_chars = len(char_to_int)
    rnn = RNN(k=unique_chars, seq_length=1000, m=100)
    book_pointer = 0
    h_prev = np.zeros((rnn.M))

    # forward
    x = one_hot(all_text[book_pointer : book_pointer + rnn.seq_length], char_to_int)
    y = one_hot(
        all_text[book_pointer + 1 : book_pointer + rnn.seq_length + 1], char_to_int
    )

    # compute loss
    probs, h, a = forward_pass(rnn, np.zeros((rnn.M)), x)
    # backward
    grads_faster = backpass(rnn, y, probs, h, np.zeros((rnn.M)), a, x)

    # Compute loss
    loss1 = compute_loss(y, probs)
    print(f"Loss: {loss1}")

    # Run another forward and pass and check if the loss decreases
    h_last= h[:, -1]
    probs, h, a = forward_pass(rnn, h_last, x)
    loss2 = compute_loss(y, probs)
    print(f"Loss: {loss2}")
    if loss2 < loss1:
        print("Loss is decreasing")
    else:
        print("Loss is not decreasing")


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
    print("Checking if gradients actually improve the model...")
    test_gradients()


def adagrad(squared_grads, grads, old_params, eta):
    """Taken from eqation 6 and 7 in the assignment
    Input: squared_grads, grads, old_params, eta
    Output: m_new, new_params
    """
    # Update the squared gradients
    m_new = squared_grads + np.square(grads)
    # Update the parameters
    new_params = old_params - eta * grads / np.sqrt(m_new + 1e-8)
    return m_new, new_params


def handle_grads(rnn, grads_fast, adagrad_params):
    # Handle exploding gradients and update weights using adagrad
    for _, att in enumerate(["W", "U", "V", "B", "C"]):
        grad = grads_fast[att]
        grad = np.clip(grad, -5, 5)  # Clipping gradients to avoid explosion
        adagrad_params[att], new_param = adagrad(
            adagrad_params[att], grad, getattr(rnn, att), rnn.eta
        )
        setattr(rnn, att, new_param)
    return adagrad_params


# Synthesize text
def synthesize_text(model, h0, x0, characters_to_generate):
    """
    model: RNN model
    h0: initial hidden state
    x0: initial input
    characters_to_generate: number of characters to generate
    Does equations 1-4 in the assignment
    """
    n = characters_to_generate
    h = np.copy(h0)[:, np.newaxis]  # Initialize hidden state
    x = np.copy(x0)[:, np.newaxis]  # Ensure x0 is a column vector
    generated_samples = np.zeros((x0.shape[0], n))  # Initialize the output array

    for t in range(n):
        # Compute the next hidden state
        a = model.W @ h + model.U @ x + model.B
        h = np.tanh(a)
        # Compute the output
        o = model.V @ h + model.C
        p = softmax(o).flatten()  # Ensure p is a flat array of probabilities
        # Sample the next character
        choice = np.random.choice(range(x.shape[0]), p=p)
        # Convert to one-hot encoding
        x = np.zeros((x.shape[0], 1))
        x[choice, 0] = 1
        # Store the result
        generated_samples[:, t] = x.flatten()

    return generated_samples


def generate_text(iteration, text):
    # Creates a .csv file. Headers are "Iteration" and "Text"
    with open("generated_text.csv", "a") as file:
        # Create the header
        if iteration == 0:
            file.write("Iteration,Text\n")
        # Write the iteration and the generated text
        file.write(f"{iteration},{text}\n")

def store_weights(rnn):
    # Store the weights
    np.save("W.npy", rnn.W)
    np.save("U.npy", rnn.U)
    np.save("V.npy", rnn.V)
    np.save("B.npy", rnn.B)
    np.save("C.npy", rnn.C)


def train_model(run_small=False,generative=False):
    all_text, char_to_int, int_to_char = preprocess()
    if run_small:
        # Run 5% of the entire text
        all_text = all_text[: int(0.05 * len(all_text))]

    epochs = 1 if run_small else 3

    unique_chars = len(char_to_int)
    rnn = RNN(k=unique_chars)

    # Initialize hidden state
    h_prev = np.zeros((rnn.M))
    adagrad_params = {
        "U": 0,
        "W": 0,
        "V": 0,
        "B": 0,
        "C": 0,
    }

    # Start the learning process
    book_pointer = 0
    loss_list = []
    iterations = [] 
    total_updates = 0

    # Starting the learning
    for epoch in trange(epochs, desc="Epoch"):
        for idx, book_pointer in enumerate(
            trange(0, len(all_text) - rnn.seq_length, rnn.seq_length, desc="Iteration")
        ):
            # Get the input and target
            data = one_hot(all_text[book_pointer : book_pointer + rnn.seq_length], char_to_int)
            target = one_hot(
                all_text[book_pointer + 1 : book_pointer + rnn.seq_length + 1], char_to_int
            )
            # Forward pass
            probs, hidden_weights, activations = forward_pass(rnn, h_prev, data)

            # Backward pas
            grads_fast = backpass(rnn, target, probs, hidden_weights, h_prev, activations, data)

            # Handle exploding gradients 
            for grad in grads_fast:
                grads_fast[grad] = np.clip(grads_fast[grad], -5, 5)
            
            for attribute in adagrad_params.keys():
                adagrad_params[attribute] += grads_fast[attribute] ** 2
                new_param = getattr(rnn, attribute) - rnn.eta * grads_fast[attribute] / np.sqrt(adagrad_params[attribute] + 1e-8)
                setattr(rnn, attribute, new_param)

            # Tracking the loss and iterations
            loss_list.append(compute_loss(target, probs))
            print
            if total_updates == 0:
                loss_list.append(compute_loss(target, probs))
                iterations.append(total_updates)
            elif total_updates % 1000 == 0 and total_updates != 0:
                current_loss = compute_loss(target, probs)
                loss = 0.999 * loss_list[-1] + 0.001 * current_loss

            if total_updates % 100== 0:
                loss_list.append(loss)


            # Update the hidden state
            h_prev = hidden_weights[:, -1]
            total_updates += 1
            if total_updates % 10000 ==0 and generative:
                # Generate text
                generated_text = synthesize_text(rnn, h_prev, data[:, 0], 200)
                generated_text = np.argmax(generated_text, axis=0)
                generated_text = "".join([int_to_char[i] for i in generated_text])
                generate_text(total_updates, generated_text)
                print(f"Generated text: {generated_text}")

    # Plot the loss
    return rnn, loss_list, iterations


def main():
    print("Run checks or not? (y/n): ")
    run_checks = input()
    if run_checks == "y":
        run_sanity_checks()
    else:
        print("Running the main program...")

    print("Would you like to run a small training session? (y/n): ")
    run_training = input()
    if run_training == "y":
        print("Running a small training session...")
        rnn, loss_list, _ = train_model(run_small=True, generative=True)
        # Plot the loss
        plt.plot(loss_list)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss over iterations")
        plt.show()

    else:
        print("Running a full training session...")
        # Delete the previous generated text
        open("generated_text.csv", "w").close()
        rnn, loss_list,_= train_model(run_small=False, generative=True)
        # Plot the loss
        plt.plot(loss_list)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss over iterations")
        plt.show()
        store_weights(rnn)


if __name__ == "__main__":
    main()
