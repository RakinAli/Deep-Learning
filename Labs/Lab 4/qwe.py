"""
Assignment 4 at DD2424 Deep Learning in Data Science by @RakinAli
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
import copy
from Rnn import RNN, softmax, compute_loss, one_hot, get_data


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

        # Inner loop for backpropagation through time
        for step_back in range(
            seq_position - 1, max(seq_position - model.seq_length, -1), -1
        ):
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

    rnn = RNN()
    rnn.W = gradient_collection["W"]
    rnn.U = gradient_collection["U"]
    rnn.V = gradient_collection["V"]
    rnn.B = gradient_collection["B"]
    rnn.C = gradient_collection["C"]

    return rnn


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

        probs, h, a = forward_pass(rnn, np.zeros((rnn.M)), x)

        # Calculate the gradients
        rnn_grads = backpass(rnn, y, probs, h, np.zeros((rnn.M)), a, x)
        # Check the gradients
        grads = gradients_numerical(rnn, x, y, np.zeros((rnn.M)))
        # Compare the gradients
        for key in grads:  # Iterate through keys in grads dictionary
            print(f"Checking gradient for {key}")
            grad_diff = np.linalg.norm(
                grads[key] - getattr(rnn_grads, key)
            )  # Use getattr to access rnn_grads attributes
            print(f"Gradient difference: {grad_diff}")


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


def main():
    all_text, char_to_int, int_to_char = get_data()
    print("Characters in all_text: ", len(all_text))

    # Get 10% of all_text
    all_text = all_text[: int(len(all_text))]

    unique_chars = len(char_to_int)
    rnn = RNN(k=unique_chars)
    best_rnn = RNN()

    # Initalize the learning
    book_pointer = 0
    loss_list = []
    h_prev = np.zeros((rnn.M))
    squared_grads = [0, 0, 0, 0, 0]
    steps = 0

    # Used to update model
    best_loss = 0

    # Check the args when compiling. If you want to check the gradients, set do_it to True
    compare_gradients(do_it=True, m_value=10)

    # Starting the learning
    for epoch in trange(2, desc="Epoch"):
        for idx, book_pointer in enumerate(
            trange(0, len(all_text) - rnn.seq_length, rnn.seq_length, desc="Iteration")
        ):
            # One-hot encoding of the sequence
            data = one_hot(
                all_text[book_pointer : book_pointer + rnn.seq_length], char_to_int
            )
            target = one_hot(
                all_text[book_pointer + 1 : book_pointer + rnn.seq_length + 1],
                char_to_int,
            )

            # Forward and backward pass
            probs, hidden_weights, a = forward_pass(rnn, h_prev, data)
            rnn_grads = backpass(rnn, target, probs, hidden_weights, h_prev, a, data)

            # Handle exploding gradients and update weights using adagrad
            for grad_x, att in enumerate(["W", "U", "V", "B", "C"]):
                grad = getattr(rnn_grads, att)
                grad = np.clip(grad, -5, 5)  # Clipping gradients to avoid explosion
                squared_grads[grad_x], new_param = adagrad(
                    squared_grads[grad_x], grad, getattr(rnn, att), rnn.eta
                )
                setattr(rnn, att, new_param)

            # Compute and track the smooth loss
            if idx == 0 and epoch == 0:
                smooth_loss = compute_loss(target, probs)
                best_loss = smooth_loss  # Initial best loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * compute_loss(target, probs)

            # Update the best model if current loss is lower
            if smooth_loss < best_loss:
                best_loss = smooth_loss
                best_rnn = copy.deepcopy(rnn)

            # Append loss for plotting every 100 iterations
            if idx % 100 == 0:
                loss_list.append(smooth_loss)
            
            if steps % 1000 == 0:
                print(f"Loss at step {steps}: {smooth_loss}")
            
            if steps % 10000 == 0:
                # Add the synthesized text to the file
                test = one_hot(".", char_to_int)
                generated_text = synthesize_text(rnn, h_prev, test, 200)
                generated_text = np.argmax(generated_text, axis=0)
                generated_text = "".join([int_to_char[x] for x in generated_text])
                print("Text: ", generated_text)

            steps += 1

            # Update the hidden state for the next sequence
            h_prev = hidden_weights[:, -1]

    # Plotting the loss
    plt.plot(np.arange(len(loss_list)) * 100, loss_list)
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.show()

    # Add the final synthesized text to the file
    test = one_hot(".", char_to_int)
    generated_text = synthesize_text(best_rnn, h_prev, test, 1000)
    generated_text = np.argmax(generated_text, axis=0)
    generated_text = "".join([int_to_char[x] for x in generated_text])
    print("Final: ", generated_text)

if __name__ == "__main__":
    main()
