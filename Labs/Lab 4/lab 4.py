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
    n: number of characters to generate
    Does equations 1-4 in the assignment
    """
    data = np.copy(x0)
    n = characters_to_generate
    hidden_weights = np.copy(h0)[:, np.newaxis]
    generated_samples = np.zeros((x0.shape[0], n))

    for t in range(n):
        # Activations
        a = model.W @ hidden_weights + model.U @ data + model.B
        # Hidden states
        h = np.tanh(a)
        # Output probabilities
        o = model.V @ h + model.C
        p = softmax(o)
        """Randomly select a character according to the probabilities"""
        choice = np.random.choice(range(data.shape[0]), 1, p=p.flatten())
        data = np.zeros((data.shape))
        data[choice] = 1
        generated_samples[:, t] = data.flatten()

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
    """
    Calculate gradients for the RNN model parameters.
    """
    # Initialize the gradients
    grad_output = -(targets - predictions).T

    # Calculating gradients for V directly
    grad_V = grad_output.T @ hidden_states.T

    # Initializing gradient storage for other parameters
    grad_W = np.zeros_like(model.W)
    grad_U = np.zeros_like(model.U)
    grad_B = np.zeros_like(model.B)
    grad_C = np.sum(grad_output, axis=0)[:, np.newaxis]  # Direct sum for grad_C

    # Loop to calculate gradients for W, U, and B
    for t in reversed(range(len(targets[0]))):
        # Calculate gradient of hidden state
        grad_h = np.dot(grad_output[t], model.V) * (1 - np.tanh(activations[:, t]) ** 2)

        if t > 0:
            prev_h = hidden_states[:, t - 1]
        else:
            prev_h = initial_hidden_state

        grad_W += np.outer(grad_h, prev_h)
        grad_U += np.outer(grad_h, input_sequence[:, t])
        grad_B += grad_h.reshape(-1, 1)

        # Propagate the gradient back through time
        for step_back in range(t - 1, max(t - model.seq_length, -1), -1):
            grad_h = np.dot(grad_h, model.W) * (
                1 - np.tanh(activations[:, step_back]) ** 2
            )
            if step_back > 0:
                grad_W += np.outer(grad_h, hidden_states[:, step_back - 1])
            else:
                grad_W += np.outer(grad_h, initial_hidden_state)
            grad_U += np.outer(grad_h, input_sequence[:, step_back])
            grad_B += grad_h.reshape(-1, 1)

    # Create a new RNN instance to store gradients
    gradients = RNN()
    gradients.V = grad_V
    gradients.W = grad_W
    gradients.U = grad_U
    gradients.B = grad_B
    gradients.C = grad_C

    return gradients


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
    """ Taken from eqation 6 and 7 in the assignment"""
    # Update the squared gradients
    m_new = squared_grads + np.square(grads)
    # Update the parameters
    new_params = old_params - eta * grads / np.sqrt(m_new + 1e-8)
    return m_new, new_params


def main():
    all_text, char_to_int, int_to_char = get_data()
    print("Characters in all_text: ", len(all_text))

    # Get 10% of all_text
    all_text = all_text[: int(len(all_text) *0.01)]

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
    compare_gradients(do_it=False, m_value=10)

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
                all_text[book_pointer + 1 : book_pointer + rnn.seq_length + 1], char_to_int,
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

            # Synthesize text every 1000 steps or at the first step
            if idx % 1000 == 0 or (idx == 0 and epoch == 0):
                test = one_hot('.', char_to_int)

                generated_text = synthesize_text(rnn, h_prev, test, 200)
                generated_text = np.argmax(generated_text, axis=0)
                generated_text = "".join([int_to_char[x] for x in generated_text])
                print(" Loss: ", smooth_loss, " Text: ", generated_text)
                with open("generated_text.txt", "a") as f:
                    f.write(f"Epoch {epoch}, Iteration {idx}, Steps: {steps}, Text: {generated_text}\n")
            
            steps += 1

            # Update the hidden state for the next sequence
            h_prev = hidden_weights[:, -1]

    # Plotting the loss
    plt.plot(np.arange(len(loss_list)) * 100, loss_list)
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.show()

    # Add the final synthesized text to the file
    test = one_hot('.', char_to_int)
    generated_text = synthesize_text(best_rnn, h_prev, test, 1000)
    generated_text = np.argmax(generated_text, axis=0)
    generated_text = "".join([int_to_char[x] for x in generated_text])
    with open("generated_text.txt", "a") as f:
        f.write(f"Final:{generated_text} \n")


if __name__ == "__main__":
    main()
