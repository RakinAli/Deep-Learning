""" 
  Author: Rakin Ali
  Date:    27/03/2022
  Description: Lab 1 - Backpropagation
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Paths
DATAPATH = "Datasets/cifar-10-batches-py/"
LENGTH = 1024  # Pixels per image
WIDTH = 32  # Width of image
D_BATCH = ['data_batch_1', 'data_batch_2',
           'data_batch_3', 'data_batch_4', 'data_batch_5']
T_BATCH = 'test_batch'
SIZE = 32  # Pixel dimension of the image


def LoadBatch(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def getting_started():
    """@docstring:
    This function will help you get started on the CIFAR-10 dataset. Reads the bathces 0 and batches 1 to do training and validation
    Reads the test_batch to do testing
    Returns:
    - data_train: A numpy array of shape (3072, 10000) containing the training data.
    - labels_train: A numpy array of shape (10000,) containing the training labels.
    - data_val: A numpy array of shape (3072, 10000) containing the validation data.
    - labels_val: A numpy array of shape (10000,) containing the validation labels.
    - data_test: A numpy array of shape (3072, 10000) containing the test data.
    - labels_test: A numpy array of shape (10000,) containing the test labels.
    - labels: A list of length 10 containing the names of the classes.
    """
    # Read training data from Batch 0
    # Loading the batch_0 as a dictionary
    batch_0 = LoadBatch(DATAPATH + D_BATCH[0])
    # print(data.keys())
    data_train = batch_0['data']  # Grabbing the data
    labels_train = batch_0['labels']  # Grabbing the labels

    # Read validation data from Batch 1
    # Loading the batch_1 as a dictionary
    batch_1 = LoadBatch(DATAPATH + D_BATCH[1])
    # print(data.keys())
    data_val = batch_1['data']  # Grabbing the data
    labels_val = batch_1['labels']  # Grabbing the labels

    # Read test data from test_batch
    # Loading the test_batch as a dictionary
    batch_test = LoadBatch(DATAPATH + T_BATCH)
    # print(data.keys())
    data_test = batch_test['data']  # Grabbing the data
    labels_test = batch_test['labels']  # Grabbing the labels

    # Grading the labels
    labels = LoadBatch(DATAPATH + 'batches.meta')['label_names']
    return data_train, labels_train, data_val, labels_val, data_test, labels_test, labels

# Preprocess data"""


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def normalise(data, mean, std):
    """@docstring:
    Preprocess the data by subtracting the mean image and dividing by the standard deviation of each feature.
    Inputs:
    - data: A numpy array of shape (N,D) containing the data to be preprocessed.
    - mean: A numpy array of shape (D,1) containing the mean image to subtract from the data.
    - std: A numpy array of shape (D,1) containing the standard deviation to divide the data by.
    Returns:
    - data: The preprocessed data, of shape (D, N)
    - mean: The mean image, of shape (D,1)
    - std: The standard deviation, of shape (D,1)
    """
    data = np.float64(data)
    if mean is None:
        mean = np.mean(data, axis=0)  # Mean of each column
    if std is None:
        std = np.std(data, axis=0)  # Standard deviation of each column
    data = (data - mean) / std
    data = np.transpose(data)  # Transpose the data to get the shape (D, N)
    return np.array(data), mean, std

def one_hot_encoding(labels, dimensions):
    """@docstring:
    One-hot encode the labels.
    Inputs:
    - labels: A list of length N containing the labels.
    - dimensions: The number of possible labels.
    Returns:
    - one_hot: A numpy array of shape (dimensions x N) containing the one-hot encoded labels.
    """
    one_hot = np.zeros((dimensions, len(labels)))
    for i in range(len(labels)):
        one_hot[labels[i], i] = 1
    return one_hot


def random_weight_bias_init(data, labels):
    """@docstring:
    Initialize the weights and biases of the network. Each entry to have Gaussian random values with 0 mean and 0.01 standard deviation.
    Inputs:
    - data: A numpy array of shape (3072, 10000) containing the training data.
    - labels: A list of length 10 containing the names of the classes.

    Returns:
    - W: A numpy array of shape (K,D) containing the weights.
    - b: A numpy array of shape (K,1) containing the biases.
    """
    weight = np.random.normal(0, 0.01, (len(labels), data.shape[0]))
    bias = np.random.normal(0, 0.01, (len(labels), 1))
    return weight, bias


def evaluate_classifier(data, weight, bias):
    """@docstring:
    Evaluate the classifier for all the input images and return the scores.
    Inputs:
    - X: A numpy array of shape (D, N) containing the image data.
    - W: A numpy array of shape (K, D) containing the weights.
    - b: A numpy array of shape (K, 1) containing the biases.
    Returns:
    - s: A numpy array of shape (K, N) containing the computed scores.
    """
    s = np.dot(weight, data) + bias
    p = softmax(s)
    return p


def get_loss(data, labels, weight, bias):
    """@docstring:
    Compute the loss for the current batch of training data.
    Inputs:
    - X: A numpy array of shape (D, N) containing the image data.
    - Y: A numpy array of shape (K, N) containing the one-hot encoded labels.
    - W: A numpy array of shape (K, D) containing the weights.
    - b: A numpy array of shape (K, 1) containing the biases.
    Returns:
    - loss: The loss for the batch of training data.
    """
    p = evaluate_classifier(data, weight, bias)
    loss = -np.log(np.sum(labels * p, axis=0))
    loss = np.sum(loss)
    loss /= data.shape[1]
    return loss


def compute_cost(data, labels, weight, bias, lamda=0):
    """@docstring:
    Compute the cost for the current batch of training data.
    Inputs:
    - X: A numpy array of shape (D, N) containing the image data.
    - Y: A numpy array of shape (K, N) containing the one-hot encoded labels.
    - W: A numpy array of shape (K, D) containing the weights.
    - b: A numpy array of shape (K, 1) containing the biases.
    Returns:
    - cost: The cost for the batch of training data.
    """
    loss = get_loss(data, labels, weight, bias)
    # Regulariser --> Punishing weights
    weight_sum = np.sum(np.square(weight))

    # Cost = sum of loss + punishment of weights.
    cost = loss + (lamda * weight_sum)
    return cost


def compute_accuracy(data, labels, weights, bias):
    """@docstring:
    Compute the accuracy of the network for the given data and labels.
    Inputs:
    - X: A numpy array of shape (D, N) containing the image data.
    - Y: A numpy array of shape (K, N) containing the one-hot encoded labels.
    - W: A numpy array of shape (K, D) containing the weights.
    - b: A numpy array of shape (K, 1) containing the biases.
    Returns:
    - acc: The accuracy of the network for the given data and labels.
    """
    p = evaluate_classifier(data, weights, bias)
    p = np.argmax(p, axis=0)
    labels = np.argmax(labels, axis=0)
    acc = np.sum(p == labels) / len(labels)
    return acc


def compute_gradients(data, labels, p, weight, lmda):
    """@docstring:
    Compute the gradients of the loss function with respect to the parameters.
    Inputs:
    - X: A numpy array of shape (D, N) containing the image data.
    - Y: A numpy array of shape (K, N) containing the one-hot encoded labels.
    - p: A numpy array of shape (K, N) containing the probabilities for the labels.
    - W: A numpy array of shape (K, D) containing the weights.
    Returns:
    - grad_W: A numpy array of shape (K, D) containing the gradients of the loss with respect to the weights.
    - grad_b: A numpy array of shape (K, 1) containing the gradients of the loss with respect to the biases.
    """
    
    g = -(labels-p)  # K x N
    weight_gradient = (np.dot(g, np.transpose(data))/data.shape[1]) + 2 * lmda * weight  # K x D
    bias_gradient = np.sum(g, axis=1, keepdims=True)/data.shape[1]  # K x 1
    grad_w = weight_gradient
    grad_b = bias_gradient
    return grad_w, grad_b


if __name__ == "__main__":
    # Getting started
    data_train, labels_train, data_val, labels_val, data_test, labels_test, labels_names = getting_started()

    # Preprocessing the data --Z Normlaisation part
    data_train, mean, std = normalise(data_train, None, None)
    data_val, _, _ = normalise(data_val, mean, std)
    data_test, _, _ = normalise(data_test, mean, std)

    # One hot encoding --> One hot encoding the labels
    labels_train = one_hot_encoding(labels_train, 10)  # k x N matrix
    labels_val = one_hot_encoding(labels_val, 10)  # k x N matrix
    labels_test = one_hot_encoding(labels_test, 10)  # k x N matrix

    # Random weight and bias initialisation
    weight, bias = random_weight_bias_init(data_train, labels_names)

    # Evaluating the classifier
    probabilities = evaluate_classifier(data_train, weight, bias)

    #Mini-batch gradient descent
    # Hyperparameters
    epochs = 40
    batch_size = 1000
    learning_rate = 0.001
    lamda = 0.1

    # Mini-batch gradient descent
    for i in range(epochs):
        for j in range(int(data_train.shape[1] / batch_size)):
            start = j * batch_size
            end = (j + 1) * batch_size
            X_batch = data_train[:, start:end]
            Y_batch = labels_train[:, start:end]

            # Do forward pass
            p = evaluate_classifier(X_batch, weight, bias)

            # Do backward pass
            grad_w, grad_b = compute_gradients(
                X_batch, Y_batch, p, weight, lamda)

            # Update the weights and biases
            weight -= learning_rate * grad_w
            bias -= learning_rate * grad_b

        # Compute the cost and accuracy on the training and validation set
        train_cost = compute_cost(data_train, labels_train, weight, bias, lamda)
        train_acc = compute_accuracy(data_train, labels_train, weight, bias)
        val_cost = compute_cost(data_val, labels_val, weight, bias, lamda)
        val_acc = compute_accuracy(data_val, labels_val, weight, bias)

        print("Epoch: %d, Train Cost: %f, Train Acc: %f, Val Cost: %f, Val Acc: %f" % (
            i, train_cost, train_acc, val_cost, val_acc))
        

