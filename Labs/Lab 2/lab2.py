""" 
  Author: Rakin Ali
  Date:    27/03/2022
  Description: Lab 1 - Backpropagation
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
from prettytable import PrettyTable


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
    batch_0 = LoadBatch("Datasets/cifar-10-batches-py/" + 'data_batch_1')
    # print(data.keys())
    data_train = batch_0['data']  # Grabbing the data
    labels_train = batch_0['labels']  # Grabbing the labels

    # Read validation data from Batch 1
    # Loading the batch_1 as a dictionary
    batch_1 = LoadBatch("Datasets/cifar-10-batches-py/" + 'data_batch_2')
    # print(data.keys())
    data_val = batch_1['data']  # Grabbing the data
    labels_val = batch_1['labels']  # Grabbing the labels

    # Read test data from test_batch
    # Loading the test_batch as a dictionary
    batch_test = LoadBatch("Datasets/cifar-10-batches-py/" + 'test_batch')
    # print(data.keys())
    data_test = batch_test['data']  # Grabbing the data
    labels_test = batch_test['labels']  # Grabbing the labels

    # Grading the labels
    labels = LoadBatch("Datasets/cifar-10-batches-py/" +
                       'batches.meta')['label_names']
    return data_train, labels_train, data_val, labels_val, data_test, labels_test, labels


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
    data = np.float64(data)  # Required for division to work
    if mean is None:
        mean = np.mean(data, axis=0)  # Mean of each column
    if std is None:
        std = np.std(data, axis=0)  # Standard deviation of each column
    data = (data - mean) / std
    data = np.transpose(data)  # Transpose the data to get the shape (D, N)
    return data, mean, std


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


def encode_all(labels_train, labels_val, labels_test):
    """@docstring:
    One hot encoding the labels
    Returns:
    - labels_train: A numpy array of<< shape (K, N) containing the one-hot encoded labels for the training data.
    - labels_val: A numpy array of shape (K, N) containing the one-hot encoded labels for the validation data.
    - labels_test: A numpy array of shape (K, N) containing the one-hot encoded labels for the test data.
    """
    # One hot encoding --> One hot encoding the labels
    labels_train = one_hot_encoding(labels_train, 10)  # k x N matrix
    labels_val = one_hot_encoding(labels_val, 10)  # k x N matrix
    labels_test = one_hot_encoding(labels_test, 10)  # k x N matrix
    return labels_train, labels_val, labels_test


def normalise_all(data_train, data_val, data_test):
    data_train_norm, mean, std = normalise(data_train, None, None)
    data_val_norm, _, _ = normalise(data_val, mean, std)
    data_test_norm, _, _ = normalise(data_test, mean, std)
    return data_train_norm, data_val_norm, data_test_norm,

def init_weights_bias(data_train, labels, hidden_nodes=50):
    """@docstring:
    Initialize the weights and bias
    Returns:
    - W : a list of numpy arrays containing the weights
    - b : a list of numpy arrays containing the bias
    """
    weight = list()
    bias = list()

    # First layer dim = (m x d)
    weight.append(np.random.normal(0,1/np.sqrt(data_train.shape[0]), (hidden_nodes, data_train.shape[0])))
    # Second layer dim = (k x m)
    weight.append(np.random.normal(0,1/np.sqrt(hidden_nodes), (labels.shape[0], hidden_nodes)))

    # First layer dim = (m x 1)
    bias.append(np.zeros((hidden_nodes, 1)))
    # Second layer dim = (k x 1)
    bias.append(np.zeros((labels.shape[0], 1)))

    return weight, bias



if __name__ == '__main__':
    # Getting started
    data_train, labels_train, data_val, labels_val, data_test, labels_test, labels = getting_started()
    #print("Shape of the training data: ", data_train.shape)

    # Normalising the data
    data_train_norm, data_val_norm, data_test_norm = normalise_all( data_train, data_val, data_test) 

    # One hot encoding the labels
    labels_train, labels_val, labels_test = encode_all(labels_train, labels_val, labels_test)

    # Initializing the weights and bias
    weights, bias = init_weights_bias(data_train_norm, labels_train, hidden_nodes=50)
    
