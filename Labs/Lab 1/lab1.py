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

def evaluate_classifier(data,weight,bias):
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

def cross_(data,labels,weight,bias):
   
    
  

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
  #print(np.shape(labels_train))

  # Random weight and bias initialisation
  weight, bias = random_weight_bias_init(data_train, labels_names)
  #print("Weight shape:",weight.shape) # (10, 3072)
  #print("Bias shape", bias.shape)  # (10, 1)

  # Evaluating the classifier
  probabilities = evaluate_classifier(data_train, weight, bias)
  print("Shape of p:", probabilities.shape)  # (10, 10000)






