# Imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys 

# Global variables
PATH = "Datasets/cifar-10-batches-py/"
batches= ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

def LoadBatch(filename):
  with open(filename, 'rb') as fo:
      dict = pickle.load(fo, encoding='latin1')
  return dict


def read_data(size_of_validation=5000):
  """@docstring:
  This function will help you get started on the CIFAR-10 dataset.
  Reads all the batches to do training and validation.
  For validation, picks random 5000 images from any of the batches.
  Reads the test_batch to do testing
  Returns:
  - data_train: A numpy array of shape (3072, 45000) containing the training data.
  - labels_train: A numpy array of shape (45000,) containing the training labels.
  - data_val: A numpy array of shape (3072, 5000) containing the validation data. 
  - labels_val: A numpy array of shape (5000,) containing the validation labels.
  - data_test: A numpy array of shape (3072, 10000) containing the test data.
  - labels_test: A numpy array of shape (10000,) containing the test labels.
  - labels: A list of length 10 containing the names of the classes.
  """
  # Read the training data for all batches
  for i in range(len(batches)):
    if i == 0:
      batch = LoadBatch(PATH + batches[i])
      data_train = batch['data']
      labels_train = batch['labels']
    else:
      batch = LoadBatch(PATH + batches[i])
      # Stack the data vertically
      data_train = np.vstack((data_train, batch['data']))
      # Stack the labels horizontally
      labels_train = np.hstack((labels_train, batch['labels']))

  # Read the test data
  batch = LoadBatch(PATH + 'test_batch')
  data_test = batch['data']
  labels_test = batch['labels']

  # Create the validation data
  random_indices = np.random.choice(data_train.shape[0], size_of_validation, replace=False)
  data_val = data_train[random_indices]
  labels_val = labels_train[random_indices]
  # Delete the validation data from the training data
  data_train = np.delete(data_train, random_indices, axis=0)
  # Delete the validation labels from the training labels
  labels_train = np.delete(labels_train, random_indices, axis=0)

  # Grabbing the labels names
  labels = LoadBatch(PATH + 'batches.meta')['label_names']

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
  """@docstring:
  Normalise the data
  Returns:
  - data_train_norm: A numpy array of shape (D, N) containing the normalised training data.
  - data_val_norm: A numpy array of shape (D, N) containing the normalised validation data.
  - data_test_norm: A numpy array of shape (D, N) containing the normalised test data.
  """
  # Normalise the data
  data_train_norm, _, _ = normalise(data_train, None, None)
  data_val_norm, _, _ = normalise(data_val, None, None)
  data_test_norm, _, _ = normalise(data_test, None, None)
  return data_train_norm, data_val_norm, data_test_norm,


def init_network(data, nodes_in_layer, layers=3, he=False, sigma=None):
  if len(nodes_in_layer) != layers:
    # System exit
    sys.exit("The number of layers does not match the number of nodes in each layer.")

  weights = list()
  biases = list()
  gamma = list()
  beta = list()

  if he:
    number = 2
  else:
    number = 1

  # First layer --> Dim = D is the data dimension and N is the number of nodes in the first layer
  if sigma is not None:
    # shape (N, D)
    weights.append(np.random.normal(0, sigma, (nodes_in_layer[0], data.shape[0])))
  else:
    # shape: (N, D)
    weights.append(np.random.normal(0, np.sqrt(number / data.shape[0]), (nodes_in_layer[0], data.shape[0])))
  
  biases.append(np.zeros((nodes_in_layer[0], 1)))

  for i in range(1, layers):
    if sigma is not None:
      weights.append(np.random.normal(0, sigma, (nodes_in_layer[i], weights[-1].shape[0])))
    else:
      weights.append(np.random.normal(0, np.sqrt(number / weights[-1].shape[0]), (nodes_in_layer[i], weights[-1].shape[0])))
    biases.append(np.zeros((nodes_in_layer[i], 1)))
  
  for nodes in nodes_in_layer[:-1]:
    gamma.append(np.ones((nodes, 1)))
    beta.append(np.zeros((nodes, 1)))

  return weights, biases, gamma, beta

def softmax(s):
  return np.exp(s) / np.sum(np.exp(s), axis=0) 

def get_scores(data,weight, bias):
  return np.dot(weight, data) + bias

def relu(s):
  return np.maximum(0, s)

def forward_pass(data, weights, bias, gamma=None, beta= None, mean = None, var = None, do_batchNorm = False):
  if not do_batchNorm:
    layers = list()
    scores_list = list()
    layers.append(np.copy(data))

    for i in range(len(weights)-1):
      scores_list.append(get_scores(layers[-1], weights[i], bias[i]))
      layers.append(relu(scores_list[-1]))
    scores_list.append(get_scores(layers[-1], weights[-1], bias[-1])) # Scores of the last layer
    layers.append(softmax(scores_list[-1])) # Softmax of the last layer 
    print("Shape of the last layer: ", layers[-1].shape)
    return layers, scores_list

def back_pass(data, labels, weights, reg, softmax, scores, s_hat, gamma= None, mean = None, var = None, do_batchNorm = False):
  grad_weights = list()
  grad_bias = list()

  if not do_batchNorm:
    # Last layer
    g = - (labels - softmax) # (K, N)

    for i in reversed(range(len(weights))):
      grad_weights.append(g @ data[i].T / data[i].shape[1] + 2 * reg * weights[i])
      grad_bias.append(np.sum(g, axis=1, keepdims=True) / data[0].shape[1])     
      g = weights[i].T @ g
      ind = np.copy(data[i])
      ind[ind > 0] = 1 # Relu derivative
      g = g * ind
    
    # Reverse the lists
    grad_weights.reverse() 
    grad_bias.reverse()
    return grad_weights, grad_bias


  else:
    print("Not implemented yet")

  



def main():
  #Getting started
  data_train, labels_train, data_val, labels_val, data_test, labels_test, labels = read_data(5000)

  # Preprocessing
  data_train, data_val, data_test = normalise_all(data_train, data_val, data_test)
  labels_train, labels_val, labels_test = encode_all(labels_train, labels_val, labels_test)

  # Initialise the network
  weights, biases, _, _ = init_network(data_train, [50, 50, 10], layers=3, he=False, sigma=0.0) 
  print("Completed the initialisation of the network")

  # Forward pass
  data, scores_list = forward_pass(data_train, weights, biases, do_batchNorm=False)
  print("Completed the forward pass")

  # Backward pass
  grad_weights, grad_bias = back_pass(data, labels_train, weights, 0.0, data[-1], scores_list[-1], None, None, None, None, do_batchNorm=False)
  print("Completed the backward pass")



  
if __name__ == "__main__":
  main()