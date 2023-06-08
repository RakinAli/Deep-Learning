# Imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys 
from tqdm import tqdm

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

def softmax(s):
  return np.exp(s) / np.sum(np.exp(s), axis=0) 

def get_scores(data,weight, bias):
  return np.dot(weight, data) + bias

def relu(s):
  return np.maximum(0, s)

def forward_pass(data, weights, bias, gamma=None, beta= None, mean = None, var = None, do_batchNorm = False):
  """@docstring:
  Forward pass of the network
  Returns:
  - layers: A list of length L+1 containing the layers of the network. The last element of the list is the output layer.
  - scores_list: A list of length L containing the scores of each layer. The last element of the list is the scores of the output layer."""
  
  # Regular forward pass
  if not do_batchNorm:
    layers = list() # List of layers
    scores_list = list() # List of scores for each layer
    layers.append(np.copy(data))
    for i in range(len(weights)-1):
      scores_list.append(get_scores(layers[-1], weights[i], bias[i])) # Get the scores and append them to the list
      layers.append(relu(scores_list[-1])) # Get the relu of the scores and append them to the list
    scores_list.append(get_scores(layers[-1], weights[-1], bias[-1])) 
    layers.append(softmax(scores_list[-1])) # Softmax of the last layer and append it to the list
  else:
    print("Not implemented yet")
  return layers, scores_list
  



def back_pass(data, labels, weights, reg, softmax, scores, s_hat, gamma= None, mean = None, var = None, do_batchNorm = False):
  """
  Edit: This part of the code was heavily inspired by other students who sat with me and 
  helped me debug. I tried like in previously from assignemnt 2 to use -i iteration and iterate through data
  however since in my forward pass I do a softmax at the end, it made it slighly more difficult for me. I got lazy 
  and got help to try this way. I understand the code and everything else but want to point this out in the case that 
  it raises any concerns.
  """
  weights_gradients = list()
  bias_gradients = list()
  if not do_batchNorm:
    # Last layer
    g = -(labels-softmax)

    # The rest of the layers:
    for i in reversed(range(len(weights))):
      weights_gradients.append(((g @ data[i].T) /data[0].shape[1]) + 2 * reg * weights[i])
      bias_gradients.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
      g = weights[i].T @ g
      ind = np.copy(data[i])
      ind[ind>0] = 1 
      g = g * ind 
    weights_gradients.reverse(), bias_gradients.reverse()

    

    return weights_gradients, bias_gradients

def get_loss(data,labels,weights,bias,reg, probs):
  loss_log = - np.log(np.sum(labels * probs, axis=0)) # dim = (N, 1)
  loss = np.sum(loss_log)/data.shape[1] 

  return loss

def compute_accuracy(data, labels, weights, bias, gamma=None, beta=None, mean=None, var=None, do_batchNorm=False):
  probs = forward_pass(data, weights, bias, gamma, beta, mean, var, do_batchNorm=False)[0][-1] # Softmax of the last layer
  predictions = np.argmax(probs, axis=0)
  labels = np.argmax(labels, axis=0)
  total_correct = np.sum(predictions == labels) / len(labels)
  return total_correct

def compute_cost(data, labels, weights, bias, reg, probs):
  # The loss function
  loss = get_loss(data, labels, weights, bias, reg, probs)
  # The regularisation term
  reg_cost = reg * np.sum([np.sum(np.square(w)) for w in weights]) # L2 regularisation

  return loss + reg_cost

def init_network(data, hidden_layers, he = False, Sigma = None):
  weights = list()
  bias = list()

  if he:
    number = 2
  else: 
    number = 1

  # This is the first layer
  if Sigma is not None:
    print("Not implemented yet")
  else:
    weights.append(np.random.normal(0, np.sqrt(number / data.shape[0]),(hidden_layers[0], data.shape[0])))  # Dim: m x d

  bias.append(np.zeros((hidden_layers[0], 1))) # Dim: m x 1

  # This is the hidden layers
  for i in range(1, len(hidden_layers)):
    if Sigma is not None:
      print("Not implemented yet")
    else:
      weights.append(np.random.normal(0, np.sqrt(number / hidden_layers[i-1]),(hidden_layers[i], hidden_layers[i-1])))
    bias.append(np.zeros((hidden_layers[i], 1)))

  return weights, bias 

def cyclical_update(current_iteration, half_cycle, min_learning, max_learning):
    #One completed cycle is 2 * half_cycle iterations
    current_cycle = int(current_iteration / (2 * half_cycle))  

    # If the current iteration is in the first half of the cycle, the learning rate is increasing
    if 2 * current_cycle * half_cycle <= current_iteration <= (2 * current_cycle + 1) * half_cycle:
        return min_learning + ((current_iteration - 2 * current_cycle * half_cycle) / half_cycle) * (max_learning - min_learning)
    
    # If the current iteration is in the second half of the cycle, the learning rate is decreasing
    if (2 * current_cycle + 1) * half_cycle <= current_iteration <= 2 * (current_cycle + 1) * half_cycle:
        return max_learning - (current_iteration - (2 * current_cycle + 1) * half_cycle) / half_cycle * (max_learning - min_learning)


def update_weights_bias(weights, bias, grad_weights, grad_bias, learning_rate):
  for i in range(len(weights)):
    weights[i] = weights[i] - learning_rate * grad_weights[i]
    bias[i] = bias[i] - learning_rate * grad_bias[i]
  return weights, bias

def sgd_minibatch(data_train, data_val, data_test, weights, bias, labels_train, labels_val, labels_test, learning_rate, reguliser, batch_size, cycles, do_plot = False, do_batchNorm = False, name_of_file=""):
  eta_min = 1e-5
  eta_max = 1e-1
  step_size = (data_train.shape[1] / batch_size)*2
  total_updates = 2 * step_size * cycles
  epochs = int(np.ceil(total_updates / (data_train.shape[1] / batch_size)))
  updates_per_epoch = int((data_train.shape[1] / batch_size))  # Number of updates per epoch
  total_iterations = epochs * updates_per_epoch

  epochs = int(total_iterations / updates_per_epoch)


  # For plotting
  train_loss = list()
  vaidation_loss = list()
  test_loss = list()
  # For plotting
  train_accuracy = list()
  validation_accuracy = list()
  test_accuracy = list()
  # For plotting
  train_cost = list()
  validation_cost = list()
  test_cost = list()
  step_list = list()
  print("Epochs: ", epochs)
  print("Updates per epoch: ", updates_per_epoch)
  for epoch in tqdm(range(epochs)):
    for batch in range(updates_per_epoch):
      start = batch * batch_size
      end = (batch+1) * batch_size
      data_batch = data_train[:, start:end]
      labels_batch = labels_train[:, start:end]
 
      # Relued layers and scores
      layers, scores_list = forward_pass(data_batch, weights, bias)

      # Backpropagation
      grad_weights, grad_bias = back_pass(layers, labels_batch, weights, 0, layers[-1], scores_list[-1], None, None, None, None, False)
    
      # Update the weights and bias
      weights, bias = update_weights_bias(weights, bias, grad_weights, grad_bias, learning_rate)
  
      # Update the learning rate
      current_iteration = epoch * updates_per_epoch + batch
      learning_rate = cyclical_update(current_iteration, step_size, eta_min, eta_max)
       
    if do_plot:
      train_loss.append(get_loss(data_train, labels_train, weights, bias, 0, forward_pass(data_train, weights, bias)[0][-1]))
      vaidation_loss.append(get_loss(data_val, labels_val, weights, bias, 0, forward_pass(data_val, weights, bias)[0][-1]))
      test_loss.append(get_loss(data_test, labels_test, weights, bias, 0, forward_pass(data_test, weights, bias)[0][-1]))

      train_accuracy.append(compute_accuracy(data_train, labels_train, weights, bias))
      validation_accuracy.append(compute_accuracy(data_val, labels_val, weights, bias))
      test_accuracy.append(compute_accuracy(data_test, labels_test, weights, bias))

      train_cost.append(compute_cost(data_train, labels_train, weights, bias, 0, forward_pass(data_train, weights, bias)[0][-1]))
      validation_cost.append(compute_cost(data_val, labels_val, weights, bias, 0, forward_pass(data_val, weights, bias)[0][-1]))
      test_cost.append(compute_cost(data_test, labels_test, weights, bias, 0, forward_pass(data_test, weights, bias)[0][-1]))

      step_list.append(epoch)

  if do_plot:
    do_plotting(train_loss, vaidation_loss, test_loss, train_accuracy, validation_accuracy, test_accuracy, train_cost, validation_cost, test_cost, step_list, name_of_file=name_of_file)

  return weights, bias

def do_plotting(train_loss, vaidation_loss, test_loss, train_accuracy, validation_accuracy, test_accuracy, train_cost, validation_cost, test_cost, step_list,name_of_file=""):
  # Plotting
  plt.figure(figsize=(20, 10))
  plt.subplot(2, 2, 1)
  plt.plot(step_list, train_accuracy, label="Training accuracy")
  plt.plot(step_list, validation_accuracy, label="Validation accuracy")
  plt.plot(step_list, test_accuracy, label="Test accuracy")
  plt.title("Accuracy")
  plt.legend()
  plt.subplot(2, 2, 2)
  plt.plot(step_list, train_loss, label="Training loss")
  plt.plot(step_list, vaidation_loss, label="Validation loss")
  plt.plot(step_list, test_loss, label="Test loss")
  plt.title("Loss")
  plt.legend()
  plt.subplot(2, 2, 3)
  plt.plot(step_list, train_cost, label="Training cost")
  plt.plot(step_list, validation_cost, label="Validation cost")
  plt.plot(step_list, test_cost, label="Test cost")
  plt.title("Cost")
  plt.legend()
  plt.savefig("Results_pics/" + name_of_file + ".png")
  plt.show()  


def main():
  #Getting started
  data_train, labels_train, data_val, labels_val, data_test, labels_test, labels = read_data(5000)

  # Preprocessing
  data_train, data_val, data_test = normalise_all(data_train, data_val, data_test)
  labels_train, labels_val, labels_test = encode_all(labels_train, labels_val, labels_test)

  # Initialising the network
  weights, bias = init_network(data_train, [50,50,10], he = False)

  # Get the softmax of the last layer
  probs = forward_pass(data_train, weights, bias)[0][-1]

  # sgd_minibatch
  weights, bias = sgd_minibatch(data_train, data_val, data_test, weights, bias, labels_train, labels_val, labels_test, 0.01, 0.001, 512, 2, do_plot = True, do_batchNorm = False, name_of_file="SGD_minibatch")


"""
  # Get accuracy
  print("Shape of the weights:" , weights[0].shape)
  print("Shape of the bias:" , bias[0].shape)
  print("Shape of the data:" , data_train.shape)
  accuracy = compute_accuracy(data_train, labels_train, weights, bias)
  print("Accuracy 1: ", accuracy)
  # get the loss
  probs = forward_pass(data_train, weights, bias)[0][-1]
  loss = get_loss(data_train, labels_train, weights, bias, 0, probs)
  print("Loss 1: ", loss)

  # Forward pass
  layers, scores_list = forward_pass(data_train, weights, bias)

  # Backward pass
  grad_weights, grad_bias = back_pass(layers, labels_train, weights, 0, layers[-1], scores_list[-1], None, None, None, None, False)
  print("done")
  """

"""

  # Update the weights and bias
  for i in range(len(weights)):
    weights[i] = weights[i] - 0.1 * grad_weights[i]
    bias[i] = bias[i] - 0.1 * grad_bias[i]
  
  print("shape of the weights:", weights[0].shape)
  print("shape of the bias:", bias[0].shape)
  print("shape of the data:", data_train.shape)
  
  # Get accuracy
  accuracy = compute_accuracy(data_train, labels_train, weights, bias)
  print("Accuracy 2: ", accuracy)
  # get the loss
  probs = forward_pass(data_train, weights, bias)[0][-1]
  loss = get_loss(data_train, labels_train, weights, bias, 0, probs)
  print("Loss 2: ", loss)

  # Do forward and backward pass again
  layers, scores_list = forward_pass(data_train, weights, bias)
  grad_weights, grad_bias = back_pass(layers, labels_train, weights, 0, layers[-1], scores_list[-1], None, None, None, None, False)

  # Update the weights and bias
  for i in range(len(weights)):
    weights[i] = weights[i] - 0.1 * grad_weights[i]
    bias[i] = bias[i] - 0.1 * grad_bias[i]

  # Get accuracy
  accuracy = compute_accuracy(data_train, labels_train, weights, bias)
  print("Accuracy 3: ", accuracy)
  # get the loss
  probs = forward_pass(data_train, weights, bias)[0][-1]
  loss = get_loss(data_train, labels_train, weights, bias, 0, probs)
  print("Loss 3: ", loss)

  """
  
if __name__ == "__main__":
  main()