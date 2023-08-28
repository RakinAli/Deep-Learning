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

def batch_normalize(score, mean, variance):
  # Taken straight from the lecture notes <-- I wouldn't have written it like this
  norm = np.diag(pow(variance + np.finfo(float).eps, -1 / 2)) @ (score - mean[:, np.newaxis])
  return norm

def batch_normalised_backpass(g, s, mean, var):
  # Taken from the assignment notes
  sigma1 = np.power(var + np.finfo(float).eps, -1 / 2).T[:, np.newaxis]
  sigma2 = np.power(var + np.finfo(float).eps, -3 / 2).T[:, np.newaxis]
  G1 = g * sigma1
  G2 = g * sigma2
  D = s - mean[:, np.newaxis]
  c = np.sum(G2 * D, axis=1)[:, np.newaxis]
  gradient_batch = G1 - (1 / D.shape[1]) * np.sum(G1, axis=1)[:, np.newaxis] - (1 / g.shape[1]) * D * c
  return gradient_batch

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
    layers = list() # List of layers
    scores_list = list() # List of scores for each layer
    batch_normalised_scores = list()  # List of batch normalised scores for each layer
    mean_list = list()  # List of average scores for each layer
    variance_list = list()  # List of variances for each layer
    # Data is the first layer
    layers.append(np.copy(data)) 
    for i in range(len(weights)-1):
      # Get the scores and append them to the list
      scores_list.append(get_scores(layers[-1], weights[i], bias[i]))
      # Get the mean and variance of the scores and append them to the list
      if mean is None and var is None:
        mean_list.append(np.mean(scores_list[-1], axis=1, dtype=np.float64))
        variance_list.append(np.var(scores_list[-1], axis=1, dtype=np.float64))
      else:
        mean_list.append(mean[i])
        variance_list.append(var[i])
      # Get the batch normalised scores and append them to the list
      batch_normalised_scores.append(batch_normalize(scores_list[-1], mean_list[-1], variance_list[-1]))
      s_tilde = gamma[i] * batch_normalised_scores[-1] + beta[i]
      layers.append(relu(s_tilde)) # Get the relu of the scores and append them to the list
    # Get the scores and append them to the list
    scores_list.append(get_scores(layers[-1], weights[-1], bias[-1]))
    layers.append(softmax(scores_list[-1])) # Softmax of the last layer and append it to the list   
  return layers, scores_list,batch_normalised_scores, mean_list, variance_list, 
  

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
    return weights_gradients, bias_gradients, None, None
  else:
    gradient_gamma = list()
    gradient_beta = list()
    # Last layer
    g = -(labels-softmax)
    weights_gradients.append(((g @ data[-2].T) /data[0].shape[1]) + 2 * reg * weights[-1]) # Derivative of the loss with respect to the weights
    bias_gradients.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1]) # Bias gradient
    g = weights[-1].T @ g # Derivative of the loss with respect to the scores
    ind = np.copy(data[-2]) # 
    ind[ind>0] = 1 # Relu derivative
    g = g * ind # Derivative of the loss with respect to the scores
    # The rest of the layers:
    for i in reversed(range(len(weights)-1)):
      gradient_gamma.append(np.sum(g * s_hat[i], axis=1)[:, np.newaxis] / data[0].shape[1])
      gradient_beta.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
      g = gamma[i] * g
      g = batch_normalised_backpass(g, scores[i], mean[i], var[i])

      weights_gradients.append(((g @ data[i].T) /data[0].shape[1]) + 2 * reg * weights[i]) # Derivative of the loss with respect to the weights
      bias_gradients.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1]) # Bias gradient
      if i>0:
        g = weights[i].T @ g # Previous layer 
        ind = np.copy(data[i]) #
        ind[ind>0] = 1 # Relu derivative
        g = g * ind # Derivative of the loss with respect to the scores

    weights_gradients.reverse(), bias_gradients.reverse(), gradient_gamma.reverse(), gradient_beta.reverse()
    return weights_gradients, bias_gradients, gradient_gamma, gradient_beta
    

def get_loss(data,labels,weights,bias,reg, probs):
  loss_log = - np.log(np.sum(labels * probs, axis=0)) # dim = (N, 1)
  loss = np.sum(loss_log)/data.shape[1] 

  return loss

def compute_accuracy(data, labels, weights, bias, gamma=None, beta=None, mean=None, var=None, do_batchNorm=False):
  probs = forward_pass(data, weights, bias, gamma, beta, mean, var, do_batchNorm=do_batchNorm)[0][-1] # Softmax of the last layer
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
    weights.append(np.random.normal(0, Sigma,(hidden_layers[0], data.shape[0])))  # Dim: m x d
  else:
    weights.append(np.random.normal(0, np.sqrt(number / data.shape[0]),(hidden_layers[0], data.shape[0])))  # Dim: m x d

  bias.append(np.zeros((hidden_layers[0], 1))) # Dim: m x 1

  # This is the hidden layers
  for i in range(1, len(hidden_layers)):
    if Sigma is not None:
      weights.append(np.random.normal(0, Sigma,(hidden_layers[i], weights[-1].shape[0])))  # Dim: l x m
    else:
      weights.append(np.random.normal(0, np.sqrt(number / hidden_layers[i-1]),(hidden_layers[i], hidden_layers[i-1])))
    bias.append(np.zeros((hidden_layers[i], 1)))

    # Generating the gamma and beta for batch normalisation
    gamma = list()
    beta = list()
    for i in range(len(hidden_layers)-1):
      gamma.append(np.ones((hidden_layers[i], 1)))
      beta.append(np.zeros((hidden_layers[i], 1)))  

  return weights, bias, gamma, beta

def cyclical_update(current_iteration, half_cycle, min_learning, max_learning):
    #One completed cycle is 2 * half_cycle iterations
    current_cycle = int(current_iteration / (2 * half_cycle))  

    # If the current iteration is in the first half of the cycle, the learning rate is increasing
    if 2 * current_cycle * half_cycle <= current_iteration <= (2 * current_cycle + 1) * half_cycle:
        return min_learning + ((current_iteration - 2 * current_cycle * half_cycle) / half_cycle) * (max_learning - min_learning)
    
    # If the current iteration is in the second half of the cycle, the learning rate is decreasing
    if (2 * current_cycle + 1) * half_cycle <= current_iteration <= 2 * (current_cycle + 1) * half_cycle:
        return max_learning - (current_iteration - (2 * current_cycle + 1) * half_cycle) / half_cycle * (max_learning - min_learning)


def update_weights_bias(weights, bias, grad_weights, grad_bias, learning_rate, gamma, beta, grad_gamma, grad_beta):
  for i in range(len(weights)):
    weights[i] = weights[i] - learning_rate * grad_weights[i]
    bias[i] = bias[i] - learning_rate * grad_bias[i]

  # Update gamma and beta
  for i in range(len(grad_gamma)):
    gamma[i] = gamma[i] - learning_rate * grad_gamma[i]
    beta[i] = beta[i] - learning_rate * grad_beta[i]
  return weights, bias, gamma, beta

def sgd_minibatch(data_train, data_val, data_test, weights, bias, labels_train, labels_val, labels_test, learning_rate, reguliser, batch_size, cycles,gamma, beta, do_plot = False, do_batchNorm = False, name_of_file=""):
  eta_min = 1e-5
  eta_max = 1e-1
  step_size = (data_train.shape[1] / batch_size)*5
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
      layers, scores_list, scores_normalised ,mean_list, variance_list = forward_pass(data_batch, weights, bias, do_batchNorm=do_batchNorm, gamma=gamma, beta=beta)

      # Backpropagation
      grad_weights, grad_bias, grad_gamma, grad_beta = back_pass(layers, labels_batch, weights, reguliser, layers[-1], scores_list[-1], scores_normalised, gamma, mean_list, variance_list, do_batchNorm=do_batchNorm)
    
      # Update the weights and bias
      weights, bias,gamma,beta  = update_weights_bias(weights, bias, grad_weights, grad_bias, learning_rate, gamma, beta, grad_gamma, grad_beta)
  
      # Update the learning rate
      current_iteration = epoch * updates_per_epoch + batch
      learning_rate = cyclical_update(current_iteration, step_size, eta_min, eta_max)
       
    if do_plot and not do_batchNorm:
      train_loss.append(get_loss(data_train, labels_train, weights, bias, reguliser, forward_pass(data_train, weights, bias, do_batchNorm=do_batchNorm)[0][-1]))
      vaidation_loss.append(get_loss(data_val, labels_val, weights, bias, reguliser, forward_pass(data_val, weights, bias,do_batchNorm=do_batchNorm)[0][-1]))
      test_loss.append(get_loss(data_test, labels_test, weights, bias, reguliser, forward_pass(data_test, weights, bias,do_batchNorm=do_batchNorm)[0][-1]))

      train_accuracy.append(compute_accuracy(data_train, labels_train, weights, bias))
      validation_accuracy.append(compute_accuracy(data_val, labels_val, weights, bias))
      test_accuracy.append(compute_accuracy(data_test, labels_test, weights, bias))

      train_cost.append(compute_cost(data_train, labels_train, weights, bias, reguliser, forward_pass(data_train, weights, bias)[0][-1]))
      validation_cost.append(compute_cost(data_val, labels_val, weights, bias, reguliser, forward_pass(data_val, weights, bias)[0][-1]))
      test_cost.append(compute_cost(data_test, labels_test, weights, bias, reguliser, forward_pass(data_test, weights, bias)[0][-1]))
      step_list.append(epoch)

    elif do_batchNorm and do_plot:
      train_loss.append(get_loss(data_train, labels_train, weights, bias, reguliser, forward_pass(data_train, weights, bias, gamma, beta, mean_list, variance_list, do_batchNorm=do_batchNorm)[0][-1]))
      vaidation_loss.append(get_loss(data_val, labels_val, weights, bias, reguliser, forward_pass(data_val, weights, bias, gamma, beta, mean_list, variance_list, do_batchNorm=do_batchNorm)[0][-1]))
      test_loss.append(get_loss(data_test, labels_test, weights, bias, reguliser, forward_pass(data_test, weights, bias, gamma, beta, mean_list, variance_list, do_batchNorm=do_batchNorm)[0][-1]))

      train_accuracy.append(compute_accuracy(data_train, labels_train, weights, bias, gamma, beta, mean_list, variance_list, do_batchNorm=do_batchNorm))
      validation_accuracy.append(compute_accuracy(data_val, labels_val, weights, bias, gamma, beta, mean_list, variance_list, do_batchNorm=do_batchNorm))
      test_accuracy.append(compute_accuracy(data_test, labels_test, weights, bias, gamma, beta, mean_list, variance_list, do_batchNorm=do_batchNorm))

      train_cost.append(compute_cost(data_train, labels_train, weights, bias, reguliser, forward_pass(data_train, weights, bias, gamma, beta, mean_list, variance_list, do_batchNorm=do_batchNorm)[0][-1]))
      validation_cost.append(compute_cost(data_val, labels_val, weights, bias, reguliser, forward_pass(data_val, weights, bias, gamma, beta, mean_list, variance_list, do_batchNorm=do_batchNorm)[0][-1]))
      test_cost.append(compute_cost(data_test, labels_test, weights, bias, reguliser, forward_pass(data_test, weights, bias, gamma, beta, mean_list, variance_list, do_batchNorm=do_batchNorm)[0][-1]))
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
  weights, bias, gamma,beta = init_network(data_train, [50,50,10], he = False)

  print("Size of the data_train: ", data_train.shape)
 
  config = {
    'data_train': data_train,
    'data_val': data_val,
    'data_test': data_test,
    'weights': weights,
    'bias': bias,
    'labels_train': labels_train,
    'labels_val': labels_val,
    'labels_test': labels_test,
    'learning_rate': 0.01,
    'reguliser': 0.005,
    'batch_size': 100,
    'cycles': 2,
    'do_plot': True,
    'do_batchNorm': True,
    'name_of_file': "SGD_minibatch",
    'gamma': gamma,
    'beta': beta
}
  
  weights1, bias1, = sgd_minibatch(**config)


if __name__ == "__main__":
  main()