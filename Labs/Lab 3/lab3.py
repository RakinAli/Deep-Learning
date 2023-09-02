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
  return np.diag(pow(variance + np.finfo(float).eps, -1 / 2)) @ (score - mean[:, np.newaxis])

def forward_pass(data, weights, bias, gamma=None, beta= None, mean = None, var = None, do_batchNorm = False):
  """@docstring:
  return layers, scores_list, s_hat, mean_list, variance_list
  """
  # Regular forward pass
  layers = list() # List of layers
  scores_list = list() # List of scores for each layer
  s_hat = list()
  mean_list = list()
  variance_list = list()
  layers.append(np.copy(data))

  if not do_batchNorm:
    for i in range(len(weights)-1):
      scores_list.append(get_scores(layers[-1], weights[i], bias[i])) # Get the scores and append them to the list
      layers.append(relu(scores_list[-1])) # Get the relu of the scores and append them to the list
    scores_list.append(get_scores(layers[-1], weights[-1], bias[-1])) 
    layers.append(softmax(scores_list[-1])) # Softmax of the last layer and append it to the list
    return layers, scores_list, None, None, None 
  else:
     # Do batch normalisation
    for i in range(len(weights)-1):
      score = get_scores(layers[-1],weights[i],bias[i])
      scores_list.append(score)
      if mean is None and var is None:
        # Calculate Mean
        mean_append = np.mean(scores_list[-1],axis=1, dtype=np.float64)
        variance_append = np.var(scores_list[-1],axis=1, dtype=np.float64)
        mean_list.append(mean_append)
        variance_list.append(variance_append)
      else:
        mean_append = mean[i]
        variance_append = var[i]
        mean_list.append(mean_append)
        variance_list.append(variance_append)
        # Batch normalisation
      batch_normalize_scores = batch_normalize(scores_list[-1], mean_list[-1], variance_list[-1])   
      s_hat.append(batch_normalize_scores)
      s_tilde = gamma[i] * s_hat[-1] + beta[i] # Scale and shift
      layers.append(relu(s_tilde)) # Get the relu of the scores and append them to the list
    scores_list.append(get_scores(layers[-1], weights[-1], bias[-1]))
    layers.append(softmax(scores_list[-1])) 
    return layers, scores_list, s_hat, mean_list, variance_list

def batch_norm_backpass(g, s, mean, variance):
  sigma_1 = ((variance + np.finfo(np.float64).eps) ** -0.5).T[:, np.newaxis]
  sigma_2 = ((variance + np.finfo(np.float64).eps) ** -1.5).T[:, np.newaxis]
  G_1 = g * sigma_1
  G_2 = g * sigma_2
  D = s - mean[:, np.newaxis]
  c = np.dot(G_2 * D, np.ones((g.shape[1], 1)))
  G_batch = G_1 - (1 / g.shape[1]) * np.dot(G_1, np.ones((g.shape[1], 1))) - ((1 / g.shape[1])) * (D * c)
  return G_batch

def back_pass(data, labels, weights, reg, softmax, scores, s_hat, gamma, mean, var, do_batchNorm = False):
  """@docstring:
  Input:
    data, labels, weights, reg, softmax, scores, s_hat, gamma= None, mean = None, var = None, do_batchNorm = False
  Output:     
    return weights_gradients, bias_gradients, gamma_gradients, beta_gradients
"""
  weights_gradients = list()
  bias_gradients = list()
  gamma_gradients = list()
  beta_gradients = list()
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
    g = -(labels-softmax)
    w_grad = ((g @ data[-2].T) / data[0].shape[1]) + 2 * reg * weights[-1]
    b_grad = np.sum(g,axis=1)[:,np.newaxis]/data[0].shape[1]
    g = weights[-1].T @ g
    ind = np.copy(data[-2])
    ind[ind>0] = 1
    g = g * ind
    # Insert the weights and gradients 
    weights_gradients.append(w_grad)
    bias_gradients.append(b_grad)
    # The rest of the layers:
    for i in reversed(range(len(weights)-1)):
      gamma_deriv = np.sum(g * s_hat[i], axis=1)[:, np.newaxis] / data[0].shape[1]
      beta_deriv = np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1]
      gamma_gradients.append(gamma_deriv)
      beta_gradients.append(beta_deriv)
      g = g * gamma[i]
      g = batch_norm_backpass(g, scores[i], mean[i], var[i])
      # Update the p
      w_grad = ((g @ data[i].T) / data[0].shape[1]) + 2 * reg * weights[i]
      b_grad = np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1]
      weights_gradients.append(w_grad)
      bias_gradients.append(b_grad)
      if i > 0:
        g = weights[i].T @ g
        ind = np.copy(data[i])
        ind[ind>0] = 1
        g = g * ind   
    # Reverse the list to get correct order
    weights_gradients.reverse(), bias_gradients.reverse(), gamma_gradients.reverse(), beta_gradients.reverse()
    return weights_gradients, bias_gradients, gamma_gradients, beta_gradients

def get_loss(data,labels,probs):
  loss_log = - np.log(np.sum(labels * probs, axis=0)) # dim = (N, 1)
  loss = np.sum(loss_log)/data.shape[1] 

  return loss

def compute_accuracy(data, labels, weights, bias, probs):
  predictions = np.argmax(probs, axis=0)
  labels = np.argmax(labels, axis=0)
  total_correct = np.sum(predictions == labels) / len(labels)
  return total_correct

def compute_cost(data, labels, weights, bias, reg, probs):
  # The loss function
  loss = get_loss(data, labels,probs)
  # The regularisation term
  reg_cost = reg * np.sum([np.sum(np.square(w)) for w in weights]) # L2 regularisation

  return loss + reg_cost

def init_network(data, hidden_layers, he = False, Sigma = None, do_batchNorm = False):
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
  if do_batchNorm:
    gamma = list()
    beta = list()
    for nodes in hidden_layers[:-1]:
      gamma.append(np.ones((nodes, 1)))
      beta.append(np.zeros((nodes, 1)))
    return weights, bias, gamma, beta
  else:
    return weights, bias, None, None

def cyclical_update(current_iteration, half_cycle, min_learning, max_learning):
    #One completed cycle is 2 * half_cycle iterations
    current_cycle = int(current_iteration / (2 * half_cycle))  
    # If the current iteration is in the first half of the cycle, the learning rate is increasing
    if 2 * current_cycle * half_cycle <= current_iteration <= (2 * current_cycle + 1) * half_cycle:
        return min_learning + ((current_iteration - 2 * current_cycle * half_cycle) / half_cycle) * (max_learning - min_learning)
    # If the current iteration is in the second half of the cycle, the learning rate is decreasing
    if (2 * current_cycle + 1) * half_cycle <= current_iteration <= 2 * (current_cycle + 1) * half_cycle:
        return max_learning - (current_iteration - (2 * current_cycle + 1) * half_cycle) / half_cycle * (max_learning - min_learning)

def update_weights_bias(weights, bias, gamma, beta, grad_weights, grad_bias, learning_rate, grad_gamma, grad_beta, do_batchNorm = False):
  for i in range(len(weights)):
    weights[i] = weights[i] - learning_rate * grad_weights[i]
    bias[i] = bias[i] - learning_rate * grad_bias[i]
  if do_batchNorm:
    #Update gamma and beta
    for i in range(len(gamma)):
      gamma[i] = gamma[i] - learning_rate * grad_gamma[i]
      beta[i] = beta[i] - learning_rate * grad_beta[i]
    return weights, bias, gamma, beta
  return weights, bias, None, None

def sgd_minibatch(data_train, data_val, data_test, weights, bias, labels_train, labels_val, labels_test, learning_rate, reguliser, batch_size, cycles, do_plot = False, do_batchNorm = False, name_of_file="", gamma=None, beta=None, alpha=0.9):
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
      # Get the batch
      data_batch = data_train[:, start:end]
      labels_batch = labels_train[:, start:end]
 
      # Relued layers and scores
      layers, scores_list, s_hat, mean_list, variance_list = forward_pass(data_batch, weights, bias, do_batchNorm=do_batchNorm, gamma=gamma, beta=beta)

      # Backpropagation
      grad_weights, grad_bias, grad_gamma, grad_beta = back_pass(layers, labels_batch, weights, reguliser, layers[-1], scores_list, s_hat, gamma, mean_list, variance_list, do_batchNorm)
    
      # Update the weights and bias
      weights, bias, gamma, beta  = update_weights_bias(weights, bias, gamma, beta, grad_weights, grad_bias, learning_rate,grad_gamma,grad_beta,do_batchNorm)
  
      # Update the learning rate
      current_iteration = epoch * updates_per_epoch + batch
      learning_rate = cyclical_update(current_iteration, step_size, eta_min, eta_max)

    # Get weighted averages
    if do_batchNorm:
      if batch == 0:  # First minibatch
        mean_average = mean_list
        variance_average = variance_list
      else:
        mean_average, variance_average = calculate_weighted_average(mean_list,variance_list,alpha)
      mean_list = mean_average
      variance_list = variance_average
    
    if do_plot:
      probs_train = forward_pass(data_train, weights, bias, do_batchNorm=do_batchNorm, gamma=gamma, beta=beta, mean=mean_list, var=variance_list)[0][-1]
      probs_validation = forward_pass(data_val,weights, bias, do_batchNorm=do_batchNorm, gamma=gamma, beta=beta,mean=mean_list, var=variance_list)[0][-1]
      probs_test = forward_pass(data_test,weights, bias, do_batchNorm=do_batchNorm, gamma=gamma, beta=beta, mean=mean_list, var=variance_list)[0][-1]

      train_loss.append(get_loss(data_train, labels_train,probs_train))
      vaidation_loss.append(get_loss(data_val, labels_val,probs_validation))
      test_loss.append(get_loss(data_test, labels_test, probs_test))

      train_accuracy.append(compute_accuracy(data_train, labels_train, weights, bias,probs_train))
      validation_accuracy.append(compute_accuracy(data_val, labels_val, weights, bias,probs_validation))
      test_accuracy.append(compute_accuracy(data_test, labels_test, weights, bias,probs_test))

      train_cost.append(compute_cost(data_train, labels_train, weights, bias, reguliser, probs_train))
      validation_cost.append(compute_cost(data_val, labels_val, weights, bias, reguliser, probs_validation))
      test_cost.append(compute_cost(data_test, labels_test, weights, bias, reguliser, probs_test))
      step_list.append(epoch)

      #Randomly shuffle the data
    random_indices = np.random.permutation(data_train.shape[1])
    data_train = data_train[:, random_indices]
    labels_train = labels_train[:, random_indices]
      
  if do_plot:
    do_plotting(train_loss, vaidation_loss, test_loss, train_accuracy, validation_accuracy, test_accuracy, train_cost, validation_cost, test_cost, step_list, name_of_file=name_of_file)

  # Get the final accuracy of the test data
  probs_test = forward_pass(data_test, weights, bias, do_batchNorm=do_batchNorm, gamma=gamma, beta=beta, mean=mean_list, var=variance_list)[0][-1]
  final_accuracy = compute_accuracy(data_test, labels_test, weights, bias, probs_test)
  return weights, bias, final_accuracy

def calculate_weighted_average(mean_list, var_list, alpha):
  mean_av = mean_list.copy()  # Start with values from mean_list
  var_av = var_list.copy()    # Start with values from var_list

  for i in range(len(mean_list)):
    new_mean = alpha * mean_av[i] + (1 - alpha) * mean_list[i]
    new_var = alpha * var_av[i] + (1 - alpha) * var_list[i]
    mean_av[i] = new_mean  # Update mean_av in place
    var_av[i] = new_var    # Update var_av in place
  return mean_av, var_av

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

def compute_gradients_slow(data, labels, weights, bias, gamma, beta, reguliser, do_batchNorm=False, h=1e-5):    
  grad_weights = []
  grad_bias = []
  grad_gamma = []
  grad_beta = []
  for j in range(len(bias)):
      grad_bias_layer = np.zeros(bias[j].shape)
      for i in tqdm(range(grad_bias_layer.shape[0])):
          for k in range(grad_bias_layer.shape[1]):
              b_try = [np.copy(x) for x in bias]
              b_try[j][i, k] -= h
              probs = forward_pass(data, weights, b_try,gamma, beta, None, None, do_batchNorm)[0][-1]
              c1 = compute_cost(data, labels, weights, b_try, reguliser, probs)
              b_try = [np.copy(x) for x in bias]
              b_try[j][i, k] += h
              probs = forward_pass(data, weights, b_try,gamma, beta, None, None, do_batchNorm)[0][-1]
              c2 = compute_cost(data, labels, weights, b_try, reguliser, probs)
              grad_bias_layer[i, k] = (c2 - c1) / (2 * h)
      grad_bias.append(grad_bias_layer)
  
  for j in range(len(weights)):
      grad_weights_layer = np.zeros(weights[j].shape)
      for i in tqdm(range(grad_weights_layer.shape[0])):
          for k in range(grad_weights_layer.shape[1]):
              w_try = [np.copy(x) for x in weights]
              w_try[j][i, k] -= h
              probs = forward_pass(data, w_try, bias,gamma, beta, None, None, do_batchNorm)[0][-1]              
              c1 = compute_cost(data, labels, w_try, bias,reguliser, probs)
              w_try = [np.copy(x) for x in weights]
              w_try[j][i, k] += h
              probs = forward_pass(data, w_try, bias,gamma, beta, None, None, do_batchNorm)[0][-1]
              c2 = compute_cost(data, labels, w_try, bias, reguliser, probs)
              grad_weights_layer[i, k] = (c2 - c1) / (2 * h)
      grad_weights.append(grad_weights_layer)
      print("Done with weights and bias")
  
  if do_batchNorm:
      for j in tqdm(range(len(gamma))):
          grad_gamma_layer = np.zeros(gamma[j].shape)
          for i in range(grad_gamma_layer.shape[0]):
              for k in range(grad_gamma_layer.shape[1]):
                  g_try = [np.copy(x) for x in gamma]
                  g_try[j][i, k] -= h
                  probs = forward_pass(data, weights, bias, g_try, beta, None, None, do_batchNorm)[0][-1]
                  c1 = compute_cost(data, labels, weights, bias, reguliser, probs)
                  g_try = [np.copy(x) for x in gamma]
                  g_try[j][i, k] += h
                  probs = forward_pass(data, weights, bias, g_try, beta, None, None, do_batchNorm)[0][-1]
                  c2 = compute_cost(data, labels, weights, bias, reguliser, probs)
                  grad_gamma_layer[i, k] = (c2 - c1) / (2 * h)
          grad_gamma.append(grad_gamma_layer)
      
      for j in tqdm(range(len(beta))):
          grad_beta_layer = np.zeros(beta[j].shape)
          for i in range(grad_beta_layer.shape[0]):
              for k in range(grad_beta_layer.shape[1]):
                  bt_try = [np.copy(x) for x in beta]
                  bt_try[j][i, k] -= h
                  probs1 = forward_pass(data, weights, bias, gamma, bt_try, None, None, do_batchNorm)[0][-1]
                  c1 = compute_cost(data, labels, weights, bias, reguliser, probs1)
                  bt_try = [np.copy(x) for x in beta]
                  bt_try[j][i, k] += h
                  probs2 = forward_pass(data, weights, bias, gamma, bt_try, None, None, do_batchNorm)[0][-1]
                  c2 = compute_cost(data, labels, weights, bias, reguliser, probs2)
                  grad_beta_layer[i, k] = (c2 - c1) / (2 * h)
          grad_beta.append(grad_beta_layer)
  
  return grad_weights, grad_bias, grad_gamma, grad_beta

def compareGradients(data, labels, weights, bias, gamma, beta, reguliser, do_batchNorm=False, h=1e-5):
   # Takes the absolute difference between the gradients computed by the two methods
  layers, scores_list, s_hat, mean, variance  = forward_pass(data, weights, bias, gamma, beta, None, None, do_batchNorm)
  grad_weights_fast, grad_bias_fast,grad_gamma_fast,grad_beta_fast = back_pass(layers, labels, weights, reguliser, layers[-1], scores_list, s_hat, gamma, mean, variance, do_batchNorm)
  grad_gamma = None
  grad_beta = None

  print("Computing gradients slow...")
  grad_weights, grad_bias, grad_gamma, grad_beta = compute_gradients_slow(data, labels, weights, bias, gamma, beta, reguliser, do_batchNorm, h)

  print("Comparing gradients...")
  for i in range(len(grad_weights)):
    print("Gradient weights layer ", i, ": ", np.max(np.abs(grad_weights[i] - grad_weights_fast[i])))
    print("Gradient bias layer ", i, ": ", np.max(np.abs(grad_bias[i] - grad_bias_fast[i])))
  if do_batchNorm:
    for i in range(len(grad_gamma)):
      print("Gradient gamma layer ", i, ": ", np.max(np.abs(grad_gamma[i] - grad_gamma_fast[i])))
      print("Gradient beta layer ", i, ": ", np.max(np.abs(grad_beta[i] - grad_beta_fast[i])))
  else:
    print("No batch normalisation")
  print("Done!")

def lamba_search(data_train, data_val, data_test, weights, bias, labels_train, labels_val, labels_test, gamma,beta, do_batchNorm):
  config = {
      'data_train': data_train,
      'data_val': data_val,
      'data_test': data_test,
      'weights': weights,
      'bias': bias,
      'labels_train': labels_train,
      'labels_val': labels_val,
      'labels_test': labels_test,
      'learning_rate': 0.0001,
      'batch_size': 100,
      'cycles': 1,
      'do_plot': False,
      'do_batchNorm': do_batchNorm,
      'name_of_file': "LabchNorm=False",
      'gamma': None,
      'beta': None,
      'alpha': 0.9
  }
  
  print("How many values would you like to search for? Number > 3")
  number_of_values = int(input())
  if number_of_values < 3:
    print("Number of values must be greater than 3")
    sys.exit(0)
  lamda_list = list()
  acc_list = list() # The index of acc_list corresponds to the index of lamda_list

  for i in range(number_of_values):
    print("Iteration ", i+1, " of ", number_of_values)
    random_number = np.random.uniform(-5,-1)
    reguliser = 10**random_number
    lamda_list.append(reguliser)
    print("Reguliser: ", reguliser)
    config['reguliser'] = reguliser
    _, _, final_accuracy = sgd_minibatch(**config)
    acc_list.append(final_accuracy)
  
  # The highest accuracy is the best 
  print("The highest accuracy is: ", np.max(acc_list))
  print("The index of the highest accuracy is: ", np.argmax(acc_list))
  print("The corresponding lambda is: ", lamda_list[np.argmax(acc_list)])

  print("Do you want to narrow the search? (y/n)")
  answer = input()
  if answer == "n":
    return lamda_list[np.argmax(acc_list)]
  elif answer == "y":
    narrow_lambda_list = list()
    # Grab the three highst accuracies and their corresponding lambdas
    highest_acc = np.max(acc_list)
    highest_acc_index = np.argmax(acc_list)
    highest_acc_lambda = lamda_list[highest_acc_index]
    acc_list.pop(highest_acc_index)
    lamda_list.pop(highest_acc_index)

    second_highest_acc = np.max(acc_list)
    second_highest_acc_index = np.argmax(acc_list)
    second_highest_acc_lambda = lamda_list[second_highest_acc_index]
    acc_list.pop(second_highest_acc_index)
    lamda_list.pop(second_highest_acc_index)

    # Uniformly at random choose a lambda between the two lambdas
    for i in range(3):
      random_number = np.random.uniform(second_highest_acc_lambda, highest_acc_lambda)
      narrow_lambda_list.append(random_number)
    
    print("Starting narrow search...")
    # Start searching for the best lambda
    for i in range(len(narrow_lambda_list)):
      print("Iteration ", i+1, " of ", len(narrow_lambda_list))
      reguliser = narrow_lambda_list[i]
      print("Reguliser: ", reguliser)
      config['reguliser'] = reguliser
      _, _, final_accuracy = sgd_minibatch(**config)
      acc_list.append(final_accuracy)
    
    # The highest accuracy is the best
    print("The highest accuracy is: ", np.max(acc_list))
    print("The index of the highest accuracy is: ", np.argmax(acc_list))
    print("The corresponding lambda is: ", narrow_lambda_list[np.argmax(acc_list)])
    return narrow_lambda_list[np.argmax(acc_list)]

def main():
  #Getting started
  data_train, labels_train, data_val, labels_val, data_test, labels_test, labels = read_data(5000)  

  # Preprocessing
  data_train, data_val, data_test = normalise_all(data_train, data_val, data_test)
  labels_train, labels_val, labels_test = encode_all(labels_train, labels_val, labels_test)

  print("Do you want to do batch normalisation? (y/n)")
  answer = input()
  if answer == "y":
     do_batchNorm = True
  else:
    do_batchNorm = False
  
  Sigma = None
  print("Do you want to have sigma? (y/n)")
  answer = input()
  if answer == "y":
    print("What sigma do you want?")
    Sigma = float(input())
  else:
    Sigma = None

  
  # Initialising the network
  weights, bias, gamma, beta = init_network(data_train, [50, 50,10], he =True, Sigma =Sigma, do_batchNorm=do_batchNorm)


  # Comparing gradients
  print("Do you want to compare the gradients? (y/n)")
  answer = input()
  if answer == "y":
    compareGradients(data_train[:, 0:100], labels_train[:, 0:100], weights, bias, gamma, beta, 0, do_batchNorm, h=1e-5)
  
  
  print("Do you want to train the network? (y/n)")
  answer = input()
  if answer == 'n':
    sys.exit(0)
  else: 
    config = {
      'data_train': data_train,
      'data_val': data_val,
      'data_test': data_test,
      'weights': weights,
      'bias': bias,
      'labels_train': labels_train,
      'labels_val': labels_val,
      'labels_test': labels_test,
      'learning_rate': 0.0001,
      'reguliser': 0.005,
      'batch_size': 100,
      'cycles': 2,
      'do_plot': True,
      'do_batchNorm': do_batchNorm,
      'name_of_file': "Lab[50, 30, 20, 20, 10, 10, 10, 10], he=True, Sigma=None, do_batchNorm=True",
      'gamma': gamma,
      'beta': beta,
      'alpha': 0.9
  }
 
  print("Do you want to do a lamba search? (y/n)")
  answer = input()
  if answer == "y":
    good_lambda = lamba_search(data_train, data_val, data_test, weights, bias, labels_train, labels_val, labels_test, do_batchNorm)
    config['reguliser'] = good_lambda
  else:
    config['reguliser'] = 0.005
  
  sgd_minibatch(**config)

if __name__ == "__main__":
  main()