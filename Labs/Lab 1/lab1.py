""" 
  Author: Rakin Ali
  Date: 27/03/2022
  Description: Lab 1 - Backpropagation

"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm as tqdm

# Paths 
DATAPATH = "Datasets/cifar-10-batches-py/"
LENGTH = 1024 # Pixels per image
WIDTH = 32 # Width of image
D_BATCH = ['data_batch_1', 'data_batch_2',
           'data_batch_3', 'data_batch_4', 'data_batch_5']
T_BATCH = 'test_batch'


# Load data
def LoadBatch(filename):
  with open(filename, 'rb') as fo:
      dict = pickle.load(fo, encoding='latin1')
  return dict

# Preprocess data"""
def normalise(data,mean,std):
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
  data = np.float64(data) # When calculating the mean and std, use float64
  data = np.transpose(data) # Converting n x d to d x n where columns = picture row = image
  #print(data.shape)
  mean = np.mean(data, axis=1, keepdims=True)
  std = np.std(data, axis=1, keepdims=True)
  
  # Normalizing the data 
  data = (data - mean) / std

  return np.array(data), mean, std

def random_weight_bias_init(data,labels):
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

  #print("Weight shape:",weight.shape)
  #print("Bias shape", bias.shape)

  return weight, bias


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


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




def compute_accuracy(data, truth_labels, weight, bias):
  """@docstring:
  Compute the accuracy of the network's predictions.
  Inputs:
  - X: A numpy array of shape (D, N) containing the image data.
  - y: A numpy array of shape (N,) containing the training labels.
  - W: A numpy array of shape (K, D) containing the weights.
  - b: A numpy array of shape (K, 1) containing the biases.
  Returns:
  - acc: The accuracy of the network.
  """
  # Getting the predicted labels
  p = evaluate_classifier(data, weight, bias)
  predicted_labels = np.argmax(p, axis=0)

  # Calculating the accuracy
  acc = np.sum(predicted_labels == truth_labels) / len(truth_labels)
  print("Accuracy: ", acc * 100, "%")

  return acc


def computeCost(data,labels,weights,bias,lmbd):
  """@docstring:
  Compute the cost function. The cost function is the average of the loss functions of the training images.
  Inputs:
  - X: A numpy array of shape (D, N) containing the image data.
  - y: A numpy array of shape (N,) containing the training labels.
  - W: A numpy array of shape (K, D) containing the weights.
  - b: A numpy array of shape (K, 1) containing the biases.
  - lmb: The regularization strength.
  Returns:
  - J: The cost function.
  """
  # Getting the probabilities
  probabilities = evaluate_classifier(data, weights, bias)

  # Calculating the loss function
  loss = -np.log(probabilities[labels, range(len(labels))])

  # Calculating the cost function
  J = np.sum(loss) / len(labels) + lmbd * np.sum(weights ** 2)

  


def ComputeGradients(data,labels,probabilities,weight,lmb):
  """@docstring:
  Compute the gradients of the loss function with respect to the parameters W and b. Inpsired by lecture 3 slide 103 and 104.

  Inputs:
  - X: A numpy array of shape (D, N) containing the image data.
  - y: A numpy array of shape (N,) containing the training labels.
  - P: A numpy array of shape (K, N) containing the probabilities.
  - W: A numpy array of shape (K, D) containing the weights.
  - lmb: The regularization strength.
  Returns:
  - grad_W: A numpy array of shape (K, D) containing the gradients of the loss function with respect to W.
  - grad_b: A numpy array of shape (K, 1) containing the gradients of the loss function
  with respect to b.
  """

  g = -(labels - probabilities) # (K,N)
  gradient_weight = np.dot(g, data.T)/ data.shape[1] + 2 * lmb * weight # (K,N) * (N,D) = (K,D)    
  gradient_bias = np.sum(g, axis=1, keepdims=True)/ data.shape[1] # (K,1)
  return gradient_weight, gradient_bias




# main function
if __name__ == "__main__":

  # Getting started
  data_train, labels_train, data_val, labels_val, data_test, labels_test, labels = getting_started()

  # Preprocessing the data
  data_train, mean_train, std_train = normalise(data_train, None, None)
  data_val, mean_val, std_val = normalise(data_val, None, None)
  data_test, mean_test, std_test  = normalise(data_test, None, None)

  # Initializing the weights and biases
  weight, bias = random_weight_bias_init(data_train, labels)

  # Evaluating the classifier
  p = evaluate_classifier(data_train, weight, bias)

  # Computing the accuracy
  acc = compute_accuracy(data_train, labels_train, weight, bias)










  

  




  




  

