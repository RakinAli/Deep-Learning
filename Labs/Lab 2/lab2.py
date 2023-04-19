""" 
  Author: Rakin Ali
  Date:    27/03/2022
  Description: Lab 1 - Backpropagation
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import sys

PATH = "Datasets/cifar-10-batches-py/"
batches= ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]


def LoadBatch(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def getting_started_all():
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
            data_train = np.vstack((data_train, batch['data'])) # Stack the data vertically
            labels_train = np.hstack((labels_train, batch['labels'])) # Stack the labels horizontally
                
    # Read the test data
    batch = LoadBatch(PATH + 'test_batch')
    data_test = batch['data']
    labels_test = batch['labels']

    # Create the validation data
    random_indices = np.random.choice(data_train.shape[0], 5000, replace=False)
    data_val = data_train[random_indices]
    labels_val = labels_train[random_indices]  
    data_train = np.delete(data_train, random_indices, axis=0) # Delete the validation data from the training data
    labels_train = np.delete(labels_train, random_indices, axis=0) # Delete the validation labels from the training labels
    
    # Grabbing the labels names
    labels = LoadBatch(PATH + 'batches.meta')['label_names']

    return data_train, labels_train, data_val, labels_val, data_test, labels_test, labels


def getting_started():
    """@docstring:
    This function will help you get started on the CIFAR-10 dataset. 
    Reads the bathces 0 and batches 1 to do training and validation
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


def get_scores(data, weights, bias):
    """@docstring:
    Compute the scores
    Returns:
    - s : a list of numpy arrays containing the scores
    """
    s = weights @ data + bias
    return s


def relu(s):
    """@docstring:
    Compute the ReLU activation function
    Returns:
    - h : a list of numpy arrays containing the ReLU activation function
    """
    h = np.maximum(0, s)
    return h


def forward_pass(data, weights, bias):
    # Note: Inpired by the forward pass in the lecture notes of the course
    # Also inpspired by other students from this course,
    # Previuosly I had a different implementation of the forward pass but it was hard coded to 2 layers
    # Afterwards when I saw the implementation of other students I decided to implement it in a more general way
    # Therefore other parts, backpropagation was modified to work with this implementation
    # 
    """@docstring:
    Compute the forward pass for all the layers
    Returns:
    - output_layer : a list of numpy arrays containing the output of each layer
    - scores_list : a list of numpy arrays containing the scores of each layer
    """

    output_layer = list()
    scores_list = list()

    # First layer
    output_layer.append(np.copy(data)) # Copying the data to the output layer, the first layer 
    scores_list.append(get_scores(data, weights[0], bias[0])) # Getting the scores for the first layer

    # Iterating over the hidden layers
    for i in range(1, len(weights)):
        output_layer.append(relu(scores_list[i-1])) # Getting the output of the previous layer
        scores_list.append(get_scores(output_layer[-1], weights[i], bias[i]))  

    return output_layer, scores_list


def softmax(s):
    """@docstring: 
    Compute the softmax activation function
    Input: s - a numpy array of shape (K, N) containing the scores
    Returns:
    - p : a numpy array of shape (K, N) containing the probabilities
    """
    p = np.exp(s) / np.sum(np.exp(s), axis=0)
    return p


def get_loss(data,labels,weight,bias):
    _, scores_list = forward_pass(data, weight, bias)
    p = softmax(scores_list[-1])
    loss = np.sum(-np.log(np.sum(labels*p, axis=0)))
    loss = loss/data.shape[1]
    return loss


def compute_cost(data,labels,weight,bias,reguliser):
    loss = get_loss(data,labels,weight,bias)
    cost = loss + reguliser * np.sum([np.sum(np.square(w)) for w in weight])
    return cost


def compute_accuracy(data,labels,weight,bias):
    _, scores_list = forward_pass(data, weight, bias)
    p = softmax(scores_list[-1]) # get s2 from the forward pass
    pred = np.argmax(p, axis=0) # get the predicted class
    acc = np.sum(pred == np.argmax(labels, axis=0)) / data.shape[1]
    return acc

def backward_pass(data, labels, weight, reg, probs):
    # Note: Inpired by the backward pass in the lecture notes of the course
    # Also inpspired by other students from this course, as the forward pass was inspired by them,
    # the  backward pass needed to be adjusted to the new implementation of the forward pass
    """@docstring:
    Compute the backward pass for all the layers
    Returns:
    - gradient_weights : a list of numpy arrays containing the gradients of the weights
    - gradient_bias : a list of numpy arrays containing the gradients of the bias
    """

    gradient_weights = list() # list of gradients for weights from last to first layer
    gradient_bias = list() # list of gradients for bias from last to first layer

    # Last layer gradient calculation
    g = -(labels - probs) # gradient of the loss function with respect to the scores of the last layer
    weight_gradient = (g @ data[-1].T) / data[0].shape[1] + 2 * reg * weight[-1] # gradient of the loss function with respect to the weights of the last layer
    bias_gradient = np.sum(g, axis=1, keepdims=True) / data[0].shape[1] # gradient of the loss function with respect to the bias of the last layer
    gradient_weights.append(weight_gradient)
    gradient_bias.append(bias_gradient)

    # Rest of the layers gradient calculation
    for i in range(len(data) -1):
        g = weight[-i+1].T @ g 
        ind = np.copy(data[-i+1]) # copy the output of the previous layer
        ind[ind > 0] = 1 # ReLU derivative
        g = g * ind # gradient of the loss function with respect to the scores of the current layer
        weight_gradient = (g @ data[-i].T)/ data[0].shape[1] + 2 * reg * weight[-i] # gradient of the loss function with respect to the weights of the current layer
        bias_gradient = np.sum(g, axis=1, keepdims=True) / data[0].shape[1] # gradient of the loss function with respect to the bias of the current layer
        gradient_weights.append(weight_gradient)
        gradient_bias.append(bias_gradient)

    # Reversing the lists to get the gradients from first to last layer
    gradient_weights.reverse()
    gradient_bias.reverse()
    return gradient_weights, gradient_bias

def update_weights_bias(weights, bias, grad_w, grad_b, learning_rate):
    """@docstring:
    Update the weights and bias
    Returns:
    - weights : a list of numpy arrays containing the updated weights
    - bias : a list of numpy arrays containing the updated bias
    """

    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * grad_w[i]
        bias[i] = bias[i] - learning_rate * grad_b[i]

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

def batch_training(data_train, data_val, data_test, weights, bias, labels_train, label_val, labels_test, learning_rate, reguliser, batch_size=100, cycles=2, do_plot=True,filename=""):
    """Quick maths:
    A total circle is 2 * stepsize iterations
    Stepsize is half the number of iterations in a cycle and is defined by us (500)
    Epochs can we either hardcode or calculate as total_updates / how many batches needed to go through the whole dataset
    Number of batches needed is equivalent to number of updates needed. Updating learning rate every iteration
    """
    # Hyperparameters
    eta_min = 0.00001
    eta_max = 0.1
    stepsize = 2 * data_train.shape[1]/batch_size
    total_updates = 2 * stepsize * cycles # Total number of updates
    epochs = int(total_updates / (data_train.shape[1]/batch_size)) # Total number of epochs
    updates_per_epoch = int(data_train.shape[1]/batch_size) # Number of updates per epoch
    
    # For plotting
    update_steps = 0
    train_accuracies_list = list()
    train_best_accuracy = 0
    train_cost_list = list()
    train_loss_list = list()
    # For plotting
    validation_accuracies_list = list()
    validation_cost_list = list()
    validation_loss_list = list()
    # For plotting
    test_accuracies_list = list()
    test_best_accuracy = 0
    test_cost_list = list()
    test_loss_list = list()
    steps_list = list()

    total_iterations = epochs * updates_per_epoch # Total number of iterations

    print("Total number of steps needed: ", total_iterations)
    if do_plot:
        print("Plotting will be done, this may take a while...")
    else:
        print("###Starting training...")
    for epoch in range(epochs):
        for batch in range(int(data_train.shape[1]/batch_size)):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            # Relued Layers and scores
            layers, scores_list = forward_pass(data_train[:,start:end], weights, bias)
            # Softmax
            probs = softmax(scores_list[-1])
            # Backward pass
            grad_w, grad_b = backward_pass(
                layers, labels_train[:, start:end], weights, reguliser, probs)
            # Update weights and bias
            weights, bias = update_weights_bias(weights, bias, grad_w, grad_b, learning_rate)
            # Update learning rate
            current_iteration = epoch * updates_per_epoch + batch
            learning_rate = cyclical_update(current_iteration, stepsize, eta_min, eta_max)
            update_steps += 1
            if update_steps % 20 == 0:
                # round the accuracy to 2 decimal places
                acc_train= round(compute_accuracy(data_train, labels_train, weights, bias), 3)
                train_accuracies_list.append(acc_train)
                if do_plot:
                    steps_list.append(update_steps)
                    train_cost_list.append(compute_cost(data_train, labels_train, weights, bias, reguliser))
                    train_loss_list.append(get_loss(data_train, labels_train, weights, bias))

                    validation_accuracies_list.append(round(compute_accuracy(data_val, label_val, weights, bias), 3)) 
                    validation_cost_list.append(compute_cost(data_val, label_val, weights, bias, reguliser))
                    validation_loss_list.append(get_loss(data_val, label_val, weights, bias))

                    test_accuracies_list.append(round(compute_accuracy(data_test, labels_test, weights, bias), 3))
                    test_cost_list.append(compute_cost(data_test, labels_test, weights, bias, reguliser))
                    test_loss_list.append(get_loss(data_test, labels_test, weights, bias))                
                print("Accuracy: ", acc_train," Update steps: ", update_steps)
    train_best_accuracy = max(train_accuracies_list)

    if do_plot:
        do_plotting(train_accuracies_list, train_loss_list, train_cost_list, validation_accuracies_list, validation_loss_list, validation_cost_list, test_accuracies_list, test_loss_list, test_cost_list, steps_list,name_of_file=filename)
    return train_best_accuracy,
    

def do_plotting(train_accuracies_list, train_loss_list, train_cost_list, validation_accuracies_list, validation_loss_list, validation_cost_list, test_accuracies_list, test_loss_list, test_cost_list, steps_list,name_of_file=""):
    # Plotting
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.plot(steps_list, train_accuracies_list, label="Training accuracy")
    plt.plot(steps_list, validation_accuracies_list, label="Validation accuracy")
    plt.plot(steps_list, test_accuracies_list, label="Test accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(steps_list, train_loss_list, label="Training loss")
    plt.plot(steps_list, validation_loss_list, label="Validation loss")
    plt.plot(steps_list, test_loss_list, label="Test loss")
    plt.title("Loss")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(steps_list, train_cost_list, label="Training cost")
    plt.plot(steps_list, validation_cost_list, label="Validation cost")
    plt.plot(steps_list, test_cost_list, label="Test cost")
    plt.title("Cost")
    plt.legend()
    plt.savefig("Results_pics/" + name_of_file + ".png")
    plt.show()


def large_lamda_search(how_many,data_train, data_val, data_test, labels_train, labels_val, labels_test, weights, bias):
    lamda_list= list()
    accuracy_list = list()
    three_lamdas = list()
    three_accuracy = list()
    for lamdas in range(how_many):
        random_number = np.random.uniform(-5, -1)
        lamda = 10**random_number
        lamda_list.append(lamda)

    lamda_list.sort()
    for reguliser in lamda_list:
        print("Lamda: ", reguliser)
        train_best_accuracy = batch_training(data_train, data_val, data_test, weights, bias, labels_train, labels_val, labels_test, learning_rate=0, reguliser=reguliser, batch_size=1000, cycles=1, do_plot=False, filename="" )
        accuracy_list.append(train_best_accuracy)
    
    # Get the best lamdas and their respective accuracies and store them in a list
    for i in range(3):
        best_lamda = lamda_list[accuracy_list.index(max(accuracy_list))]
        best_accuracy = max(accuracy_list)
        three_lamdas.append(best_lamda)
        three_accuracy.append(best_accuracy)
        accuracy_list.remove(best_accuracy)
        lamda_list.remove(best_lamda) 

    # Store the best lamdas and their 
    with open("data/lamda_search.txt", "w") as f:
        f.write("Lamda: " + str(three_lamdas) + " Accuracy: " + str(three_accuracy))

    return three_lamdas, three_accuracy


def narrow_lamda_search(lamda_list, data_val, data_test, labels_train, labels_val, labels_test, weights, bias):
    accuracy_list = list()
    for reguliser in lamda_list:
        print("Lamda: ", reguliser)
        train_best_accuracy = batch_training(data_train, data_val, data_test, weights, bias, labels_train, labels_val, labels_test, learning_rate=0, reguliser=reguliser, batch_size=1000, cycles=3, do_plot=True, filename="lamda_"+str(reguliser))
        accuracy_list.append(train_best_accuracy)
    
    # Find the best lamda and its accuracy
    best_lamda = lamda_list[accuracy_list.index(max(accuracy_list))]
    best_accuracy = max(accuracy_list)
    print("Best lamda: ", best_lamda, " Best accuracy: ", best_accuracy)

    return best_lamda


if __name__ == '__main__':
    print("Starting the program...")
    print("Are you doing the lamda search? (y/n)")
    lamda_search = input()
    if lamda_search == "y":
        data_train, labels_train, data_val, labels_val, data_test, labels_test, labels = getting_started_all()
    elif lamda_search == "n":
        data_train, labels_train, data_val, labels_val, data_test, labels_test, labels = getting_started()
    else: 
        print("Wrong input, please try again")
        sys.exit()
    
    print("Do you want to do plotting at the end? (y/n)")
    do_plot = input()
    if do_plot == "y":
        do_plot = True
    elif do_plot == "n":
        do_plot = False
    else:
        print("Wrong input, please try again")
        sys.exit()
    
    # Normalising the data
    data_train, data_val, data_test = normalise_all(data_train, data_val, data_test) 

    # One hot encoding the labels
    labels_train, labels_val, labels_test = encode_all(labels_train, labels_val, labels_test)

    # Initializing the weights and bias
    weights, bias = init_weights_bias(data_train, labels_train, hidden_nodes=50)

    if lamda_search == "y":
        # Searching for the best lamda
        three_lamdas, three_accuracy = large_lamda_search(10,data_train, data_val, data_test, labels_train, labels_val, labels_test, weights, bias)
        print("Best lamdas: ", three_lamdas)
        print("Best accuracies: ", three_accuracy)
        best_reg = narrow_lamda_search(three_lamdas, data_val, data_test, labels_train, labels_val, labels_test, weights, bias)
    else:
        best_lamda = 0.01
    # Training the best reguliser
    best_acc = batch_training(data_train, data_val, data_test, weights, bias, labels_train, labels_val, labels_test,
                              learning_rate=0.0, reguliser=0.007700753202063438, batch_size=100, cycles=3, do_plot=True, filename="Results_pics/Maxed_Reg_results.png")

