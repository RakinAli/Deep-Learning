# DD2424-Deep-Learning

## Assignment 1: 
In this assignment, an one layer network was trained and tested. The training was done using mini-batch gradient descent on a cost function that computes the cross-entropy loss of the classifier on the CIFAR-10 dataset.
#

## Assignment 2: 
This assignment is a further implementation of assignment 1. Now, a two-layer neural network with multiple outputs for image classification on the CIFAR-10 dataset was trained and evaluated. This was accomplished with, mini-batch gradient descent, a popular optimization algorithm for training neural networks. To optimize the network, a composite cost function is employed, which computes the cross-entropy loss by applying the classifier to labeled training data. Furthermore, the optimization process incorporates an L2 regularization term applied to the weight matrix.
#

## Assignment 3:
This assignment is a further implementation of assignment 2. Now, a k-layer neural network with multiple outputs for image classification on the CIFAR-10 dataset was trained and evaluated. In this assignment, batch normalization is also incorporated into the k-layer network both for training and testing. To accomplish this, mini-batch gradient descent was used.
#

## Assignment 4:
In this assignment, a Recurrent Neural Network (RNN) was trained on how to create English text letter by letter. A basic RNN model with outputs was used. To achieve this, text from the book "The Goblet of Fire" authored by J.K. Rowling was used as the dataset. For the optimization process, a variant of Stochastic Gradient Descent (SGD) called AdaGrad was used.
#

## Project:
The project showcases the exploration of Convolutional Neural Networks (CNNs) with the CIFAR-10 dataset. The goal was to optimize accuracy by testing different architectures. Three VGG models (VGG1, VGG2, and VGG3), incorporating techniques like dropout, weight decay, and data augmentation were built and extended. Deeper models proved more effective, avoiding gradient issues. Further, batch normalization, dropout rates, and optimizers observing Adam's superiority over SGD were studied. AdamW struggled with VGG3 due to inappropriate weight decay. The order of batch normalization and dropout affected accuracy; prior batch normalization slightly outperformed. To exceed 90% accuracy, a ResNet was introduced and explored
