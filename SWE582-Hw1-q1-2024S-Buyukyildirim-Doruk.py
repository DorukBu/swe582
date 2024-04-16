# NAME:       Doruk Büyükyıldırım
# STUDENT ID: 2022719033
# DATE:       16.04.2024
# SUMMARY:    SPRING 2024 SWE582 - SWE 582: Sp. Tp. Machine Learning for Data Analytic Homework #1 Question 2 Due April 16, 2024 by 23:59
#             This Python code does the following:
#             -> Load 2 different data sets* of 3-D (including bias terms) data and correstponding labels of 2 classes (minus 1 & plus 1)
#             -> Train 2 perceptrons using PLA for both datasets
#             -> Print out number of iterations, final weight vector and plot the datapoints as well as decision boundary
#             -> Train perceptrons on same dataset (small dataset) and print out final model vector each time for comparison
#             * The datasets in .npy format should be placed inside the working directory
#             Acknowledgement: Generative AI is used to generate plotting style and functions

import numpy as np
from random import choice
import matplotlib.pyplot as plt
import os

## Get the current working directory
cwd = os.getcwd()

## Construct the relative paths to .npy files
data_large_path = os.path.join(cwd, 'data_large.npy')
label_large_path = os.path.join(cwd, 'label_large.npy')
data_small_path = os.path.join(cwd, 'data_small.npy')
label_small_path = os.path.join(cwd, 'label_small.npy')

## Load the .npy files using the relative path
data_large = np.load(data_large_path)
label_large = np.load(label_large_path)
data_small = np.load(data_small_path)
label_small = np.load(label_small_path)


def train_perceptron(training_data):
    '''
    Train a perceptron model using the Perceptron Learning Algorithm (PLA).
    
    :param training_data: A list containing the training data and labels. 
                          training_data[0] contains the data points and 
                          training_data[1] contains the corresponding labels.
                          Labels are +1/-1.
    :return: learned model vector and the number of iterations required for convergence.
    '''
    X = training_data[0]
    y = training_data[1]
    model_size = X.shape[1]
    w = np.zeros(model_size)#np.random.rand(model_size)
    iteration = 1
    while True:
        # compute results according to the hypothesis
        results = np.matmul(X, w)

        # get incorrect predictions (you can get the indices)
        misclassified_indices = np.where(y * results <= 0)[0]

        # Check the convergence criteria (if there are no misclassified
        # points, the PLA is converged and we can stop.)
        if len(misclassified_indices) == 0:
            break

        # Pick one misclassified example.
        misclassified_index = choice(misclassified_indices)

        # Update the weight vector with perceptron update rule
        w += y[misclassified_index] * X[misclassified_index]

        iteration += 1

    return w, iteration


def plot_decision_boundary(data, labels, model):
    '''
    Plot decision boundary given data points, corresponding labels, and a model vector.
    
    :param data: An array containing data points.
    :param labels: An array containing corresponding labels for the data points.
                   Labels are expected to be +1 or -1.
    :param model: A model vector representing the decision boundary.
    
    :return: None
    '''
    # Initialize empty arrays for each class
    class_1 = []
    class_minus_1 = []

    # Iterate over labels and data, splitting them into class_1 and class_minus_1
    for i, lbl in enumerate(labels):
        if lbl == 1:
            class_1.append(data[i])
        elif lbl == -1:
            class_minus_1.append(data[i])

    # Convert lists to numpy arrays
    class_1 = np.array(class_1)
    class_minus_1 = np.array(class_minus_1)

    plt.scatter(class_1[:, 1], class_1[:, 2], marker='o', label='Class 1', alpha=1, edgecolors='k', facecolor='r')
    plt.scatter(class_minus_1[:, 1], class_minus_1[:, 2], marker='x', label='Class -1', alpha=1, facecolor='g')

    x_min, x_max = data[:, 1].min() - 0.2, data[:, 1].max() + 0.2
    y_min, y_max = data[:, 2].min() - 0.2, data[:, 2].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    # Adjust the indices for the model vector
    Z = np.sign(np.dot(np.c_[np.ones_like(xx.ravel()), xx.ravel(), yy.ravel()], model))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, colors=['green', 'red'], alpha=0.05)
    plt.legend()


## Train model on small dataset, print out the iteration count, plot the dataset, and model vector
rnd_data = [data_small,label_small]

trained_model, iteration = train_perceptron(rnd_data)
print("Number of iterations: {}".format(iteration))

plot_decision_boundary(data_small, label_small, trained_model)
plt.show()
print("The weight vector:")
print(trained_model)


## Train model on large dataset, print out the iteration count, plot the dataset, and model vector
rnd_data = [data_large,label_large]

trained_model, iteration = train_perceptron(rnd_data)
print("Number of iterations: {}".format(iteration))

plot_decision_boundary(data_large, label_large, trained_model)
plt.show()
print("The weight vector:")
print(trained_model)


## Train model on small dataset multiple times to compare model vectors
rnd_data = [data_small,label_small]

for i in range(10):
    trained_model, iteration = train_perceptron(rnd_data)
    print("Training {}: Number of iterations: {}, Model Vector: {}".format(i+1, iteration, trained_model))
