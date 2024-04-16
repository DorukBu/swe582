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

import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
import os

## Get the current working directory
cwd = os.getcwd()

## Construct the relative paths to .npy files
data_path = os.path.join(cwd, 'Rice_Cammeo_Osmancik.arff')

arff_file = arff.loadarff(data_path)

df = pd.DataFrame(arff_file[0])

# Encode labels
df['Class'] = pd.Categorical(df['Class']).codes

# Split features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Normalize features
X_normalized = (X - X.mean()) / X.std()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

def sigmoid(z):
    """
    Compute the sigmoid function of input z.

    Parameters:
    z (array_like): Input value(s) to the sigmoid function.

    Returns:
    array_like: The sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, theta, alpha, num_iterations):
    """
    Perform logistic regression using full batch gradient descent.

    Parameters:
    X (array_like): Input features of shape (m, n), where m is the number of samples and n is the number of features.
    y (array_like): Target labels of shape (m,).
    theta (array_like): Initial parameters (weights) of shape (n,).
    alpha (float): Learning rate.
    num_iterations (int): Number of iterations for gradient descent.

    Returns:
    array_like: Optimized parameters (weights) after training.
    """
    m = len(y)
    for i in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        theta -= alpha * gradient
    return theta

def regularized_logistic_regression(X, y, theta, alpha, lambd, num_iterations):
    """
    Perform regularized logistic regression using full batch gradient descent.

    Parameters:
    X (array_like): Input features of shape (m, n), where m is the number of samples and n is the number of features.
    y (array_like): Target labels of shape (m,).
    theta (array_like): Initial parameters (weights) of shape (n,).
    alpha (float): Learning rate.
    lambd (float): Regularization parameter.
    num_iterations (int): Number of iterations for gradient descent.

    Returns:
    array_like: Optimized parameters (weights) after training.
    """
    m = len(y)
    for i in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = (np.dot(X.T, (h - y)) + lambd * theta) / m
        gradient[0] -= lambd * theta[0]  # Exclude regularization for bias term
        theta -= alpha * gradient
    return theta

def full_batch_gradient_descent(X, y, theta, alpha, num_iterations):
    """
    Perform full batch gradient descent for logistic regression.

    Parameters:
    X (array_like): Input features of shape (m, n), where m is the number of samples and n is the number of features.
    y (array_like): Target labels of shape (m,).
    theta (array_like): Initial parameters (weights) of shape (n,).
    alpha (float): Learning rate.
    num_iterations (int): Number of iterations for gradient descent.

    Returns:
    array_like: Optimized parameters (weights) after training.
    """
    return logistic_regression(X, y, theta, alpha, num_iterations)

def stochastic_gradient_descent(X, y, theta, alpha, num_iterations):
    """
    Perform stochastic gradient descent for logistic regression.

    Parameters:
    X (array_like): Input features of shape (m, n), where m is the number of samples and n is the number of features.
    y (array_like): Target labels of shape (m,).
    theta (array_like): Initial parameters (weights) of shape (n,).
    alpha (float): Learning rate.
    num_iterations (int): Number of iterations for gradient descent.

    Returns:
    array_like: Optimized parameters (weights) after training.
    """
    m = len(y)
    for i in range(num_iterations):
        for j in range(m):
            rand_index = np.random.randint(m)
            X_i = X[rand_index:rand_index+1]
            y_i = y[rand_index:rand_index+1]
            z = np.dot(X_i, theta)
            h = sigmoid(z)
            gradient = np.dot(X_i.T, (h - y_i))
            theta -= alpha * gradient
    return theta

def k_fold_cross_validation(X, y, k, alpha, lambd_values, num_iterations):
    """
    Perform k-fold cross-validation to determine the best regularization parameter for logistic regression.

    Parameters:
    X (array_like): Input features of shape (m, n), where m is the number of samples and n is the number of features.
    y (array_like): Target labels of shape (m,).
    k (int): Number of folds for cross-validation.
    alpha (float): Learning rate.
    lambd_values (array_like): List of regularization parameter values to evaluate.
    num_iterations (int): Number of iterations for gradient descent.

    Returns:
    float: Best regularization parameter determined by cross-validation.
    """
    m = len(y)
    fold_size = m // k
    accuracies = []

    for lambd in lambd_values:
        accuracy_sum = 0.0

        for i in range(k):
            X_train = np.concatenate((X[:i * fold_size], X[(i + 1) * fold_size:]), axis=0)
            y_train = np.concatenate((y[:i * fold_size], y[(i + 1) * fold_size:]), axis=0)
            X_val = X[i * fold_size:(i + 1) * fold_size]
            y_val = y[i * fold_size:(i + 1) * fold_size]

            theta = np.zeros(X_train.shape[1])
            theta = regularized_logistic_regression(X_train, y_train, theta, alpha, lambd, num_iterations)

            y_pred = predict(X_val, theta)
            accuracy = np.mean(y_pred == y_val)
            accuracy_sum += accuracy

        accuracies.append(accuracy_sum / k)

    best_lambd_index = np.argmax(accuracies)
    best_lambd = lambd_values[best_lambd_index]
    return best_lambd

def predict(X, theta):
    """
    Predict the labels for input features using logistic regression model parameters.

    Parameters:
    X (array_like): Input features of shape (m, n), where m is the number of samples and n is the number of features.
    theta (array_like): Parameters (weights) of the logistic regression model.

    Returns:
    array_like: Predicted labels for the input features.
    """
    prob = sigmoid(np.dot(X, theta))
    return (prob >= 0.5).astype(int)

# Initialize parameters
alpha = 0.01
num_iterations = 1000
lambd_values = [0.01, 0.1, 1, 10, 100]
k = 5

# Perform k-fold cross-validation to determine the best regularization parameter
best_lambd = k_fold_cross_validation(X, y, k, alpha, lambd_values, num_iterations)

# Train with best regularization parameter
theta_gd = np.zeros(X.shape[1])
theta_gd = full_batch_gradient_descent(X, y, theta_gd, alpha, num_iterations)
theta_reg_gd = np.zeros(X.shape[1])
theta_reg_gd = regularized_logistic_regression(X, y, theta_reg_gd, alpha, best_lambd, num_iterations)

# Report training and test performance
train_accuracy_gd = np.mean(predict(X, theta_gd) == y)
train_accuracy_reg_gd = np.mean(predict(X, theta_reg_gd) == y)

print("The best regularization parameter is: {}".format(best_lambd))
print("Training Performance:")
print("Logistic Regression (GD): {:.2f}%".format(train_accuracy_gd * 100))
print("Regularized Logistic Regression (GD): {:.2f}%".format(train_accuracy_reg_gd * 100))

### WARNING the functions written for SDG does not stop executing due to errors
# Train SGD with best regularization parameter
theta_sgd = np.zeros(X.shape[1])
theta_sgd = stochastic_gradient_descent(X, y, theta_gd, alpha, num_iterations)

# Report training and test performance
train_accuracy_sgd = np.mean(predict(X, theta_gd) == y)

print("The best regularization parameter is: {}".format(best_lambd))
print("Training Performance:")
print("Logistic Regression (SGD): {:.2f}%".format(train_accuracy_sgd * 100))
